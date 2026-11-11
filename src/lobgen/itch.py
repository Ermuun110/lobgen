"""Streaming reader/filter for Nasdaq TotalView-ITCH 5.0 binary files.

Wire format: each message is a 2-byte big-endian length followed by that many
bytes. Byte 0 of the message is the type char, bytes 1-2 the stock locate code,
3-4 tracking number, 5-10 a 6-byte nanoseconds-since-midnight timestamp.

A full-day file is 4-18 GB gzipped and holds every Nasdaq symbol, so `filter_file`
makes one streaming pass and keeps only the system-event ('S') messages plus
messages whose locate code belongs to a requested ticker. The output is itself a
valid ITCH stream, small enough to parse in pure Python.
"""

from __future__ import annotations

import gzip
import struct
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from numba import njit

# Message body lengths (excluding the 2-byte length prefix), ITCH 5.0 spec.
MSG_LEN = {
    b"S": 12, b"R": 39, b"H": 25, b"Y": 20, b"L": 26, b"V": 35, b"W": 12,
    b"K": 28, b"J": 35, b"h": 21, b"A": 36, b"F": 40, b"E": 31, b"C": 36,
    b"X": 23, b"D": 19, b"U": 35, b"P": 44, b"Q": 40, b"B": 19, b"I": 50,
    b"N": 20, b"O": 48,
}

_S = ord("S")
_R = ord("R")


@njit(cache=True)
def _scan(buf, n, want_names, locate_map, out):
    """Walk length-prefixed messages in buf[:n]; copy kept ones into out.

    Returns (bytes_consumed, bytes_written, messages_seen). Stops at the last
    complete message so the caller can carry the partial tail to the next chunk.
    locate_map is updated in place when a wanted ticker's 'R' message is seen.
    """
    pos = 0
    w = 0
    seen = 0
    k_names = want_names.shape[0]
    while pos + 2 <= n:
        length = (np.int64(buf[pos]) << 8) | np.int64(buf[pos + 1])
        end = pos + 2 + length
        if end > n:
            break
        t = buf[pos + 2]
        loc = (np.int64(buf[pos + 3]) << 8) | np.int64(buf[pos + 4])
        keep = False
        if t == _S:
            keep = True
        elif t == _R:
            # Stock symbol: 8 bytes at message offset 11 (+2 for prefix).
            for k in range(k_names):
                match = True
                for j in range(8):
                    if buf[pos + 13 + j] != want_names[k, j]:
                        match = False
                        break
                if match:
                    locate_map[loc] = k
                    keep = True
                    break
        elif locate_map[loc] >= 0:
            keep = True
        if keep:
            for i in range(pos, end):
                out[w] = buf[i]
                w += 1
        pos = end
        seen += 1
    return pos, w, seen


def _pad_symbol(sym: str) -> bytes:
    b = sym.encode("ascii")
    if len(b) > 8:
        raise ValueError(f"symbol too long: {sym}")
    return b.ljust(8, b" ")


def filter_file(
    raw_path: str | Path,
    tickers: list[str],
    out_path: str | Path,
    chunk_bytes: int = 1 << 27,
    allow_truncated: bool = False,
    log_every_s: float = 30.0,
) -> dict:
    """Stream a gzipped ITCH file (or '-' for gzipped stdin), keep 'S' + messages for `tickers`.

    Reading stdin lets a remote box run `curl URL | lobgen filter - ...` without storing
    the raw multi-GB day.
    """
    from_stdin = str(raw_path) == "-"
    out_path = Path(out_path)
    want = np.frombuffer(b"".join(_pad_symbol(t) for t in tickers), dtype=np.uint8)
    want = want.reshape(len(tickers), 8).copy()
    locate_map = np.full(1 << 16, -1, dtype=np.int64)

    buf = np.empty(chunk_bytes, dtype=np.uint8)
    out = np.empty(chunk_bytes, dtype=np.uint8)
    mv = memoryview(buf)
    have = 0
    total_seen = total_kept_bytes = total_raw_bytes = 0
    truncated = False
    compressed_size = None if from_stdin else Path(raw_path).stat().st_size
    t0 = last_log = time.time()
    src = gzip.GzipFile(fileobj=sys.stdin.buffer) if from_stdin else gzip.open(raw_path, "rb")

    with src as f, open(out_path, "wb") as o:
        while True:
            try:
                n = f.readinto(mv[have:])
            except EOFError:
                if not allow_truncated:
                    raise
                truncated, n = True, 0
            have += n
            total_raw_bytes += n
            consumed, w, seen = _scan(buf, have, want, locate_map, out)
            o.write(memoryview(out)[:w])
            total_seen += seen
            total_kept_bytes += w
            rem = have - consumed
            if consumed == 0 and have == chunk_bytes:
                raise ValueError(f"message longer than chunk_bytes={chunk_bytes}")
            if rem:
                buf[:rem] = buf[consumed:have].copy()
            have = rem
            now = time.time()
            if now - last_log > log_every_s:
                pct = f"{f.fileobj.tell() / compressed_size:6.1%} compressed read | " if compressed_size else ""
                print(
                    f"  {pct}{total_raw_bytes / 1e9:7.2f} GB raw | {total_seen / 1e6:8.1f} M msgs | "
                    f"{(now - t0) / 60:5.1f} min",
                    file=sys.stderr, flush=True,
                )
                last_log = now
            if n == 0:
                break

    if have and not truncated:
        raise ValueError(f"{have} trailing bytes do not form a complete message")
    found = {t: int(np.flatnonzero(locate_map == k)[0]) if (locate_map == k).any() else None
             for k, t in enumerate(tickers)}
    missing = [t for t, loc in found.items() if loc is None]
    if missing:
        raise ValueError(f"tickers not in stock directory: {missing}")
    return {
        "messages_seen": total_seen,
        "raw_bytes": total_raw_bytes,
        "kept_bytes": total_kept_bytes,
        "locates": found,
        "truncated": truncated,
        "minutes": (time.time() - t0) / 60,
    }


# ---------------------------------------------------------------------------
# Pure-Python message decoding for the (small) filtered stream.
# ---------------------------------------------------------------------------

def load_stream(path: str | Path) -> bytes:
    """Read a filtered ITCH file (.gz ok) into memory."""
    p = Path(path)
    return gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()


def iter_messages(src: str | Path | bytes) -> Iterator[bytes]:
    """Yield raw message bodies (without length prefix) from a filtered stream or its path."""
    data = src if isinstance(src, (bytes, bytearray)) else load_stream(src)
    pos, n = 0, len(data)
    while pos + 2 <= n:
        (length,) = struct.unpack_from(">H", data, pos)
        end = pos + 2 + length
        if end > n:
            raise ValueError(f"truncated message at byte {pos}")
        yield data[pos + 2:end]
        pos = end


def header(msg: bytes) -> tuple[str, int, int]:
    """(type char, locate, timestamp_ns)."""
    t = chr(msg[0])
    locate = (msg[1] << 8) | msg[2]
    ts = int.from_bytes(msg[5:11], "big")
    return t, locate, ts


# Body decoders, offsets after the 11-byte common header.
_U64 = struct.Struct(">Q")
_ADD = struct.Struct(">QcI8sI")          # ref, side, shares, stock, price
_EXEC = struct.Struct(">QIQ")            # ref, shares, match
_EXEC_PX = struct.Struct(">QIQcI")       # ref, shares, match, printable, price
_CANCEL = struct.Struct(">QI")           # ref, cancelled shares
_REPLACE = struct.Struct(">QQII")        # orig ref, new ref, shares, price
_TRADE = struct.Struct(">QcI8sIQ")       # ref, side, shares, stock, price, match
_CROSS = struct.Struct(">Q8sIQc")        # shares, stock, price, match, cross type
_HALT = struct.Struct(">8scc4s")         # stock, state, reserved, reason
_DIR = struct.Struct(">8scc")            # stock, market category, FS status


def decode(msg: bytes):
    """Decode one message into (type, locate, ts, fields tuple). Unused types -> fields None."""
    t, locate, ts = header(msg)
    if t in ("A", "F"):
        f = _ADD.unpack_from(msg, 11)
    elif t == "E":
        f = _EXEC.unpack_from(msg, 11)
    elif t == "C":
        f = _EXEC_PX.unpack_from(msg, 11)
    elif t == "X":
        f = _CANCEL.unpack_from(msg, 11)
    elif t == "D":
        f = _U64.unpack_from(msg, 11)
    elif t == "U":
        f = _REPLACE.unpack_from(msg, 11)
    elif t == "P":
        f = _TRADE.unpack_from(msg, 11)
    elif t == "Q":
        f = _CROSS.unpack_from(msg, 11)
    elif t == "H":
        f = _HALT.unpack_from(msg, 11)
    elif t == "R":
        f = _DIR.unpack_from(msg, 11)
    elif t == "S":
        f = (msg[11:12],)
    else:
        f = None
    return t, locate, ts, f
