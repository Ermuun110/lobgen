"""Convert a filtered ITCH 5.0 stream into LOBSTER-format message + orderbook files.

LOBSTER message columns: time (s after midnight), type, order id, size, price, direction.
  type 1 submit | 2 partial cancel | 3 delete | 4 visible execution | 5 hidden execution
       6 cross trade | 7 trading halt
  direction: 1 buy limit order, -1 sell limit order (an execution's direction is the
  resting order's side, so a type-4 with direction -1 is a buyer-initiated trade).
  price: integer dollars x 10000 (same as ITCH's 4 implied decimals; never converted to float).

Conversion choices (documented because they change the statistics):
  * The book is rebuilt from the first message of the file so pre-market orders exist
    at the open; only events with start <= t < end are written.
  * ITCH 'U' (replace) -> type 3 delete of the old ref, then type 1 submit of the new ref,
    same timestamp, side inherited from the old order.
  * ITCH 'C' (execution with price) -> type 4 at the resting order's LIMIT price, so every
    type-4 row reduces the book level at its own price. The actual print price differs;
    counts of such rows are in the report.
  * ITCH 'P' (hidden execution) -> type 5, order id 0. Its side flag is unreliable: Nasdaq's
    spec says it has always been 'B' since 2014-07-14. Do not use type-5 direction.
  * ITCH 'Q' (cross) -> type 6, order id 0, direction -1. Cross executions of displayed
    orders also arrive as non-printable 'E'/'C' rows (type 4), so cross volume appears twice.
  * ITCH 'H' state H (halt) -> type 7 price -1; Q (quote only) -> price 0; T (trading) -> 1.
  * Empty book levels: ask 9999999999 / bid -9999999999, size 0.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .book import BUY, SELL, OrderBook
from .itch import decode, header, iter_messages, load_stream

SUBMIT, CANCEL, DELETE, EXEC, HIDDEN, CROSS, HALT = 1, 2, 3, 4, 5, 6, 7
NS = 1_000_000_000
_SIDE = {b"B": BUY, b"S": SELL}
_HALT_PRICE = {b"H": -1, b"P": -1, b"Q": 0, b"T": 1}


class _Rows:
    """Preallocated row buffer (a full GOOG day is ~9M rows x 40 cols; doubling would need ~8 GB)."""

    def __init__(self, n: int, width: int) -> None:
        self.a = np.empty((n, width), dtype=np.int64)
        self.n = 0

    def append(self, row) -> None:
        self.a[self.n] = row
        self.n += 1

    def array(self) -> np.ndarray:
        if self.n != len(self.a):
            raise RuntimeError(f"event count mismatch: counted {len(self.a)}, emitted {self.n}")
        return self.a


def _count_events(data: bytes, sym: bytes, start_ns: int, end_ns: int) -> int:
    """Number of LOBSTER rows convert() will emit (a replace 'U' emits two)."""
    locate, n = None, 0
    for raw in iter_messages(data):
        t, loc, ts = header(raw)
        if t == "R" and raw[11:19] == sym:
            locate = loc
        elif loc == locate and start_ns <= ts < end_ns and t in _EMITTING:
            n += 2 if t == "U" else 1
    return n


_EMITTING = frozenset("AFECXDUPQH")


@dataclass
class Report:
    ticker: str
    date: str
    locate: int | None = None
    itch_counts: Counter = field(default_factory=Counter)
    lobster_counts: Counter = field(default_factory=Counter)
    exec_with_price: int = 0
    exec_with_price_nonprintable: int = 0
    exec_with_price_differs: int = 0
    crossed_rows: int = 0
    first_crossed_s: float | None = None
    last_crossed_s: float | None = None
    open_orders_at_end: int = 0
    system_events: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["itch_counts"] = dict(sorted(self.itch_counts.items()))
        d["lobster_counts"] = {int(k): v for k, v in sorted(self.lobster_counts.items())}
        return d


def convert(
    filtered_path: str | Path,
    ticker: str,
    date: str,
    levels: int = 10,
    start_s: int = 34_200,
    end_s: int = 57_600,
) -> tuple[np.ndarray, np.ndarray, Report]:
    """Returns (messages[N,6] with time in ns, book[N, 4*levels], report)."""
    start_ns, end_ns = start_s * NS, end_s * NS
    sym = ticker.encode().ljust(8, b" ")
    data = load_stream(filtered_path)
    n_rows = _count_events(data, sym, start_ns, end_ns)
    book = OrderBook()
    msgs, snaps = _Rows(n_rows, 6), _Rows(n_rows, 4 * levels)
    rep = Report(ticker=ticker, date=date)

    def emit(ts: int, typ: int, ref: int, size: int, price: int, direction: int) -> None:
        if not (start_ns <= ts < end_ns):
            return
        msgs.append((ts, typ, ref, size, price, direction))
        row = book.snapshot(levels)
        snaps.append(row)
        rep.lobster_counts[typ] += 1
        ask, bid = row[0], row[2]
        if row[1] and row[3] and bid >= ask:
            rep.crossed_rows += 1
            s = ts / NS
            rep.first_crossed_s = s if rep.first_crossed_s is None else rep.first_crossed_s
            rep.last_crossed_s = s

    _replay(data, sym, book, emit, rep)
    if rep.locate is None:
        raise ValueError(f"{ticker} not found in stock directory of {filtered_path}")
    rep.open_orders_at_end = len(book.orders)
    return msgs.array(), snaps.array(), rep


def preopen_orders(filtered_path: str | Path, ticker: str, start_s: int = 34_200) -> np.ndarray:
    """Order-level book just before the first `ticker` message at or after start_s.

    Returns int64 [K, 4] rows (ref, side, price, shares) in time priority (the order rows
    entered the book; a replace re-enters at the back). With the LOBSTER messages from
    start_s onward this is enough to rebuild the full order-level book at any later row.
    """
    sym = ticker.encode().ljust(8, b" ")
    book = OrderBook()
    rep = Report(ticker=ticker, date="")
    _replay(load_stream(filtered_path), sym, book, lambda *a: None, rep, stop_ns=start_s * NS)
    if rep.locate is None:
        raise ValueError(f"{ticker} not found in stock directory of {filtered_path}")
    rows = [(ref, side, price, shares) for ref, (side, price, shares) in book.orders.items()]
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def _replay(data: bytes, sym: bytes, book: OrderBook, emit, rep: Report,
            stop_ns: int | None = None) -> None:
    """Apply every message of `sym` to `book`, calling emit(...) once per LOBSTER row.

    With stop_ns, returns just before the first message of `sym` with timestamp >= stop_ns.
    """
    locate = None
    for raw in iter_messages(data):
        t, loc, ts, f = decode(raw)
        if t == "S":
            rep.system_events.append((f[0].decode(), ts / NS))
            continue
        if t == "R":
            if f[0] == sym:
                locate = loc
                rep.locate = loc
            continue
        if loc != locate or locate is None:
            continue
        if stop_ns is not None and ts >= stop_ns:
            return
        rep.itch_counts[t] += 1

        if t in ("A", "F"):
            ref, side, shares, _stock, price = f[:5]
            d = _SIDE[side]
            book.add(ref, d, price, shares)
            emit(ts, SUBMIT, ref, shares, price, d)
        elif t == "E":
            ref, shares, _match = f
            d, price, _ = book.reduce(ref, shares)
            emit(ts, EXEC, ref, shares, price, d)
        elif t == "C":
            ref, shares, _match, printable, px = f
            d, price, _ = book.reduce(ref, shares)
            rep.exec_with_price += 1
            rep.exec_with_price_nonprintable += printable == b"N"
            rep.exec_with_price_differs += px != price
            emit(ts, EXEC, ref, shares, price, d)
        elif t == "X":
            ref, shares = f
            d, price, _ = book.reduce(ref, shares)
            emit(ts, CANCEL, ref, shares, price, d)
        elif t == "D":
            (ref,) = f
            d, price, shares = book.delete(ref)
            emit(ts, DELETE, ref, shares, price, d)
        elif t == "U":
            old, new, shares, price = f
            d, old_price, old_shares = book.delete(old)
            emit(ts, DELETE, old, old_shares, old_price, d)
            book.add(new, d, price, shares)
            emit(ts, SUBMIT, new, shares, price, d)
        elif t == "P":
            _ref, side, shares, _stock, price, _match = f
            emit(ts, HIDDEN, 0, shares, price, _SIDE[side])
        elif t == "Q":
            shares, _stock, price, _match, _ctype = f
            emit(ts, CROSS, 0, shares, price, -1)
        elif t == "H":
            _stock, state, _res, _reason = f
            emit(ts, HALT, 0, 0, _HALT_PRICE.get(state, -1), -1)


def lobster_stem(ticker: str, date: str, start_s: int, end_s: int) -> str:
    return f"{ticker}_{date}_{start_s * 1000}_{end_s * 1000}"


def format_time(ts_ns: np.ndarray) -> np.ndarray:
    """Exact 'seconds.nanoseconds' strings; no float rounding."""
    sec, ns = np.divmod(ts_ns, NS)
    return np.char.add(np.char.add(sec.astype(str), "."), np.char.zfill(ns.astype(str), 9))


def write_lobster(
    msgs: np.ndarray, book: np.ndarray, out_dir: str | Path, ticker: str, date: str,
    levels: int, start_s: int = 34_200, end_s: int = 57_600, chunk: int = 500_000,
    csv: bool = True,
) -> list[Path]:
    """Always writes a compressed .npz; LOBSTER csv pair only if csv=True (a full day is ~2.4 GB)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = lobster_stem(ticker, date, start_s, end_s)
    npath = out_dir / f"{stem}_{levels}.npz"
    np.savez_compressed(npath, messages=msgs, book=book, levels=levels)
    if not csv:
        return [npath]
    mpath = out_dir / f"{stem}_message_{levels}.csv"
    bpath = out_dir / f"{stem}_orderbook_{levels}.csv"
    write_lobster_csv(msgs, book, mpath, bpath, chunk)
    return [npath, mpath, bpath]


def write_lobster_csv(msgs: np.ndarray, book: np.ndarray, mpath: Path, bpath: Path,
                      chunk: int = 500_000) -> None:
    with open(mpath, "w") as fm, open(bpath, "w") as fb:
        for i in range(0, len(msgs), chunk):
            m = msgs[i:i + chunk]
            df = pd.DataFrame(m[:, 1:], copy=False)
            df.insert(0, "time", format_time(m[:, 0]))
            df.to_csv(fm, header=False, index=False)
            pd.DataFrame(book[i:i + chunk], copy=False).to_csv(fb, header=False, index=False)
