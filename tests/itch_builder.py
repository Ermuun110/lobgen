"""Build synthetic ITCH 5.0 messages for tests."""

import struct


def _msg(t: str, locate: int, ts: int, body: bytes) -> bytes:
    m = t.encode() + struct.pack(">HH", locate, 0) + ts.to_bytes(6, "big") + body
    return struct.pack(">H", len(m)) + m


def _sym(s: str) -> bytes:
    return s.encode().ljust(8, b" ")


def S(ts, code="O"):
    return _msg("S", 0, ts, code.encode())


def R(locate, sym, ts=0):
    body = (_sym(sym) + b"Q" + b"N" + struct.pack(">I", 100) + b"N" + b"C" + b"Z "
            + b"P" + b"N" + b"1" + b"N" + b"N" + struct.pack(">I", 0) + b"N")
    return _msg("R", locate, ts, body)


def A(locate, ts, ref, side, shares, price, sym="GOOG"):
    return _msg("A", locate, ts, struct.pack(">QcI8sI", ref, side.encode(), shares, _sym(sym), price))


def F(locate, ts, ref, side, shares, price, sym="GOOG"):
    return _msg("F", locate, ts, struct.pack(">QcI8sI4s", ref, side.encode(), shares, _sym(sym), price, b"MPID"))


def E(locate, ts, ref, shares, match=1):
    return _msg("E", locate, ts, struct.pack(">QIQ", ref, shares, match))


def C(locate, ts, ref, shares, price, printable="Y", match=1):
    return _msg("C", locate, ts, struct.pack(">QIQcI", ref, shares, match, printable.encode(), price))


def X(locate, ts, ref, shares):
    return _msg("X", locate, ts, struct.pack(">QI", ref, shares))


def D(locate, ts, ref):
    return _msg("D", locate, ts, struct.pack(">Q", ref))


def U(locate, ts, old, new, shares, price):
    return _msg("U", locate, ts, struct.pack(">QQII", old, new, shares, price))


def P(locate, ts, shares, price, side="B", sym="GOOG"):
    return _msg("P", locate, ts, struct.pack(">QcI8sIQ", 0, side.encode(), shares, _sym(sym), price, 1))


def Q(locate, ts, shares, price, sym="GOOG", ctype="O"):
    return _msg("Q", locate, ts, struct.pack(">Q8sIQc", shares, _sym(sym), price, 1, ctype.encode()))


def H(locate, ts, state, sym="GOOG"):
    return _msg("H", locate, ts, struct.pack(">8scc4s", _sym(sym), state.encode(), b" ", b"    "))


def I_noii(locate, ts):
    """NOII message: 50-byte body we never decode, used to test the filter skips by length."""
    return _msg("I", locate, ts, bytes(39))
