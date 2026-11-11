"""The market as an input: how news reaches a single stock's order book.

A tariff headline, a rate decision or a sector sell-off does not arrive in GOOG's order book as
text. It arrives as a move in the whole market, and market makers in GOOG re-price off index
futures and ETFs within milliseconds. So the channel this module gives the model is the market
itself: the Nasdaq quotes of QQQ (Nasdaq-100) and SMH (semiconductors, for INTC), taken from
the same ITCH files as the stocks, turned into clock-time features at every stock event.

Per ETF and per time scale tau in CLOCK_TAUS_S (0.1, 1, 10, 100 s):
    dev_tau(t) = log mid(t) - EMA_tau(log mid)(t)        in bps / 10, clipped to +-5
a continuous-time exponential average, so dev_tau ~ the ETF's return over the last ~tau; and
per tau in VOL_TAUS_S (1, 30 s) a turbulence measure:
    vol_tau(t) = log1p( sum_k |d log mid_k| e^{-(t - t_k)/tau} in bps ) / 3
The same clock-time deviations of the stock's *own* mid (CLOCK_TAUS_S) are carried in the
feature stream (features.py), so the model can see the gap between how far the market moved and
how far the stock has followed: lead-lag catch-up is what makes a stock "go down as much as it
should" after market-wide news.

Lookups are exact and causal: features at time t use only ETF quotes at or before t. During a
rollout the generated event times query the same series, so a generated run is conditioned on
the market path: the real one (what the market did), a frozen one (no news after the start),
or a scenario (e.g. QQQ -1% over 30 s: a headline) - see Market.scenario().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

ETFS = ("QQQ", "SMH")
CLOCK_TAUS_S = (0.1, 1.0, 10.0, 100.0)
VOL_TAUS_S = (1.0, 30.0)
N_ETF_FEAT = len(CLOCK_TAUS_S) + len(VOL_TAUS_S)
N_OWN = len(CLOCK_TAUS_S)
N_MKT = N_OWN + N_ETF_FEAT * len(ETFS)
NS = 1_000_000_000


# ----------------------------------------------------------------------------- extraction

def bbo_series(filtered_path: str | Path, ticker: str, start_s: int = 32_400,
               end_s: int = 57_600) -> dict[str, np.ndarray]:
    """Best bid / ask after every change, from a filtered ITCH stream (prices x 10000).
    Starts at 09:00 so that the averages are warm at the open."""
    from .book import OrderBook
    from .itch import load_stream
    from .lobster import Report, _replay

    sym = ticker.encode().ljust(8, b" ")
    book = OrderBook()
    t_out, bid_out, ask_out = [], [], []
    last = [None, None]
    lo, hi = start_s * NS, end_s * NS

    def emit(ts, *_):
        if not lo <= ts < hi:
            return
        b, a = book.best_bid(), book.best_ask()
        if (b, a) != (last[0], last[1]):
            last[0], last[1] = b, a
            t_out.append(ts)
            bid_out.append(-1 if b is None else b)
            ask_out.append(-1 if a is None else a)

    rep = Report(ticker=ticker, date="")
    _replay(load_stream(filtered_path), sym, book, emit, rep)
    if rep.locate is None:
        raise ValueError(f"{ticker} not in {filtered_path}")
    return {"t": np.array(t_out, dtype=np.int64), "bid": np.array(bid_out, dtype=np.int64),
            "ask": np.array(ask_out, dtype=np.int64)}


def save_bbo(s: dict, path: str | Path) -> None:
    np.savez_compressed(path, t=s["t"], bid=s["bid"], ask=s["ask"])


# ----------------------------------------------------------------------------- features

@njit(cache=True)
def _precompute(t, x, taus_ns, vtaus_ns, E, R):
    """E[k, j]: EMA_tau_j of the piecewise-constant x at time t[k] (just before x[k] takes
    over). R[k, j]: decayed sum of |jumps| up to and including the jump at t[k], in bps."""
    n = t.shape[0]
    for j in range(taus_ns.shape[0]):
        E[0, j] = x[0]
    for j in range(vtaus_ns.shape[0]):
        R[0, j] = 0.0
    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        for j in range(taus_ns.shape[0]):
            a = np.exp(-dt / taus_ns[j])
            E[k, j] = a * E[k - 1, j] + (1.0 - a) * x[k - 1]
        jump = abs(x[k] - x[k - 1]) * 1e4
        for j in range(vtaus_ns.shape[0]):
            R[k, j] = R[k - 1, j] * np.exp(-dt / vtaus_ns[j]) + jump


class Series:
    """One ETF's log mid as a step function of time, with what the feature lookups need.

    Scenarios do not copy the (multi-million point) arrays: frozen() stops the path at t0 (the
    averages keep decaying towards the last value, exactly as for a quote that never changes
    again) and shifted() adds a small overlay path. Averages are linear, so the overlay's
    deviations add; jumps at different times add in the turbulence sum."""

    def __init__(self, t: np.ndarray, logmid: np.ndarray) -> None:
        order = np.argsort(t, kind="stable")
        self.t = np.asarray(t, dtype=np.int64)[order]
        self.x = np.asarray(logmid, dtype=np.float64)[order]
        if len(self.t) == 0:
            raise ValueError("empty series")
        self.taus = np.array(CLOCK_TAUS_S) * NS
        self.vtaus = np.array(VOL_TAUS_S) * NS
        self.E = np.empty((len(self.t), len(self.taus)))
        self.R = np.empty((len(self.t), len(self.vtaus)))
        _precompute(self.t, self.x, self.taus, self.vtaus, self.E, self.R)
        self.t_end: int | None = None          # frozen after this time
        self.overlays: list["Series"] = []

    @classmethod
    def from_bbo(cls, s: dict) -> "Series":
        ok = (s["bid"] > 0) & (s["ask"] > 0) & (s["ask"] > s["bid"])
        mid = np.where(ok, (s["bid"] + s["ask"]) / 2.0, np.nan)
        # carry the last good quote over one-sided or crossed moments
        idx = np.where(ok, np.arange(len(mid)), -1)
        np.maximum.accumulate(idx, out=idx)
        keep = idx >= 0
        t, m = s["t"][keep], mid[idx[keep]]
        # one quote per timestamp, the last: a replace ('U') is a delete then an add at the same
        # ns, and the book between the two is not a quote anyone saw
        last = np.append(t[1:] != t[:-1], True)
        t, m = t[last], m[last]
        change = np.concatenate([[True], np.diff(m) != 0])
        return cls(t[change], np.log(m[change]))

    def _copy(self) -> "Series":
        c = object.__new__(Series)
        c.__dict__.update(self.__dict__)
        c.overlays = list(self.overlays)
        return c

    def _index(self, q: np.ndarray) -> np.ndarray:
        qc = q if self.t_end is None else np.minimum(q, self.t_end)
        return np.searchsorted(self.t, qc, side="right") - 1

    def value(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.int64)
        k = self._index(q)
        v = np.where(k >= 0, self.x[np.maximum(k, 0)], self.x[0])
        for o in self.overlays:
            ko = o._index(q)
            v = v + np.where(ko >= 0, o.x[np.maximum(ko, 0)], 0.0)
        return v

    def _raw(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(log mid - its averages [n, taus], turbulence sum in bps [n, vol taus]); 0 before
        the first point."""
        k = self._index(q)
        before = k < 0
        k = np.maximum(k, 0)
        dt = np.maximum(q - self.t[k], 0).astype(np.float64)
        a = np.exp(-dt[:, None] / self.taus[None, :])
        x = self.x[k][:, None]
        dev = x - (a * self.E[k] + (1 - a) * x)
        R = self.R[k] * np.exp(-dt[:, None] / self.vtaus[None, :])
        dev[before] = 0.0
        R[before] = 0.0
        return dev, R

    def features(self, q: np.ndarray) -> np.ndarray:
        """[len(q), N_ETF_FEAT] at times q (ns since midnight)."""
        q = np.asarray(q, dtype=np.int64)
        dev, R = self._raw(q)
        for o in self.overlays:
            d2, r2 = o._raw(q)
            dev, R = dev + d2, R + r2
        return np.hstack([np.clip(dev * 1e4 / 10, -5, 5), np.log1p(R) / 3])

    def shifted(self, t0: int, bps: float, ramp_s: float, step_ns: int = 10_000_000) -> "Series":
        """This path with an extra move of `bps` that builds up linearly (in 10 ms steps) over
        ramp_s seconds from t0 and then stays: a news shock on top of what the market did."""
        n = max(int(ramp_s * NS) // step_ns, 1)
        j = np.arange(n + 1)
        ov = Series(t0 + j * step_ns, bps / 1e4 * j / n)
        ov.t_end = self.t_end
        c = self._copy()
        c.overlays.append(ov)
        return c

    def frozen(self, t0: int) -> "Series":
        """The path up to t0, then flat: a market with no news after t0."""
        c = self._copy()
        c.t_end = t0 if self.t_end is None else min(t0, self.t_end)
        for i, o in enumerate(c.overlays):
            c.overlays[i] = o._copy()
            c.overlays[i].t_end = c.t_end
        return c


class Market:
    """The ETF series of one day, in ETFS order; features() gives the ETF part of the input."""

    def __init__(self, series: list[Series], date: str = "") -> None:
        if len(series) != len(ETFS):
            raise ValueError(f"need one series per ETF {ETFS}")
        self.series, self.date = series, date

    @classmethod
    def load(cls, date: str, root: str | Path = "data/market") -> "Market":
        root = Path(root)
        out = []
        for e in ETFS:
            p = root / f"{e}_{date}.npz"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run notebooks/fetch_itch_colab.ipynb "
                                        "(market part) and download MyDrive/lobgen/market/")
            z = np.load(p)
            out.append(Series.from_bbo({k: z[k] for k in ("t", "bid", "ask")}))
        return cls(out, date)

    def features(self, q: np.ndarray) -> np.ndarray:
        return np.hstack([s.features(q) for s in self.series])

    def scenario(self, t0: int, bps: float, ramp_s: float = 30.0, etfs=ETFS) -> "Market":
        """News scenario: the chosen ETFs move an extra `bps` over ramp_s seconds from t0."""
        return Market([s.shifted(t0, bps, ramp_s) if e in etfs else s
                       for e, s in zip(ETFS, self.series)], self.date)

    def frozen(self, t0: int) -> "Market":
        return Market([s.frozen(t0) for s in self.series], self.date)

    def ret_bps(self, t0: int, t1: int, etf: str = "QQQ") -> float:
        s = self.series[ETFS.index(etf)]
        return float((s.value(np.array([t1]))[0] - s.value(np.array([t0]))[0]) * 1e4)


# ----------------------------------------------------------------------------- own clock EMA

@njit(cache=True)
def own_scan(t, mid, taus_ns, E, state, out):
    """Clock-time deviations of the stock's own log mid, in place.

    state = [last log mid (nan before the first), last time]; E [n_tau]. mid: forward-filled
    mid after each event, ticks (nan until the first). out[i] after event i (bps / 10)."""
    n = t.shape[0]
    T = taus_ns.shape[0]
    for i in range(n):
        x_prev = state[0]
        if not np.isnan(x_prev):
            dt = float(t[i]) - state[1]
            if dt > 0:
                for j in range(T):
                    a = np.exp(-dt / taus_ns[j])
                    E[j] = a * E[j] + (1.0 - a) * x_prev
        if not np.isnan(mid[i]):
            x = np.log(mid[i])
            if np.isnan(x_prev):
                for j in range(T):
                    E[j] = x
            state[0] = x
        state[1] = float(t[i])
        for j in range(T):
            d = 0.0
            if not np.isnan(state[0]):
                d = (state[0] - E[j]) * 1e4 / 10
            out[i, j] = min(max(d, -5.0), 5.0)


OWN_TAUS_NS = np.array(CLOCK_TAUS_S) * NS
