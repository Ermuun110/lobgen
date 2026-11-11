"""Model input features for each event (numpy + numba only, so windows can carry state).

Token for event i = the action as applied (type/side, price field, queue position) + what the
event did (size, gap) + the book right after it + time of day + a *day-so-far* summary.

The day-so-far summary is the reason this module keeps state. GOOG prints ~400 events a second,
so a 256-event context is ~0.6 s of history: far too little to tell what kind of day it is
(on Fri 2025-12-12 79% of GOOG submits were exactly 100 shares, on Mon 34%). Each token
therefore also carries exponentially weighted means, over the day so far, of 14 per-event
quantities (event mix, round lots, sizes, gaps, pointer cancels, spread, mid moves) at two
half-lives: 2,000 events (GOOG ~5 s, INTC ~20 s) and 100,000 events (GOOG ~4 min, INTC ~18 min).
They are causal and are carried through generated events exactly as through real ones.

Order-flow memory (v3, ModelConfig.flow). A 128-event context holds only ~3 visible trades, but
real trade signs stay correlated for hundreds of trades (large orders are split into many small
ones). So each token also carries, in *trade time*, exponentially weighted means of the aggressor
sign and of the signed log size of visible trades at half-lives of 8, 64 and 512 trades; the log
number of events since the last trade; and how far the mid is from its own moving average at
half-lives of 256 and 4,096 events (in ticks / 10, clipped to +-3), so that price moves can
revert as they do after real trades.

Market inputs (v4, ModelConfig.market; see market.py). The last N_MKT columns: clock-time
deviations of the stock's own log mid from its moving averages at 0.1 / 1 / 10 / 100 s, then the
same for the QQQ and SMH quotes plus their turbulence. The ETF columns need a market.Market for
the day (token_features(..., market=)); without one they are 0, which only models trained
without them may see.

FeatState holds what a stream needs to continue: the last valid mid and the running sums.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .market import N_MKT, N_OWN, OWN_TAUS_NS, own_scan
from .sim import CANCEL, DELETE, DT, EXEC, MODELED_TYPES, PRICE, QPOS, SIDE, SUBMIT, TICK, TYPE

N_TS = 10                    # (type 1..5) x (side)
P_LO, P_HI = 20, 60          # input embedding range of the price field
Q_MAX, SPREAD_MAX, SIZE_BUCKETS = 31, 31, 42
N_CAT = 4
N_BASE = 47                  # per-event continuous features
HALF_LIVES = (2_000, 100_000)
N_REG = 14                   # per-event quantities averaged over the day so far
N_CONT = N_BASE + N_REG * len(HALF_LIVES)  # v2 inputs (regime)
REGIME_COLS = list(range(N_BASE, N_CONT))
TRADE_HALF_LIVES = (8, 64, 512)             # in visible trades
MID_HALF_LIVES = (256, 4_096)               # in events
N_FLOW = 2 * len(TRADE_HALF_LIVES) + 1 + len(MID_HALF_LIVES)
N_ALL = N_CONT + N_FLOW                     # v3 inputs (regime + order-flow memory)
FLOW_COLS = list(range(N_CONT, N_ALL))
FLOW_BOOK = list(range(N_ALL - len(MID_HALF_LIVES), N_ALL))  # mid vs its average: book-derived
N_FULL = N_ALL + N_MKT                      # v4 inputs (+ own clock-time moves + market)
OWN_COLS = list(range(N_ALL, N_ALL + N_OWN))  # the stock's own clock-time moves: book-derived
ETF_COLS = list(range(N_ALL + N_OWN, N_FULL))
BOOK_COLS = list(range(5, N_BASE))          # per-event columns that describe the book
REGIME_BOOK = (12, 13)                      # regime quantities derived from the book
_DECAY = np.array([0.5 ** (1.0 / h) for h in HALF_LIVES])
_TR_DECAY = np.array([0.5 ** (1.0 / h) for h in TRADE_HALF_LIVES])
_MID_DECAY = np.array([0.5 ** (1.0 / h) for h in MID_HALF_LIVES])


def ts_index(typ: np.ndarray, side: np.ndarray) -> np.ndarray:
    return (np.clip(typ, 1, 5) - 1) * 2 + (side > 0)


def ts_decode(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return ts // 2 + 1, np.where(ts % 2 == 1, 1, -1)


def size_bucket(size):
    """Half-octave buckets of the size field, conditioning later heads on the chosen size.
    0 (an execution of the whole order) has its own bucket."""
    try:
        import torch
        if isinstance(size, torch.Tensor):
            b = torch.round(2 * torch.log2(size.clamp(min=1).float())).long() + 1
            return torch.where(size <= 0, 0, b.clamp(1, SIZE_BUCKETS - 1))
    except ImportError:
        pass
    s = np.asarray(size)
    b = np.clip(np.round(2 * np.log2(np.maximum(s, 1))).astype(np.int64) + 1, 1, SIZE_BUCKETS - 1)
    return np.where(s <= 0, 0, b)


@dataclass
class FeatState:
    """Everything a feature stream carries from one event to the next."""
    prev_mid: float = np.nan     # last valid mid, ticks
    num: np.ndarray = field(default_factory=lambda: np.zeros((len(HALF_LIVES), N_REG)))
    den: np.ndarray = field(default_factory=lambda: np.zeros(len(HALF_LIVES)))
    # order-flow memory: [sign, signed log size] sums per trade half-life, events since the
    # last trade, and the mid's moving-average sums per mid half-life
    tr_num: np.ndarray = field(default_factory=lambda: np.zeros((len(TRADE_HALF_LIVES), 2)))
    tr_den: np.ndarray = field(default_factory=lambda: np.zeros(len(TRADE_HALF_LIVES)))
    since: np.ndarray = field(default_factory=lambda: np.full(1, 1e4))
    m_num: np.ndarray = field(default_factory=lambda: np.zeros(len(MID_HALF_LIVES)))
    m_den: np.ndarray = field(default_factory=lambda: np.zeros(len(MID_HALF_LIVES)))
    # own clock-time averages of the log mid, and [last log mid, last event time ns]
    c_E: np.ndarray = field(default_factory=lambda: np.zeros(N_OWN))
    c_state: np.ndarray = field(default_factory=lambda: np.array([np.nan, 0.0]))

    def copy(self) -> "FeatState":
        return FeatState(self.prev_mid, self.num.copy(), self.den.copy(), self.tr_num.copy(),
                         self.tr_den.copy(), self.since.copy(), self.m_num.copy(),
                         self.m_den.copy(), self.c_E.copy(), self.c_state.copy())


def _mids(book: np.ndarray) -> np.ndarray:
    b = np.asarray(book)
    two = (b[:, 1] > 0) & (b[:, 3] > 0)
    return np.where(two, (b[:, 0] + b[:, 2]) / (2 * TICK), np.nan)


def _mid_moves(mid: np.ndarray, prev_mid: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Mid change per event (0 when either side is empty; forward-filled), the forward-filled
    mid after each event (nan until the first valid one), and the last mid."""
    full = np.concatenate([[prev_mid], mid])
    idx = np.where(~np.isnan(full), np.arange(len(full)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = full[idx]
    return np.nan_to_num(np.diff(filled), nan=0.0), filled[1:], float(filled[-1])


def _base(actions, msgs, book, dmid) -> tuple[np.ndarray, np.ndarray]:
    a, m, b = np.asarray(actions), np.asarray(msgs), np.asarray(book)
    n = len(a)
    typ, side, p, q, dt = a[:, TYPE], a[:, SIDE], a[:, PRICE], a[:, QPOS], a[:, DT]
    size, t = m[:, 3], m[:, 0]
    ask, askv, bid, bidv = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    two = (askv > 0) & (bidv > 0)
    cat = np.empty((n, N_CAT), dtype=np.int64)
    cat[:, 0] = ts_index(typ, side)
    cat[:, 1] = np.clip(p, -P_LO, P_HI) + P_LO
    cat[:, 2] = np.clip(q, 0, Q_MAX)
    cat[:, 3] = np.clip(np.where(two, (ask - bid) // TICK, 0), 0, SPREAD_MAX)

    cont = np.zeros((n, N_FULL), dtype=np.float32)
    cont[:, 0] = np.log1p(np.maximum(dt, 0)) / 10
    cont[:, 1] = dt == 0
    cont[:, 2] = np.log1p(np.maximum(size, 0)) / 5      # shares the event moved (message)
    cont[:, 3] = (size % 100) == 0
    cont[:, 4] = (t / 1e9 - 34_200) / 23_400
    levels = b.shape[1] // 4
    av, bv = b[:, 1::4][:, :10], b[:, 3::4][:, :10]
    ap, bp = b[:, 0::4][:, :10], b[:, 2::4][:, :10]
    cont[:, 5:5 + av.shape[1]] = np.log1p(av) / 5
    cont[:, 15:15 + bv.shape[1]] = np.log1p(bv) / 5
    if levels > 1:
        k = min(levels, 10) - 1
        ga = np.where((av[:, 1:] > 0) & (av[:, :1] > 0), (ap[:, 1:] - ap[:, :1]) / TICK / 10, 0)
        gb = np.where((bv[:, 1:] > 0) & (bv[:, :1] > 0), (bp[:, :1] - bp[:, 1:]) / TICK / 10, 0)
        cont[:, 25:25 + k] = np.clip(ga, 0, 10)
        cont[:, 34:34 + k] = np.clip(gb, 0, 10)
    tot = askv + bidv
    cont[:, 43] = np.where(tot > 0, (bidv - askv) / np.where(tot > 0, tot, 1), 0)
    cont[:, 44] = np.clip(dmid, -10, 10) / 10
    cont[:, 45] = np.log1p(av.sum(1)) / 8
    cont[:, 46] = np.log1p(bv.sum(1)) / 8
    return cat, cont


def regime_inputs(actions, msgs, book, dmid) -> np.ndarray:
    """The 14 per-event quantities averaged by the day-so-far features, float64 [N, 14]."""
    a, m, b = np.asarray(actions), np.asarray(msgs), np.asarray(book)
    typ, side, p, dt = a[:, TYPE], a[:, SIDE], a[:, PRICE], a[:, DT]
    size = m[:, 3]
    sub = typ == SUBMIT
    two = (b[:, 1] > 0) & (b[:, 3] > 0)
    x = np.empty((len(a), N_REG))
    for k in range(5):
        x[:, k] = typ == k + 1
    x[:, 5] = side > 0
    x[:, 6] = sub & (size == 100)
    x[:, 7] = sub & (size % 100 == 0)
    x[:, 8] = np.log1p(np.maximum(size, 0)) / 5
    x[:, 9] = np.log1p(np.maximum(dt, 0)) / 10
    x[:, 10] = dt == 0
    x[:, 11] = ((typ == CANCEL) | (typ == DELETE)) & (p < 0)
    x[:, 12] = np.where(two, np.clip((b[:, 0] - b[:, 2]) / TICK, 0, 10) / 10, 0)
    x[:, 13] = np.clip(np.abs(dmid), 0, 5) / 5
    return x


@njit(cache=True)
def _ema_scan(x, valid, decay, num, den, out):
    """Day-so-far means, in place: num/den updated by each valid event; out[i] after event i."""
    n, r = x.shape
    h = decay.shape[0]
    for i in range(n):
        if valid[i]:
            for k in range(h):
                den[k] = decay[k] * den[k] + 1.0
                for j in range(r):
                    num[k, j] = decay[k] * num[k, j] + x[i, j]
        for k in range(h):
            inv = 1.0 / den[k] if den[k] > 0 else 0.0
            for j in range(r):
                out[i, k * r + j] = num[k, j] * inv


def flow_inputs(msgs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visible trades (LOBSTER type 4), aggressor sign (-direction of the resting order, as in
    facts.py) and log size / 5."""
    m = np.asarray(msgs)
    trade = m[:, 1] == EXEC
    return trade, -m[:, 5].astype(np.float64), np.log1p(np.maximum(m[:, 3], 0)) / 5


@njit(cache=True)
def _flow_scan(trade, eps, lsize, valid, mid, tr_decay, m_decay, tr_num, tr_den, since,
               m_num, m_den, out):
    """Order-flow memory, in place (see the module docstring); out[i] after event i."""
    n = trade.shape[0]
    T, M = tr_decay.shape[0], m_decay.shape[0]
    for i in range(n):
        if valid[i]:
            since[0] += 1.0
            if trade[i]:
                since[0] = 0.0
                for k in range(T):
                    tr_den[k] = tr_decay[k] * tr_den[k] + 1.0
                    tr_num[k, 0] = tr_decay[k] * tr_num[k, 0] + eps[i]
                    tr_num[k, 1] = tr_decay[k] * tr_num[k, 1] + eps[i] * lsize[i]
            if not np.isnan(mid[i]):
                for k in range(M):
                    m_den[k] = m_decay[k] * m_den[k] + 1.0
                    m_num[k] = m_decay[k] * m_num[k] + mid[i]
        for k in range(T):
            inv = 1.0 / tr_den[k] if tr_den[k] > 0 else 0.0
            out[i, k] = tr_num[k, 0] * inv
            out[i, T + k] = tr_num[k, 1] * inv
        out[i, 2 * T] = np.log1p(since[0]) / 10
        for k in range(M):
            d = 0.0
            if m_den[k] > 0 and not np.isnan(mid[i]):
                d = (mid[i] - m_num[k] / m_den[k]) / 10
            out[i, 2 * T + 1 + k] = min(max(d, -3.0), 3.0)


def _scan(st: FeatState, a: np.ndarray, msgs, book, want_features: bool):
    """Advance `st` in place over a stream; return (dmid, day-so-far + flow + own clock-time
    columns or None)."""
    dmid, filled, st.prev_mid = _mid_moves(_mids(book), st.prev_mid)
    own = np.empty((len(a), N_OWN))
    own_scan(np.asarray(msgs)[:, 0], filled, OWN_TAUS_NS, st.c_E, st.c_state, own)
    x = regime_inputs(a, msgs, book, dmid)
    valid = np.isin(a[:, TYPE], MODELED_TYPES)
    reg = np.empty((len(a), N_REG * len(HALF_LIVES)))
    _ema_scan(x, valid, _DECAY, st.num, st.den, reg)
    flow = np.empty((len(a), N_FLOW))
    trade, eps, lsize = flow_inputs(msgs)
    _flow_scan(trade, eps, lsize, valid, filled, _TR_DECAY, _MID_DECAY, st.tr_num, st.tr_den,
               st.since, st.m_num, st.m_den, flow)
    return dmid, (np.hstack([reg, flow, own]) if want_features else None)


def token_features(actions: np.ndarray, msgs: np.ndarray, book: np.ndarray,
                   state: FeatState | None = None,
                   market=None) -> tuple[np.ndarray, np.ndarray, FeatState]:
    """One stream of consecutive events: actions [N, 6] as applied, LOBSTER messages [N, 6],
    book [N, 4L] after each event. Returns (cat int64 [N, 4], cont float32 [N, N_FULL], state
    after the last event). Chunking a stream and passing the state on gives identical output.
    A model reads the first ModelConfig.n_cont columns (v1: N_BASE, v2: N_CONT, v3: N_ALL,
    v4: N_FULL). market: market.Market of the day for the ETF columns (else they are 0)."""
    st = FeatState() if state is None else state.copy()
    a = np.asarray(actions)
    dmid, extra = _scan(st, a, msgs, book, True)
    cat, cont = _base(a, msgs, book, dmid)
    cont[:, N_BASE:N_ALL + N_OWN] = extra
    if market is not None:
        cont[:, N_ALL + N_OWN:] = market.features(np.asarray(msgs)[:, 0])
    return cat, cont, st


def advance(state: FeatState, actions, msgs, book) -> FeatState:
    """State after a stream of events, without building the per-event features."""
    st = state.copy()
    _scan(st, np.asarray(actions), msgs, book, False)
    return st


def states_before(actions, msgs, book, rows, chunk: int = 1_000_000) -> list[FeatState]:
    """Feature state just before each row index in `rows` (ascending) of a day's stream."""
    rows = list(rows)
    if rows != sorted(rows):
        raise ValueError("rows must be ascending")
    st, j, out = FeatState(), 0, []
    for r in rows:
        while j < r:
            e = min(r, j + chunk)
            st = advance(st, actions[j:e], msgs[j:e], book[j:e])
            j = e
        out.append(st.copy())
    return out


class BatchState:
    """FeatStates of B independent streams, stepped one event at a time (rollouts).
    markets: one market.Market (or None) per stream, for the ETF columns."""

    def __init__(self, states: list[FeatState], markets: list | None = None) -> None:
        self.markets = markets if markets is not None else [None] * len(states)
        self.c_E = np.stack([s.c_E for s in states])          # [B, N_OWN]
        self.c_x = np.array([s.c_state[0] for s in states], dtype=np.float64)
        self.c_t = np.array([s.c_state[1] for s in states], dtype=np.float64)
        self.prev_mid = np.array([s.prev_mid for s in states], dtype=np.float64)
        self.num = np.stack([s.num for s in states])          # [B, H, R]
        self.den = np.stack([s.den for s in states])          # [B, H]
        self.tr_num = np.stack([s.tr_num for s in states])    # [B, T, 2]
        self.tr_den = np.stack([s.tr_den for s in states])    # [B, T]
        self.since = np.array([s.since[0] for s in states], dtype=np.float64)
        self.m_num = np.stack([s.m_num for s in states])      # [B, M]
        self.m_den = np.stack([s.m_den for s in states])      # [B, M]

    def step(self, actions, msgs, book) -> tuple[np.ndarray, np.ndarray]:
        """One new event per stream (row b = stream b). Same values token_features gives."""
        mid = _mids(book)
        dmid = np.nan_to_num(mid - self.prev_mid, nan=0.0)
        self.prev_mid = np.where(np.isnan(mid), self.prev_mid, mid)
        cat, cont = _base(actions, msgs, book, dmid)
        x = regime_inputs(actions, msgs, book, dmid)
        valid = np.isin(np.asarray(actions)[:, TYPE], MODELED_TYPES)
        d = _DECAY[None, :]
        self.den = np.where(valid[:, None], d * self.den + 1.0, self.den)
        self.num = np.where(valid[:, None, None], d[..., None] * self.num + x[:, None, :], self.num)
        inv = np.where(self.den > 0, 1.0 / np.where(self.den > 0, self.den, 1), 0.0)
        cont[:, N_BASE:N_CONT] = (self.num * inv[..., None]).reshape(len(x), -1)
        cont[:, N_CONT:N_ALL] = self._flow_step(msgs, valid, self.prev_mid)
        t = np.asarray(msgs)[:, 0]
        cont[:, N_ALL:N_ALL + N_OWN] = self._own_step(t, self.prev_mid)
        groups: dict[int, list[int]] = {}
        for b, mk in enumerate(self.markets):
            if mk is not None:
                groups.setdefault(id(mk), []).append(b)
        for rows in groups.values():
            cont[rows, N_ALL + N_OWN:] = self.markets[rows[0]].features(t[rows])
        return cat, cont

    def _own_step(self, t, mid) -> np.ndarray:
        """market.own_scan for one event of each stream (mid = forward-filled mid after it)."""
        t = t.astype(np.float64)
        dt = t - self.c_t
        upd = ~np.isnan(self.c_x) & (dt > 0)
        a = np.exp(-np.where(upd, dt, 0.0)[:, None] / OWN_TAUS_NS[None, :])
        self.c_E = np.where(upd[:, None], a * self.c_E + (1 - a) * np.nan_to_num(self.c_x)[:, None],
                            self.c_E)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.log(mid)
        has = ~np.isnan(x)
        first = np.isnan(self.c_x) & has
        self.c_E = np.where(first[:, None], np.nan_to_num(x)[:, None], self.c_E)
        self.c_x = np.where(has, x, self.c_x)
        self.c_t = t
        d = np.where(np.isnan(self.c_x)[:, None], 0.0,
                     (np.nan_to_num(self.c_x)[:, None] - self.c_E) * 1e4 / 10)
        return np.clip(d, -5, 5)

    def _flow_step(self, msgs, valid, mid) -> np.ndarray:
        """_flow_scan for one event of each stream (mid = forward-filled mid after it)."""
        trade, eps, lsize = flow_inputs(msgs)
        tr = valid & trade
        self.since = np.where(valid, np.where(trade, 0.0, self.since + 1.0), self.since)
        d = _TR_DECAY[None, :]
        self.tr_den = np.where(tr[:, None], d * self.tr_den + 1.0, self.tr_den)
        new = d[..., None] * self.tr_num + np.stack([eps, eps * lsize], -1)[:, None, :]
        self.tr_num = np.where(tr[:, None, None], new, self.tr_num)
        has_mid = valid & ~np.isnan(mid)
        dm = _MID_DECAY[None, :]
        self.m_den = np.where(has_mid[:, None], dm * self.m_den + 1.0, self.m_den)
        self.m_num = np.where(has_mid[:, None], dm * self.m_num + np.nan_to_num(mid)[:, None],
                              self.m_num)
        inv = np.where(self.tr_den > 0, 1.0 / np.where(self.tr_den > 0, self.tr_den, 1), 0.0)
        mavg = self.m_num / np.where(self.m_den > 0, self.m_den, 1)
        dev = np.where((self.m_den > 0) & ~np.isnan(mid)[:, None], (mid[:, None] - mavg) / 10, 0.0)
        return np.hstack([self.tr_num[..., 0] * inv, self.tr_num[..., 1] * inv,
                          np.log1p(self.since)[:, None] / 10, np.clip(np.nan_to_num(dev), -3, 3)])
