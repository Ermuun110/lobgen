"""Shock test: after a large market order, does the book react the way the real one did?

This is the question a trader asks of a simulator ("if I trade into it, does it react like the
market?"), and the one the detective cannot answer: the detective scores runs from ordinary
moments, while the hard part of market behaviour is the reaction to an aggressive order.

Two experiments, both measured in trader units (ticks, shares, milliseconds):

  real shocks   Every large aggressive order of the test day (top 1% by size, the first of any
                cluster) is left in the context, and each generator continues from the real
                book right after it. Its continuation is compared with what really followed:
                the mid-price path (signed, so + means the price kept moving in the order's
                direction), the spread, the depth refilling on the side that was hit, the
                follow-on aggressive volume, and how fast events arrive.
  injection     A market order of chosen size is inserted at ordinary moments of the day,
                where no such order really happened (a counterfactual: there is no real
                continuation). The model's impact = run with the order minus run without it,
                on the same windows. Real orders of similar size, binned, give the reference
                impact-vs-size curve (computed on every aggressive order of the day).

A market order shows up in the data as several visible executions (LOBSTER type 4) with the
same timestamp against resting orders on one side: one group = one aggressive order. Its
direction is the opposite of the resting side (LOBSTER direction -1 = a sell order was hit =
buyer-initiated).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .data import Day, Window, windows_at
from .sim import BUY, EXEC, SELL, TICK, SimBook, Simulator

EVENTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)       # events after the shock
TIMES_NS = (1_000_000, 10_000_000, 100_000_000, 1_000_000_000, 10_000_000_000)  # 1 ms .. 10 s
NS = 1_000_000_000


@dataclass
class Shock:
    first: int    # day row of the first execution
    last: int     # day row of the last execution
    t: int        # ns since midnight
    sign: int     # +1 buyer-initiated (asks were hit), -1 seller-initiated
    size: int     # shares
    levels: int   # how many ticks the same-side best moved during the order


def aggressor_groups(msgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(first row, last row) of every aggressive order: runs of consecutive visible executions
    with the same timestamp against the same side."""
    m = np.asarray(msgs)
    ex = m[:, 1] == EXEC
    cont = np.zeros(len(m), dtype=bool)
    cont[1:] = ex[1:] & ex[:-1] & (m[1:, 0] == m[:-1, 0]) & (m[1:, 5] == m[:-1, 5])
    first = np.flatnonzero(ex & ~cont)
    is_last = ex & ~np.append(cont[1:], False)
    return first, np.flatnonzero(is_last)


def all_orders(day: Day) -> list[Shock]:
    """Every aggressive order of the day (no filtering)."""
    m, b = day.msgs, day.book
    first, last = aggressor_groups(m)
    csum = np.concatenate([[0], np.cumsum(m[:, 3])])
    sizes = csum[last + 1] - csum[first]
    out = []
    for f, l, s in zip(first.tolist(), last.tolist(), sizes.tolist()):
        sign = -int(m[f, 5])
        col = 0 if sign > 0 else 2                    # best ask moves when asks are hit
        before = b[f - 1, col] if f > 0 else b[f, col]
        moved = abs(int(b[l, col] - before)) // TICK if b[l, col + 1] > 0 else -1  # -1: side emptied
        out.append(Shock(f, l, int(m[f, 0]), sign, int(s), moved))
    return out


def select_shocks(day: Day, quantile: float = 0.99, min_size: int | None = None,
                  cond_len: int = 500, gen_len: int = 2000, gap: int = 1000,
                  margin_s: int = 300, max_n: int | None = None, seed: int = 0) -> list[Shock]:
    """Large aggressive orders: size >= the day's `quantile` of order sizes (or >= min_size),
    away from the open and close, with room for the context and the continuation. Of orders
    less than `gap` events apart only the first is kept: the later ones are part of the real
    reaction a generator has to produce."""
    orders = all_orders(day)
    sizes = np.array([o.size for o in orders])
    thr = min_size if min_size is not None else float(np.quantile(sizes, quantile))
    t0, t1 = (34_200 + margin_s) * NS, (57_600 - margin_s) * NS
    kept, last_row = [], -10 ** 12
    for o in orders:
        if o.size < thr:
            continue
        if o.first - last_row < gap:
            last_row = o.last  # extend the cluster
            continue
        last_row = o.last
        if t0 <= o.t < t1 and o.first >= cond_len and o.last + 1 + gen_len <= len(day.msgs):
            kept.append(o)
    if max_n is not None and len(kept) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(len(kept), max_n, replace=False))
        kept = [kept[i] for i in idx]
    return kept


def shock_windows(day: Day, shocks: list[Shock], cond_len: int = 500,
                  gen_len: int = 2000) -> list[Window]:
    """Context ends right after each shock's last execution, so the shock is the last thing a
    generator sees."""
    return windows_at(day, [s.last + 1 for s in shocks], cond_len, gen_len)


def inject(w: Window, sign: int, size: int, dt_ns: int = 1_000) -> tuple[Window, dict]:
    """Insert a market order of `size` shares (sign +1 buy, -1 sell) right after the window's
    context: it executes against the opposite side in price-time order, like a real
    marketable order, and becomes the last events of the context. Returns the new window and
    the reference values for measure()."""
    if not w.encoding[1]:
        raise ValueError("injection writes whole fills as size 0: needs the whole_fill encoding")
    sim = Simulator(SimBook.from_orders(w.state), w.t_prev)
    resting = SELL if sign > 0 else BUY
    bk = sim.book
    pre = w.cond_book[-1]
    msgs, books, acts = [], [], []
    left = int(size)
    while left > 0 and bk.n_levels(resting):
        price = bk.best(resting)
        ref = bk.ref_at(resting, price, 0, from_front=True)
        have = bk.side[resting][price].q[ref]
        take = min(have, left)
        msg, snap = sim.step(dt_ns if not msgs else 0, EXEC, resting, 0, 0,
                             0 if take == have else take)
        msgs.append(msg)
        books.append(snap)
        acts.append(sim.last_action)
        left -= take
    nw = replace(w, state=sim.book.to_orders(), t_prev=sim.t,
                 cond_msgs=np.vstack([w.cond_msgs, np.array(msgs, dtype=np.int64)]),
                 cond_book=np.vstack([w.cond_book, np.array(books, dtype=np.int64)]),
                 cond_actions=np.vstack([w.cond_actions, np.array(acts, dtype=np.int64)]))
    return nw, ref_point(pre, w.t_prev + dt_ns, sign, size - left)


def _mid_ticks(row: np.ndarray) -> float:
    if row[1] > 0 and row[3] > 0:
        return (row[0] + row[2]) / (2 * TICK)
    return np.nan


def ref_point(book_before: np.ndarray, t: int, sign: int, size: int) -> dict:
    """What a response is measured against: the book just before the order, its time and side."""
    col = 1 if sign > 0 else 3                      # volume at the best of the side that is hit
    return {"t": int(t), "sign": int(sign), "size": int(size),
            "mid": _mid_ticks(book_before), "depth": int(book_before[col])}


def shock_ref(day: Day, s: Shock) -> dict:
    return ref_point(day.book[s.first - 1], s.t, s.sign, s.size)


def _ffill_mid(book: np.ndarray, start: float) -> np.ndarray:
    two = (book[:, 1] > 0) & (book[:, 3] > 0)
    mid = np.where(two, (book[:, 0] + book[:, 2]) / (2 * TICK), np.nan)
    full = np.concatenate([[start], mid])
    idx = np.where(~np.isnan(full), np.arange(len(full)), 0)
    np.maximum.accumulate(idx, out=idx)
    return full[idx]                                  # [N+1]: after 0..N continuation events


def measure(msgs: np.ndarray, book: np.ndarray, last_ctx_book: np.ndarray, ref: dict,
            events=EVENTS, times=TIMES_NS) -> dict[str, np.ndarray]:
    """Response of one continuation (real or generated) to the order described by `ref`.

    move_*    signed mid change from just before the order, ticks (+ = continued in the
              order's direction); after k events (move_ev) or t after the order (move_t; nan
              when the run ends earlier, i.e. that horizon is not covered)
    spread_*  bid-ask spread, ticks
    depth_ev  visible volume at the best of the side that was hit, shares
    flow_ev   follow-on aggressive volume in the order's direction, shares (buys - sells,
              signed by the order)
    ms_ev     milliseconds from the order to the k-th event (how fast the market reacts)
    """
    m, b = np.asarray(msgs), np.asarray(book)
    s = ref["sign"]
    n = len(m)
    mid = _ffill_mid(b, _mid_ticks(last_ctx_book))
    spread = np.concatenate([[last_ctx_book[0] - last_ctx_book[2]], b[:, 0] - b[:, 2]]) / TICK
    two = np.concatenate([[last_ctx_book[1] > 0 and last_ctx_book[3] > 0],
                          (b[:, 1] > 0) & (b[:, 3] > 0)])
    spread = np.where(two, spread, np.nan)
    col = 1 if s > 0 else 3
    depth = np.concatenate([[last_ctx_book[col]], b[:, col]]).astype(float)
    ex = m[:, 1] == EXEC
    flow = np.concatenate([[0], np.cumsum(np.where(ex, -m[:, 5] * m[:, 3], 0))]) * s
    k = np.asarray(events)
    ok = k <= n
    kk = np.minimum(k, n)
    t_ev = np.concatenate([[ref["t"]], m[:, 0]])
    out = {
        "move_ev": np.where(ok, s * (mid[kk] - ref["mid"]), np.nan),
        "spread_ev": np.where(ok, spread[kk], np.nan),
        "depth_ev": np.where(ok, depth[kk], np.nan),
        "flow_ev": np.where(ok, flow[kk], np.nan),
        "ms_ev": np.where(ok, (t_ev[kk] - ref["t"]) / 1e6, np.nan),
    }
    tq = ref["t"] + np.asarray(times, dtype=np.int64)
    idx = np.searchsorted(m[:, 0], tq, side="right")   # number of events at or before t
    cov = t_ev[-1] >= tq                                # the run reaches t: state there is known
    out["move_t"] = np.where(cov, s * (mid[idx] - ref["mid"]), np.nan)
    out["spread_t"] = np.where(cov, spread[idx], np.nan)
    return out


def stack(rows: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def summarize(x: np.ndarray, n_boot: int = 1000, seed: int = 0, min_cover: float = 0.95) -> dict:
    """Mean over shocks with a 95% bootstrap interval, per horizon (columns). x: [n_shocks, H]
    (already averaged over samples of the same shock). A horizon covered by fewer than
    `min_cover` of the runs is reported as nan: the runs that reach it are a biased subset."""
    x = np.asarray(x, dtype=float)
    cover = np.mean(~np.isnan(x), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    with warnings.catch_warnings():   # all-nan columns are reported as nan below
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(x, 0)
        boots = np.stack([np.nanmean(x[i], 0) for i in idx])
        lo, hi = np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)
    bad = cover < min_cover
    f = lambda a: [None if (b or np.isnan(v)) else round(float(v), 4) for v, b in zip(a, bad)]  # noqa: E731
    return {"mean": f(mean), "lo": f(lo), "hi": f(hi), "cover": [round(float(c), 3) for c in cover]}


def tracking(gen: np.ndarray, real: np.ndarray, gen2: np.ndarray | None = None) -> dict:
    """Does the generator know *which* shocks were followed by big moves? Per horizon:
    corr(generated, real) across shocks; with a second independent sample per shock also
    corr(generated, generated'). For a generator that matched the true conditional law the two
    would be equal (both = share of the move's variance that the context predicts), so
    corr(gen, real) << corr(gen, gen') means the model is confident about things that do not
    happen, and corr(gen, real) > corr(gen, gen') means it misses predictable structure."""
    def corr(a, b):
        out = []
        for j in range(a.shape[1]):
            ok = ~np.isnan(a[:, j]) & ~np.isnan(b[:, j])
            if ok.sum() < 10 or a[ok, j].std() == 0 or b[ok, j].std() == 0:
                out.append(None)
            else:
                out.append(round(float(np.corrcoef(a[ok, j], b[ok, j])[0, 1]), 4))
        return out
    res = {"corr_real": corr(gen, real)}
    if gen2 is not None:
        res["corr_self"] = corr(gen, gen2)
    return res


def size_bins(day: Day, edges, times=TIMES_NS, margin_s: int = 300) -> dict:
    """Real impact vs order size from every aggressive order of the day: mean signed mid move
    from just before the order to t after it, per size bin, measured on the real stream."""
    m, b = day.msgs, day.book
    t = m[:, 0]
    mid = _ffill_mid(b, np.nan)[1:]
    lo_t, hi_t = (34_200 + margin_s) * NS, (57_600 - margin_s) * NS
    orders = [o for o in all_orders(day) if lo_t <= o.t < hi_t and o.first > 0]
    sizes = np.array([o.size for o in orders])
    out = {"edges": list(map(int, edges)), "bins": []}
    for a, c in zip(edges[:-1], edges[1:]):
        sel = [o for o, s in zip(orders, sizes) if a <= s < c]
        rows = []
        for o in sel:
            pre = mid[o.first - 1]
            idx = np.searchsorted(t, o.t + np.array(times), side="right") - 1
            rows.append(o.sign * (mid[idx] - pre))
        x = np.array(rows) if rows else np.zeros((0, len(times)))
        out["bins"].append({"lo": int(a), "hi": int(c), "n": len(sel),
                            "median_size": float(np.median(sizes[(sizes >= a) & (sizes < c)]))
                            if len(sel) else None,
                            **(summarize(x) if len(sel) >= 5 else {})})
    return out
