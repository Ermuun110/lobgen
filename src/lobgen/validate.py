"""Invariant checks and summary stats for converted LOBSTER data.

Without LOBSTER to diff against, correctness rests on invariants that must hold for any
faithful conversion:
  1. Each orderbook row differs from the previous one exactly by the event on that row
     (+size at its price for a submit, -size for cancel/delete/execution, no change for
     hidden executions, crosses and halts), wherever both rows show that price.
  2. Times never decrease; sizes and prices are positive for order events.
  3. The book is not crossed during continuous trading (it can be before the opening cross).
"""

from __future__ import annotations

import numpy as np

from .book import EMPTY_ASK, EMPTY_BID

_SIGN = {1: 1, 2: -1, 3: -1, 4: -1}
NS = 1_000_000_000


def _side_levels(row: np.ndarray, levels: int):
    asks = {int(row[4 * i]): int(row[4 * i + 1]) for i in range(levels) if row[4 * i + 1] > 0}
    bids = {int(row[4 * i + 2]): int(row[4 * i + 3]) for i in range(levels) if row[4 * i + 3] > 0}
    return asks, bids


def check_book_transitions(msgs: np.ndarray, book: np.ndarray, levels: int, max_examples: int = 5) -> dict:
    bad, examples = 0, []
    prev_a, prev_b = _side_levels(book[0], levels)
    for i in range(1, len(msgs)):
        cur_a, cur_b = _side_levels(book[i], levels)
        _, typ, _, size, price, d = (int(x) for x in msgs[i])
        for side, prev, cur in ((-1, prev_a, cur_a), (1, prev_b, cur_b)):
            # Prices are only comparable inside the range both rows fully show.
            full_prev, full_cur = len(prev) == levels, len(cur) == levels
            if side == -1:
                lim = min(max(prev) if full_prev else EMPTY_ASK, max(cur) if full_cur else EMPTY_ASK)
                inside = lambda p: p <= lim  # noqa: E731
            else:
                lim = max(min(prev) if full_prev else EMPTY_BID, min(cur) if full_cur else EMPTY_BID)
                inside = lambda p: p >= lim  # noqa: E731
            expected = {}
            if typ in _SIGN and d == side and inside(price):
                expected[price] = _SIGN[typ] * size
            diff = {}
            for p in set(prev) | set(cur):
                if inside(p):
                    delta = cur.get(p, 0) - prev.get(p, 0)
                    if delta:
                        diff[p] = delta
            if diff != expected:
                bad += 1
                if len(examples) < max_examples:
                    examples.append({"row": i, "side": side, "msg": msgs[i].tolist(),
                                     "expected": expected, "got": diff})
        prev_a, prev_b = cur_a, cur_b
    return {"rows_checked": len(msgs) - 1, "violations": bad, "examples": examples}


def check_messages(msgs: np.ndarray) -> dict:
    t, typ, size, price, d = msgs[:, 0], msgs[:, 1], msgs[:, 3], msgs[:, 4], msgs[:, 5]
    order_ev = np.isin(typ, [1, 2, 3, 4, 5])
    return {
        "time_decreases": int((np.diff(t) < 0).sum()),
        "nonpositive_size": int((size[order_ev] <= 0).sum()),
        "nonpositive_price": int((price[order_ev] <= 0).sum()),
        "bad_direction": int((~np.isin(d[order_ev], [1, -1])).sum()),
    }


def summary(msgs: np.ndarray, book: np.ndarray, tick: int = 100) -> dict:
    typ = msgs[:, 1]
    n = len(msgs)
    ask, askv, bid, bidv = book[:, 0], book[:, 1], book[:, 2], book[:, 3]
    two_sided = (askv > 0) & (bidv > 0)
    spread_ticks = (ask[two_sided] - bid[two_sided]) // tick
    dt = np.diff(msgs[:, 0])
    # Submissions priced exactly at the touch of their own side (point mass in the post).
    prev_ask, prev_bid = ask[:-1], bid[:-1]
    sub = typ[1:] == 1
    at_touch = np.where(msgs[1:, 5] == 1, msgs[1:, 4] == prev_bid, msgs[1:, 4] == prev_ask)
    vals, counts = np.unique(typ, return_counts=True)
    sp_vals, sp_counts = np.unique(np.clip(spread_ticks, -5, 10), return_counts=True)
    return {
        "events": n,
        "first_time_s": float(msgs[0, 0] / NS) if n else None,
        "last_time_s": float(msgs[-1, 0] / NS) if n else None,
        "event_share": {int(v): round(c / n, 4) for v, c in zip(vals, counts)},
        "trade_share_visible_plus_hidden": round(float(np.isin(typ, [4, 5]).mean()), 4),
        "dt_zero_share": round(float((dt == 0).mean()), 4) if len(dt) else None,
        "dt_median_us": float(np.median(dt) / 1e3) if len(dt) else None,
        "submit_at_touch_share": round(float(at_touch[sub].mean()), 4) if sub.any() else None,
        "spread_ticks_dist (clipped -5..10)": {int(v): round(c / len(spread_ticks), 4)
                                              for v, c in zip(sp_vals, sp_counts)},
        "crossed_or_locked_rows": int((spread_ticks <= 0).sum()),
    }
