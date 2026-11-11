"""Which orders do deletes hit? Real vs generated runs.

For every delete, the target's recency rank: how many live orders on the same side were
submitted after it (0 = the newest live order on that side). Real deletes are aimed: in GOOG a
quoting ladder of 100-share orders is pulled a few events after being placed. Generators that
pick a delete by (level, queue position) reach those orders only indirectly.

Run: uv run --group model python scripts/delete_targets.py [--model checkpoints/v1.pt]
Writes results/delete_targets.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sortedcontainers import SortedList

from lobgen.baselines import HistoryBootstrap
from lobgen.data import Day, make_windows

STEM = "data/lobster/{t}_2025-12-12_34200000_57600000_10.npz"


def delete_targets(state: np.ndarray, msgs: np.ndarray) -> list[tuple[int, int]]:
    """(recency rank, shares) of every delete in `msgs`, replayed from order-level `state`."""
    live, rec, out = {}, {1: SortedList(), -1: SortedList()}, []
    for k, (ref, side, _price, sh) in enumerate(state.tolist()):  # priority order = age order
        live[ref] = (side, k - len(state), sh)
        rec[side].add(k - len(state))
    for i, (_t, typ, ref, size, _price, side) in enumerate(msgs.tolist()):
        if typ == 1:
            live[ref] = (side, i, size)
            rec[side].add(i)
        elif typ in (2, 4) and ref in live:
            s, q, sh = live[ref]
            live[ref] = (s, q, sh - size)
            if sh - size <= 0:
                live.pop(ref)
                rec[s].remove(q)
        elif typ == 3 and ref in live:
            s, q, sh = live.pop(ref)
            out.append((len(rec[s]) - rec[s].bisect_right(q), sh))
            rec[s].remove(q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GOOG", "INTC"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--model", default="checkpoints/v1.pt")
    ap.add_argument("--model-name", default="v1")
    a = ap.parse_args()
    out = {}
    for t in a.tickers:
        W = make_windows(Day.load(STEM.format(t=t)), a.n, 500, a.gen, seed=1)
        runs = {"real": [w.real_msgs for w in W],
                "history": [g.msgs for g in HistoryBootstrap().generate(W, a.gen)]}
        if a.model:
            from lobgen.model import ModelGenerator
            gen = ModelGenerator.load(a.model, name=a.model_name, batch=a.n)
            runs[a.model_name] = [g.msgs for g in gen.generate(W, a.gen)]
        for name, ms in runs.items():
            r = np.array([x for w, m in zip(W, ms) for x in delete_targets(w.state, m)])
            res = {"deletes": len(r), "rank_le4": round(float(np.mean(r[:, 0] <= 4)), 3),
                   "rank_le9": round(float(np.mean(r[:, 0] <= 9)), 3),
                   "size100": round(float(np.mean(r[:, 1] == 100)), 3),
                   "log10_size": round(float(np.log10(np.maximum(r[:, 1], 1)).mean()), 3)}
            out.setdefault(t, {})[name] = res
            print(t, f"{name:8s}", res, flush=True)
    Path("results/delete_targets.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
