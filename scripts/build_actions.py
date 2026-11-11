"""Encode every converted day into simulator actions and prove the round trip is exact.

For each data/lobster/{T}_{date}_..._10.npz:
  1. rebuild the order-level book at 09:30 from the filtered ITCH stream (pre-open orders);
  2. encode all LOBSTER rows into actions with a strict simulator;
  3. decode the actions with fresh order ids and compare every message and book row.
Writes {stem}_actions.npz (actions, inexact, init_orders) next to the day file and
results/sim_replay.json. Exit 1 if any non-inexact row fails to reproduce.

Run: uv run python scripts/build_actions.py [--workers 2]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lobgen.lobster import preopen_orders
from lobgen.sim import PTR_RANKS, encode, verify_decode

LOB = Path("data/lobster")
FILTERED = Path("data/filtered")


def one(npz: str, pointer: int = PTR_RANKS, whole_fill: bool = True) -> tuple[str, dict]:
    p = Path(npz)
    ticker, date = p.name.split("_")[:2]
    t0 = time.time()
    init = preopen_orders(FILTERED / f"{date}.itch.gz", ticker)
    z = np.load(p)
    msgs, book, levels = z["messages"], z["book"], int(z["levels"])
    actions, inexact = encode(msgs, init, levels=levels, pointer=pointer, whole_fill=whole_fill)
    np.savez_compressed(p.with_name(p.stem + "_actions.npz"), actions=actions, inexact=inexact,
                        init_orders=init, pointer=pointer, whole_fill=whole_fill)
    res = verify_decode(msgs, book, init, actions, inexact, levels)
    res["init_orders"] = len(init)
    res["pointer"], res["whole_fill"] = pointer, whole_fill
    ex = actions[:, 1] == 4
    res["whole_fill_share"] = round(float(np.mean(actions[ex, 5] == 0)), 4) if ex.any() else None
    res["pointer_share"] = round(float(np.mean(actions[:, 3][np.isin(actions[:, 1], (2, 3))] < 0)), 4)
    res["seconds"] = round(time.time() - t0, 1)
    return f"{ticker}_{date}", res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pointer", type=int, default=PTR_RANKS,
                    help="cancels/deletes of the N newest orders use a pointer (0 = the v1 encoding)")
    ap.add_argument("--no-whole-fill", action="store_true",
                    help="executions keep their share count (with --pointer 0: the v1 encoding)")
    a = ap.parse_args()
    days = sorted(str(p) for p in LOB.glob("*_34200000_57600000_10.npz"))
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for name, res in ex.map(one, days, [a.pointer] * len(days),
                                [not a.no_whole_fill] * len(days)):
            results[name] = res
            print(name, json.dumps(res), flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/sim_replay.json").write_text(json.dumps(results, indent=2) + "\n")
    bad = sum(r["message_mismatches"] + r["id_mismatches"] + r["book_mismatches"]
              for r in results.values())
    print("ALL EXACT" if not bad else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
