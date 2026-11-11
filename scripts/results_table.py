"""Markdown results table from results/detective.json (pasted into README).

Headline column "caught by": the run length at which the detective's AUC first reaches 0.9,
interpolated on log run length ("<5" if already there at the shortest length, ">1000" if never).

Run: uv run python scripts/results_table.py [--lengths 5 10 50 100 1000]
"""

import argparse
import json
import math
from pathlib import Path

ORDER = ["same_day", "replay", "real_2025-12-11", "real_2025-12-08"]
BASELINES = ["history", "pooled", "markov"]
LABEL = {"same_day": "real, neighbouring window (floor)", "replay": "replay through simulator (check)",
         "real_2025-12-11": "real, other day (Thu)", "real_2025-12-08": "real, other day (Mon)",
         "history": "baseline: history bootstrap", "pooled": "baseline: pooled i.i.d.",
         "markov": "baseline: Markov"}


def caught_by(r: dict, thr: float = 0.9) -> str:
    pts = sorted((int(k), v["auc"]) for k, v in r.items())
    if pts[0][1] >= thr:
        return f"<{pts[0][0]}" if pts[0][0] > 1 else "1"
    for (l0, a0), (l1, a1) in zip(pts, pts[1:]):
        if a1 >= thr:
            x = math.log(l0) + (thr - a0) / (a1 - a0) * (math.log(l1) - math.log(l0))
            return f"{math.exp(x):.0f}"
    return f">{pts[-1][0]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[5, 10, 50, 100, 1000])
    ap.add_argument("--file", default="results/detective.json")
    a = ap.parse_args()
    res = json.loads(Path(a.file).read_text())
    for t, r in res.items():
        names = [n for n in ORDER + BASELINES if n in r]
        names += sorted(n for n in r if n not in names and not n.startswith("_"))
        meta = r.get("_meta", {})
        print(f"\n**{t}** — test day {meta.get('test')}, {meta.get('n_windows')} windows, "
              f"AUC (95% CI) by run length\n")
        print("| generator | " + " | ".join(f"{L} events" for L in a.lengths) + " | caught by (AUC 0.9) |")
        print("|---|" + "---|" * (len(a.lengths) + 1))
        for n in names:
            cells = []
            for L in a.lengths:
                v = r[n].get(str(L))
                cells.append("" if v is None else f"{v['auc']:.2f} ({v['auc_ci'][0]:.2f}-{v['auc_ci'][1]:.2f})")
            print(f"| {LABEL.get(n, 'model: ' + n)} | " + " | ".join(cells) + f" | {caught_by(r[n])} |")


if __name__ == "__main__":
    main()
