"""Why are generators caught after only a few events? Per-feature real-vs-fake AUC at short L.

For each generator, every detective feature is scored on its own: univariate AUC of real vs
generated runs at run length L (0.5 = the feature carries no signal; reported as
max(AUC, 1 - AUC) so direction does not matter), with the real and generated means. Single-feature
AUCs are easier to read than permutation importance, which splits credit between correlated
features and drops to 0 when several features each separate perfectly.

Run: OMP_NUM_THREADS=2 uv run --group bench python scripts/diagnose_short.py --tickers GOOG \
        [--n 300] [--lengths 5 10] [--model checkpoints/v1.pt --model-name v1]
Writes results/diagnose_short.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lobgen.baselines import HistoryBootstrap, Markov, PooledIID, RealOtherDay, train_pool
from lobgen.data import Day, make_windows
from lobgen.detective import feature_matrix

LOB = Path("data/lobster")
STEM = "{t}_{d}_34200000_57600000_10.npz"
TEST, TRAIN = "2025-12-12", ["2025-12-08", "2025-12-09", "2025-12-10"]


def feature_aucs(real, fake, L: int, top: int) -> list[dict]:
    Xr, names = feature_matrix(real, L)
    Xf, _ = feature_matrix(fake, L)
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xf))]
    rows = []
    for k, name in enumerate(names):
        x = np.r_[Xr[:, k], Xf[:, k]]
        nan = np.isnan(x)
        # A missing value (e.g. "no submits in the run") is itself a value: put it below all others.
        x = np.where(nan, np.nanmin(x) - 1 if (~nan).any() else 0.0, x)
        auc = 0.5 if np.ptp(x) == 0 else roc_auc_score(y, x)
        rows.append({"feature": name, "auc": round(max(auc, 1 - auc), 3),
                     "real": round(float(np.nanmean(Xr[:, k])), 3) if (~np.isnan(Xr[:, k])).any() else None,
                     "fake": round(float(np.nanmean(Xf[:, k])), 3) if (~np.isnan(Xf[:, k])).any() else None})
    return sorted(rows, key=lambda r: -r["auc"])[:top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GOOG"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--lengths", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--model")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = {}
    for t in a.tickers:
        test = Day.load(LOB / STEM.format(t=t, d=TEST))
        W = make_windows(test, a.n, 500, a.gen, seed=0)
        real = [(w.real_msgs, w.real_book) for w in W]
        pool = train_pool([Day.load(LOB / STEM.format(t=t, d=d), book=False) for d in TRAIN])
        gens = [RealOtherDay(Day.load(LOB / STEM.format(t=t, d="2025-12-08"), actions=False)),
                HistoryBootstrap(), PooledIID(pool), Markov(pool)]
        if a.model:
            from lobgen.model import ModelGenerator
            gens.append(ModelGenerator.load(a.model, device=a.device, name=a.model_name))
        res = out.setdefault(t, {})
        for g in gens:
            fake = [(o.msgs, o.book) for o in g.generate(W, a.gen, seed=0)]
            res[g.name] = {L: feature_aucs(real, fake, L, a.top) for L in a.lengths}
            for L in a.lengths:
                print(f"{t} {g.name} L={L}: " + ", ".join(
                    f"{r['feature']} {r['auc']:.2f} ({r['real']} vs {r['fake']})"
                    for r in res[g.name][L][:5]), flush=True)
    Path("results/diagnose_short.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
