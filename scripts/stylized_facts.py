"""Stylized facts (spread, return tails, volatility clustering, trade-sign memory, price impact,
depth profile, ...) of real runs vs other days vs generators, on the detective's test windows.

Each generator continues the same windows of the test day (Fri 2025-12-12) from the real book;
facts are pooled over windows (src/lobgen/facts.py). "real_2025-12-08" (another real day at
the same clock times) shows how much the facts move between two real days: a discrepancy that
size is not a model failure.

Run: uv run --group bench --group model python scripts/stylized_facts.py \
        [--n 300] [--model v2=checkpoints/v2.pt full=checkpoints/full.pt]
Writes results/facts.json, results/facts_{T}.png and prints a markdown discrepancy table.
"""

import argparse
import json
from pathlib import Path

from lobgen.baselines import HistoryBootstrap, Markov, RealOtherDay, train_pool
from lobgen.data import Day, make_windows
from lobgen.facts import discrepancies, stylized_facts
from lobgen.plots import plot_facts

LOB = Path("data/lobster")
STEM = "{t}_{d}_34200000_57600000_10.npz"
TEST, TRAIN = "2025-12-12", ["2025-12-08", "2025-12-09", "2025-12-10"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GOOG", "INTC"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gen", type=int, default=1000)
    ap.add_argument("--model", nargs="*", default=[], help="name=checkpoint pairs")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out_path = Path("results/facts.json")
    allres = json.loads(out_path.read_text()) if out_path.exists() else {}
    for t in a.tickers:
        test = Day.load(LOB / STEM.format(t=t, d=TEST))
        W = make_windows(test, a.n, 500, a.gen, seed=0, feat_states=bool(a.model))
        facts = {"real": stylized_facts([(w.real_msgs, w.real_book) for w in W])}
        gens = [RealOtherDay(Day.load(LOB / STEM.format(t=t, d="2025-12-08"), actions=False)),
                HistoryBootstrap(),
                Markov(train_pool([Day.load(LOB / STEM.format(t=t, d=d), book=False) for d in TRAIN]))]
        if a.model:
            from lobgen.model import ModelGenerator, attach_market
            gens += [attach_market(ModelGenerator.load(p, device=a.device, name=n), TEST)
                     for n, p in (m.split("=", 1) for m in a.model)]
        for g in gens:
            facts[g.name] = stylized_facts([(o.msgs, o.book) for o in g.generate(W, a.gen)])
            print(t, g.name, "done", flush=True)
        res = allres.setdefault(t, {})
        res["facts"] = facts
        res["discrepancy"] = {n: discrepancies(facts["real"], f) for n, f in facts.items() if n != "real"}
        plot_facts(facts, t, f"results/facts_{t}.png")
        cols = list(next(iter(res["discrepancy"].values())))
        print(f"\n**{t}**: discrepancy from the real continuations (0 = identical)\n")
        print("| fact | " + " | ".join(res["discrepancy"]) + " |")
        print("|---|" + "---|" * len(res["discrepancy"]))
        for c in cols:
            print(f"| {c} | " + " | ".join(f"{d[c]:.3f}" for d in res["discrepancy"].values()) + " |")
        out_path.write_text(json.dumps(allres, indent=1) + "\n")


if __name__ == "__main__":
    main()
