"""Detective AUC vs run length for every generator, plus the two real-data references.

Test day Fri 2025-12-12. Windows: n stratified across the day (5 min from open/close),
cond_len real events of context, gen_len events generated. Every generator continues from the
real order-level book at the window start. References:
  same_day   real runs of neighbouring windows (pairs of adjacent windows, ~20 s apart) - what a
             perfect conditional generator would score; should be ~0.5.
  real_<d>   real runs of day d at the same time of day - a generator that knows "a typical
             day" but not *this* day. Mon 12-08 and Thu 12-11.
  replay     the real run re-played through the simulator - 0.5 by construction (sanity check).
Baselines are trained on Mon-Wed (12-08..12-10).

Results are merged into results/detective.json (so model runs can be added later with
--model) and drawn to results/detective_{ticker}.png.

Run: uv run --group bench python scripts/run_detective.py [--tickers GOOG INTC] [--n 1000]
"""

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lobgen.baselines import HistoryBootstrap, Markov, PooledIID, RealOtherDay, Replay, train_pool
from lobgen.data import Day, make_windows
from lobgen.detective import detect
from lobgen.plots import plot_detective

LOB = Path("data/lobster")
OUT = Path("results")
STEM = "{t}_{d}_34200000_57600000_10.npz"
TEST, TRAIN, OTHERS = "2025-12-12", ["2025-12-08", "2025-12-09", "2025-12-10"], ["2025-12-08", "2025-12-11"]
LENGTHS = [5, 10, 20, 50, 100, 200, 500, 1000]


def repairs_per_event(gen, n_events) -> dict:
    tot = sum((g.repairs for g in gen), Counter())
    n = len(gen) * n_events
    return {k: round(v / n, 5) for k, v in tot.most_common()} | {
        "skipped_per_event": round(sum(g.skipped for g in gen) / n, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GOOG", "INTC"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cond", type=int, default=500)
    ap.add_argument("--gen", type=int, default=1000)
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--only", nargs="*", help="generator names to (re)run; default all")
    ap.add_argument("--model", help="checkpoint of a trained model (adds generator 'model')")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / "detective.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    want = (lambda name: True) if not a.only else (lambda name: name in a.only)

    for t in a.tickers:
        res = allres.setdefault(t, {})
        meta = res.setdefault("_meta", {})
        meta.update({"test": TEST, "train": TRAIN, "n_windows": a.n, "cond_len": a.cond,
                     "gen_len": a.gen, "seed": a.seed, "repairs": meta.get("repairs", {})})
        test = Day.load(LOB / STEM.format(t=t, d=TEST))
        W = make_windows(test, a.n, a.cond, a.gen, seed=a.seed)
        real = [(w.real_msgs, w.real_book) for w in W]

        def run(name, fake, real_=real, rg=None, fg=None):
            t0 = time.time()
            res[name] = detect(real_, fake, a.lengths, rg, fg, seed=a.seed)
            print(f"{t} {name:18s} " + " ".join(f"{L}:{res[name][L]['auc']:.3f}"
                                                for L in a.lengths)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

        if want("same_day"):
            k = np.arange(len(W) // 2)
            run("same_day", real[1::2][:len(k)], real[0::2][:len(k)], k, k)

        for d in OTHERS:
            name = f"real_{d}"
            if want(name):
                other = Day.load(LOB / STEM.format(t=t, d=d), actions=False)
                gen = RealOtherDay(other).generate(W, a.gen)
                del other
                gc.collect()
                run(name, [(g.msgs, g.book) for g in gen])

        gens = []
        if want("replay"):
            gens.append(Replay())
        if want("history"):
            gens.append(HistoryBootstrap())
        if any(want(n) for n in ("pooled", "markov")):
            pool = train_pool([Day.load(LOB / STEM.format(t=t, d=d), book=False) for d in TRAIN])
            if want("pooled"):
                gens.append(PooledIID(pool))
            if want("markov"):
                gens.append(Markov(pool))
        if a.model:
            from lobgen.model import ModelGenerator, attach_market
            gens.append(attach_market(ModelGenerator.load(a.model, device=a.device,
                                                          name=a.model_name), TEST))

        for g in gens:
            t0 = time.time()
            out = g.generate(W, a.gen, seed=a.seed)
            print(f"{t} {g.name:18s} generated in {time.time() - t0:.0f}s", flush=True)
            meta["repairs"][g.name] = repairs_per_event(out, a.gen)
            run(g.name, [(o.msgs, o.book) for o in out])
            del out
        path.write_text(json.dumps(allres, indent=1) + "\n")
        plot_detective({k: v for k, v in res.items() if not k.startswith("_")}, t,
                       OUT / f"detective_{t}.png")
        del test, W, real
        gc.collect()


if __name__ == "__main__":
    main()
