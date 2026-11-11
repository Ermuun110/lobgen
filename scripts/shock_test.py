"""Shock test (src/lobgen/shock.py): real large market orders of the test day, and injected ones.

Part 1, real shocks. Every top-1% aggressive order of Fri 2025-12-12 (first of each cluster,
up to --max-shocks) ends a window's context; each generator continues from the real book right
after it, --samples times. Compared with what really followed: signed mid move (ticks) at
1 ms .. 10 s and 1 .. 2000 events, spread, depth on the hit side, follow-on aggressive volume,
time to the k-th event. Tracking: corr(generated move, real move) across shocks, against
corr(sample 1, sample 2) of the same generator (see shock.tracking).

Part 2, injection. A market order of each size in --inject-sizes (default: the day's 90th,
99th and 99.9th percentile order size and 3x the last) is inserted at --inject-n ordinary
moments; impact = mean move with the order - mean move without it (same windows, same seeds).
The real reference is every aggressive order of the day, binned by size.

Run: uv run --group bench --group model python scripts/shock_test.py --model v2=checkpoints/v2.pt
Writes results/shock.json (merged per ticker and generator), results/shock_{T}.png.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from lobgen.baselines import HistoryBootstrap, Markov, train_pool
from lobgen.data import Day, make_windows
from lobgen.shock import (EVENTS, TIMES_NS, inject, measure, ref_point, select_shocks,
                          shock_ref, shock_windows, size_bins, stack, summarize, tracking)

LOB = Path("data/lobster")
STEM = "{t}_{d}_34200000_57600000_10.npz"
TEST, TRAIN = "2025-12-12", ["2025-12-08", "2025-12-09", "2025-12-10"]
METRICS = ["move_t", "spread_t", "move_ev", "spread_ev", "depth_ev", "flow_ev", "ms_ev"]


def nice(x: float) -> int:
    """Round to two significant digits."""
    e = 10 ** max(int(np.floor(np.log10(max(x, 1)))) - 1, 0)
    return int(round(x / e) * e)


def run_gen(g, W, refs, n_events, seed, chunk=200) -> dict[str, np.ndarray]:
    """Generate in chunks and measure at once (a 2000-event GOOG run with its book is ~0.6 MB)."""
    rows = []
    for k in range(0, len(W), chunk):
        out = g.generate(W[k:k + chunk], n_events, seed=seed + k)
        for w, r, o in zip(W[k:k + chunk], refs[k:k + chunk], out):
            rows.append(measure(o.msgs, o.book, w.cond_book[-1], r))
        del out
    return stack(rows)


def summary(per_shock: dict) -> dict:
    return {m: summarize(per_shock[m]) for m in METRICS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GOOG", "INTC"])
    ap.add_argument("--model", nargs="*", default=[], help="name=checkpoint pairs")
    ap.add_argument("--baselines", nargs="*", default=["history", "markov"])
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--max-shocks", type=int, default=300)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gen", type=int, default=2000)
    ap.add_argument("--inject-n", type=int, default=100)
    ap.add_argument("--inject-gen", type=int, default=1000)
    ap.add_argument("--inject-sizes", type=int, nargs="*", default=None)
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--market-dir", default="data/market", help="ETF quotes for market models")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shock.json"
    allres = json.loads(out_path.read_text()) if out_path.exists() else {}

    for t in a.tickers:
        t0 = time.time()
        day = Day.load(LOB / STEM.format(t=t, d=TEST))
        shocks = select_shocks(day, a.quantile, gen_len=a.gen, max_n=a.max_shocks)
        W = shock_windows(day, shocks, 500, a.gen)
        refs = [shock_ref(day, s) for s in shocks]
        res = allres.setdefault(t, {})
        sizes = np.array([s.size for s in shocks])
        res["_meta"] = {"test": TEST, "n_shocks": len(shocks), "quantile": a.quantile,
                        "min_size": int(sizes.min()), "median_size": float(np.median(sizes)),
                        "buy_share": float(np.mean([s.sign > 0 for s in shocks])),
                        "levels_moved_share": float(np.mean([s.levels > 0 for s in shocks])),
                        "samples": a.samples, "gen_len": a.gen,
                        "events": list(EVENTS), "times_ms": [x / 1e6 for x in TIMES_NS]}
        real = stack([measure(w.real_msgs, w.real_book, w.cond_book[-1], r) for w, r in zip(W, refs)])
        res["real"] = summary(real)
        print(f"{t}: {len(shocks)} shocks >= {sizes.min()} shares (median {np.median(sizes):.0f}), "
              f"real move at {[x / 1e6 for x in TIMES_NS]} ms: {res['real']['move_t']['mean']}",
              flush=True)

        # Real impact vs size from every aggressive order of the day.
        from lobgen.shock import all_orders
        osz = np.array([o.size for o in all_orders(day)])
        q = np.quantile(osz, [0.5, 0.9, 0.99, 0.999])
        edges = sorted({1, *map(int, q), int(osz.max()) + 1})
        res["real_by_size"] = size_bins(day, edges)
        inj_sizes = a.inject_sizes or sorted({nice(q[1]), nice(q[2]), nice(q[3]), nice(3 * q[3])})
        Wc = [] if a.no_inject else make_windows(day, a.inject_n, 500, a.inject_gen, seed=1)

        gens = []
        if any(b in a.baselines for b in ("history",)):
            gens.append(HistoryBootstrap())
        if "markov" in a.baselines:
            gens.append(Markov(train_pool([Day.load(LOB / STEM.format(t=t, d=d), book=False)
                                           for d in TRAIN])))
        if a.model:
            from lobgen.model import ModelGenerator, attach_market
            gens += [attach_market(ModelGenerator.load(p, device=a.device, name=n), TEST,
                                   a.market_dir)
                     for n, p in (m.split("=", 1) for m in a.model)]
        del day
        gc.collect()

        for g in gens:
            tg = time.time()
            samples = [run_gen(g, W, refs, a.gen, seed=1000 * s) for s in range(a.samples)]
            mean = {m: np.nanmean(np.stack([s[m] for s in samples]), 0) for m in METRICS}
            entry = summary(mean)
            entry["tracking"] = {m: tracking(samples[0][m], real[m],
                                             samples[1][m] if a.samples > 1 else None)
                                 for m in ("move_t", "move_ev")}
            if Wc:
                signs = [1 if i % 2 == 0 else -1 for i in range(len(Wc))]
                ctrl_refs = [ref_point(w.cond_book[-1], w.t_prev + 1_000, s, 0)
                             for w, s in zip(Wc, signs)]
                ctrl = run_gen(g, Wc, ctrl_refs, a.inject_gen, seed=7)
                inj = {"sizes": inj_sizes, "control": summarize(ctrl["move_t"]), "by_size": []}
                for size in inj_sizes:
                    pairs = [inject(w, s, size) for w, s in zip(Wc, signs)]
                    Wi, ri = [p[0] for p in pairs], [p[1] for p in pairs]
                    got = run_gen(g, Wi, ri, a.inject_gen, seed=7)
                    inj["by_size"].append({"size": size, "filled": float(np.mean([r["size"] for r in ri])),
                                           "move_t": summarize(got["move_t"]),
                                           "impact_t": summarize(got["move_t"] - ctrl["move_t"]),
                                           "spread_t": summarize(got["spread_t"])})
                entry["inject"] = inj
            res[g.name] = entry
            mt = entry["move_t"]["mean"]
            print(f"{t} {g.name:10s} move_t {mt}  corr {entry['tracking']['move_t']}  "
                  f"({time.time() - tg:.0f}s)", flush=True)
            out_path.write_text(json.dumps(allres, indent=1) + "\n")
        out_path.write_text(json.dumps(allres, indent=1) + "\n")
        from lobgen.plots import plot_shock
        plot_shock(res, t, out_dir / f"shock_{t}.png")
        print(f"{t} done in {(time.time() - t0) / 60:.1f} min", flush=True)
        del W, Wc, real
        gc.collect()


if __name__ == "__main__":
    main()
