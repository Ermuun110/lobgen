"""Tracking test: given what the market did, does the generated stock follow the real one?

A generator cannot know news before it happens, and neither can anyone else. What it can do is
react to it: news reaches a single stock through the market (market.py), so a model that reads
the market should, given the market's real path, move the stock about as far as it really
moved. This script measures that on the held-out test day (Fri 2025-12-12, which had a
market-wide sell-off around 10:58 ET: GOOG -1.5% and INTC -3.7% on the day).

For --n windows spread over the day, each generator runs --samples times from the real book
for --horizon seconds of clock time. Compared with the real move of the stock over the same
interval (bps):
  corr      correlation across windows of the generator's mean move with the real move
  r2_zero   1 - sum (real - pred)^2 / sum real^2: skill against "the price stays put"
  std       std of single generated moves vs real moves (too calm / too wild)
Generators:
  <model>            a model that does not read the market (v2, v3): it cannot track
  <model>            a market model (v4) given the real market path
  <model>@frozen     the same model, the market frozen at the window start (no news): how much
                     of its tracking comes from the market input
  beta               linear baseline: real move = beta . (QQQ, SMH) move, OLS on the training
                     days at the same horizon; uses the same market information as v4
News scenario (--scenario-bps, default -50): the market model again on the same windows with
QQQ and SMH pushed an extra -50 bps over 10 s from the start (a headline). The model's
response = scenario move - real-market move; it is compared with the real stock's beta, from
the training days, to market moves of at least 10 bps.

Case study: the window that starts 30 s before the test day's worst one-minute QQQ move,
--case-samples runs over --case-seconds, mid paths drawn against the real path
(results/track_case_{T}.png).

Run: uv run --group bench --group model python scripts/track_test.py \
        --model v3=checkpoints/v3.pt v4=checkpoints/v4.pt [--n 60] [--horizon 10]
Needs data/market/ (notebooks/fetch_itch_colab.ipynb, market part). Writes results/track.json.
"""

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from lobgen.data import NS, Day, make_windows, windows_at
from lobgen.market import Market
from lobgen.shock import _ffill_mid

LOB = Path("data/lobster")
STEM = "{t}_{d}_34200000_57600000_10.npz"
TEST, TRAIN = "2025-12-12", ["2025-12-08", "2025-12-09", "2025-12-10"]


def day_mid(day: Day) -> tuple[np.ndarray, np.ndarray]:
    """(times, log mid) after every row of a day."""
    mid = _ffill_mid(day.book, np.nan)[1:]
    return day.msgs[:, 0], np.log(mid)


def log_mid_at(t: np.ndarray, x: np.ndarray, q) -> np.ndarray:
    k = np.searchsorted(t, np.asarray(q), side="right") - 1
    return np.where(k >= 0, x[np.maximum(k, 0)], np.nan)


def fit_beta(t: str, horizon_s: float, market_dir: str) -> dict:
    """OLS (no intercept) of the stock's horizon return on QQQ's and SMH's, training days,
    non-overlapping intervals 09:35-15:55; also the beta on large market moves only."""
    X, y = [], []
    for d in TRAIN:
        day = Day.load(LOB / STEM.format(t=t, d=d), actions=False)
        ts, x = day_mid(day)
        del day
        mk = Market.load(d, market_dir)
        grid = np.arange(34_500 * NS, 57_300 * NS, int(horizon_s * NS))
        sx = np.diff(log_mid_at(ts, x, grid)) * 1e4
        ex = np.stack([np.diff(s.value(grid)) * 1e4 for s in mk.series], 1)
        ok = ~np.isnan(sx) & ~np.isnan(ex).any(1)
        X.append(ex[ok])
        y.append(sx[ok])
        gc.collect()
    X, y = np.vstack(X), np.concatenate(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    big = np.abs(X[:, 0]) >= 10
    b_big = float(np.sum(X[big, 0] * y[big]) / np.sum(X[big, 0] ** 2)) if big.sum() >= 5 else None
    b_qqq = float(np.sum(X[:, 0] * y) / np.sum(X[:, 0] ** 2))
    return {"beta": beta.tolist(), "n": int(len(y)), "beta_qqq_only": b_qqq,
            "beta_qqq_big_moves": b_big, "n_big": int(big.sum()),
            "r2_in_sample": float(1 - np.sum((y - X @ beta) ** 2) / np.sum(y ** 2))}


def gen_moves(g, W, K, horizon_ns, n_events, seed, chunk=200) -> np.ndarray:
    """[K, len(W)] log-mid move (bps) of each generated run from its start to start + horizon
    (nan when the run ends before then). The K samples of a window share a batch."""
    rep = [w for _ in range(K) for w in W]
    out = []
    for k in range(0, len(rep), chunk):
        res = g.generate(rep[k:k + chunk], n_events, seed=seed + k)
        for w, o in zip(rep[k:k + chunk], res):
            mid = _ffill_mid(o.book, (w.cond_book[-1][0] + w.cond_book[-1][2]) / 2 / 100)
            if o.msgs[-1, 0] < w.t_prev + horizon_ns:
                out.append(np.nan)
                continue
            i = np.searchsorted(o.msgs[:, 0], w.t_prev + horizon_ns, side="right")
            out.append((np.log(mid[i]) - np.log(mid[0])) * 1e4)
        del res
    return np.array(out).reshape(K, len(W))


def gen_paths(g, W, n_events, seed, grid_ns) -> np.ndarray:
    """Mid paths (bps from the start) on a clock grid, one row per run."""
    res = g.generate(W, n_events, seed=seed)
    rows = []
    for w, o in zip(W, res):
        mid = _ffill_mid(o.book, (w.cond_book[-1][0] + w.cond_book[-1][2]) / 2 / 100)
        i = np.searchsorted(o.msgs[:, 0], w.t_prev + grid_ns, side="right")
        p = (np.log(mid[i]) - np.log(mid[0])) * 1e4
        p[w.t_prev + grid_ns > o.msgs[-1, 0]] = np.nan
        rows.append(p)
    return np.array(rows)


def scores(pred: np.ndarray, real: np.ndarray, single: np.ndarray | None = None) -> dict:
    ok = ~np.isnan(pred) & ~np.isnan(real)
    p, r = pred[ok], real[ok]
    out = {"n": int(ok.sum()), "coverage": float(ok.mean()),
           "corr": float(np.corrcoef(p, r)[0, 1]) if ok.sum() > 5 and p.std() > 0 else None,
           "r2_zero": float(1 - np.sum((r - p) ** 2) / np.sum(r ** 2)) if ok.sum() else None,
           "mae_bps": float(np.mean(np.abs(r - p))) if ok.sum() else None,
           "std_real": float(r.std()) if ok.sum() else None}
    if single is not None:
        s = single[~np.isnan(single)]
        out["std_single_run"] = float(s.std()) if len(s) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GOOG", "INTC"])
    ap.add_argument("--model", nargs="*", default=[], help="name=checkpoint pairs")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--horizon", type=float, default=10.0, help="seconds")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--slack", type=float, default=2.0,
                    help="generate this many times the real event count of the horizon")
    ap.add_argument("--scenario-bps", type=float, default=-50.0)
    ap.add_argument("--scenario-ramp", type=float, default=10.0)
    ap.add_argument("--case-samples", type=int, default=8)
    ap.add_argument("--case-seconds", type=float, default=30.0)
    ap.add_argument("--case-slack", type=float, default=1.5)
    ap.add_argument("--market-dir", default="data/market")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from lobgen.model import ModelGenerator
    from lobgen.plots import plot_track_case

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "track.json"
    allres = json.loads(out_path.read_text()) if out_path.exists() else {}
    H = int(a.horizon * NS)
    gens = [ModelGenerator.load(p, device=a.device, name=n) for n, p in
            (m.split("=", 1) for m in a.model)]
    for t in a.tickers:
        t_start = time.time()
        res = allres.setdefault(t, {})
        beta = fit_beta(t, a.horizon, a.market_dir)
        res["beta_fit"] = beta
        day = Day.load(LOB / STEM.format(t=t, d=TEST))
        mk = Market.load(TEST, a.market_dir)
        ts, x = day_mid(day)
        W = make_windows(day, a.n, 500, 1000, seed=2)
        t0s = np.array([w.t_prev for w in W])
        real = (log_mid_at(ts, x, t0s + H) - log_mid_at(ts, x, t0s)) * 1e4
        n_real = np.searchsorted(ts, t0s + H) - np.searchsorted(ts, t0s)
        n_events = int(np.ceil(a.slack * np.quantile(n_real, 0.9)))
        etf = np.stack([[(s.value(np.array([t0 + H]))[0] - s.value(np.array([t0]))[0]) * 1e4
                         for s in mk.series] for t0 in t0s])
        res["_meta"] = {"test": TEST, "n_windows": len(W), "horizon_s": a.horizon,
                        "samples": a.samples, "gen_events": n_events,
                        "real_events_median": float(np.median(n_real)),
                        "real_move_std_bps": float(np.nanstd(real)),
                        "qqq_move_std_bps": float(np.std(etf[:, 0]))}
        res["beta"] = scores(etf @ np.array(beta["beta"]), real)
        res["zero"] = scores(np.zeros_like(real), real)
        print(f"{t}: horizon {a.horizon}s, {len(W)} windows, {n_events} events per run; "
              f"beta {np.round(beta['beta'], 2)} -> corr {res['beta']['corr']:.3f}", flush=True)

        # Case-study window: 30 s before the worst one-minute QQQ move of the test day.
        q = mk.series[0]
        grid = np.arange(34_500 * NS, 57_000 * NS, NS)
        r60 = q.value(grid[60:]) - q.value(grid[:-60])
        t_case = int(grid[int(np.argmin(r60))] - 30 * NS)
        row = int(np.searchsorted(ts, t_case))
        case_w = windows_at(day, [row], 500, 1000)[0]
        case_grid = np.arange(0, int(a.case_seconds * NS) + 1, NS // 2)
        n_case = int(a.case_slack * (np.searchsorted(ts, t_case + case_grid[-1]) - row)) + 100
        case = {"t0_s": t_case / NS, "grid_s": (case_grid / NS).tolist(),
                "real": ((log_mid_at(ts, x, case_w.t_prev + case_grid)
                          - log_mid_at(ts, x, case_w.t_prev)) * 1e4).tolist(),
                "qqq": [(q.value(np.array([case_w.t_prev + g]))[0]
                         - q.value(np.array([case_w.t_prev]))[0]) * 1e4 for g in case_grid],
                "beta_qqq": beta["beta_qqq_only"], "runs": {}}
        if res.get("case", {}).get("t0_s") == case["t0_s"]:   # keep other models' runs
            case["runs"].update(res["case"]["runs"])
        del day
        gc.collect()

        for g in gens:
            tg = time.time()
            reads_market = g.model.cfg.market
            for w in W:
                w.market = mk if reads_market else None
            runs = gen_moves(g, W, a.samples, H, n_events, seed=0)
            with np.errstate(all="ignore"):
                res[g.name] = scores(np.nanmean(runs, 0), real, runs[0])
            print(f"{t} {g.name:16s} {res[g.name]}  ({time.time() - tg:.0f}s)", flush=True)
            out_path.write_text(json.dumps(allres, indent=1) + "\n")
            if reads_market:
                for w in W:
                    w.market = mk.frozen(w.t_prev)
                fr = gen_moves(g, W, a.samples, H, n_events, seed=0)
                with np.errstate(all="ignore"):
                    res[f"{g.name}@frozen"] = scores(np.nanmean(fr, 0), real, fr[0])
                print(f"{t} {g.name + '@frozen':16s} {res[f'{g.name}@frozen']}", flush=True)
            if reads_market and a.scenario_bps:
                for w in W:
                    w.market = mk.scenario(w.t_prev, a.scenario_bps, a.scenario_ramp)
                sc = gen_moves(g, W, a.samples, H, n_events, seed=0)   # same seeds as `runs`
                with np.errstate(all="ignore"):
                    resp = np.nanmean(sc - runs, 0)
                ok = ~np.isnan(resp)
                rng = np.random.default_rng(0)
                boot = [np.mean(rng.choice(resp[ok], ok.sum())) for _ in range(1000)]
                res[f"{g.name}@scenario"] = {
                    "scenario_bps": a.scenario_bps, "ramp_s": a.scenario_ramp, "n": int(ok.sum()),
                    "response_bps": float(np.mean(resp[ok])),
                    "response_ci": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
                    "implied_beta": float(np.mean(resp[ok]) / a.scenario_bps),
                    "real_beta_big_moves": beta["beta_qqq_big_moves"],
                    "real_beta_all": beta["beta_qqq_only"]}
                print(f"{t} {g.name}@scenario {res[f'{g.name}@scenario']}", flush=True)
            for w in W:
                w.market = None
            # case study: K runs with the real market (and K with it frozen) in one batch
            conds = [(g.name, mk if reads_market else None)]
            if reads_market:
                conds.append((f"{g.name}@frozen", mk.frozen(case_w.t_prev)))
            cw = [replace(case_w, market=m) for _, m in conds for _ in range(a.case_samples)]
            paths = gen_paths(g, cw, n_case, 99, case_grid)
            for j, (name, _) in enumerate(conds):
                case["runs"][name] = paths[j * a.case_samples:(j + 1) * a.case_samples].tolist()
            out_path.write_text(json.dumps(allres | {t: res | {"case": case}}, indent=1) + "\n")
        res["case"] = case
        out_path.write_text(json.dumps(allres, indent=1) + "\n")
        plot_track_case(case, t, out_dir / f"track_case_{t}.png")
        print(f"{t} done in {(time.time() - t_start) / 60:.1f} min", flush=True)
        del W
        gc.collect()


if __name__ == "__main__":
    main()
