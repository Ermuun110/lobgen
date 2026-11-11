"""Is each failure of a model already there at the first generated event, or does it grow as the
model keeps generating?

For a few event-level rates (100-share submits, submits inside a wide spread, zero gaps, trades,
pointer cancels) this compares, on the detective's test windows (Fri 2025-12-12):

  one-step   the model's predicted rate on REAL contexts (teacher forcing, real continuation
             events as targets), against the real rate on the same events
  rollout    the rate in the model's own 1000-event runs, by position in the run, against the
             real continuations of the same windows

Right at one step but drifting in the rollout = the model reacts to its own output (exposure
bias); wrong at one step = the model itself is miscalibrated.

Run: uv run --group bench --group model python scripts/diagnose_rollout.py [--model checkpoints/v2.pt]
Writes results/diagnose_rollout.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lobgen.data import Day, make_windows
from lobgen.features import token_features
from lobgen.model import ModelGenerator
from lobgen.sim import CANCEL, DELETE, DT, EXEC, PRICE, QPOS, SIDE, SIZE, SUBMIT, TICK, TYPE

LOB = Path("data/lobster")
STEM = "{t}_{d}_34200000_57600000_10.npz"
TEST = "2025-12-12"
BUCKETS = [(0, 50), (50, 200), (200, 500), (500, 1000)]
RATES = ["submit_100", "inside_when_wide", "dt0", "trade", "pointer_cancel"]


def _two_sided_spread(book_before: np.ndarray) -> np.ndarray:
    two = (book_before[:, 1] > 0) & (book_before[:, 3] > 0)
    return np.where(two, (book_before[:, 0] - book_before[:, 2]) / TICK, np.nan)


def masks(acts: np.ndarray, book_before: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """rate name -> (events it is measured on, events that count)."""
    typ = acts[:, TYPE]
    sub = typ == SUBMIT
    wide = sub & (_two_sided_spread(book_before) >= 2)
    can = (typ == CANCEL) | (typ == DELETE)
    every = np.ones(len(acts), dtype=bool)
    return {
        "submit_100": (sub, acts[:, SIZE] == 100),
        "inside_when_wide": (wide, acts[:, PRICE] < 0),
        "dt0": (every, acts[:, DT] == 0),
        "trade": (every, typ == EXEC),
        "pointer_cancel": (can, acts[:, PRICE] < 0),   # negative price field = pointer form
    }


def pooled(runs) -> dict[str, float]:
    """runs: list of (acts, book_before); rate = pooled share over all runs."""
    num, den = dict.fromkeys(RATES, 0), dict.fromkeys(RATES, 0)
    for acts, bb in runs:
        for k, (on, hit) in masks(acts, bb).items():
            num[k] += int((on & hit).sum())
            den[k] += int(on.sum())
    return {k: num[k] / den[k] if den[k] else float("nan") for k in RATES}


def by_bucket(seqs) -> dict[str, dict[str, float]]:
    """seqs: list of (acts, book_before) for whole runs -> rates per position bucket."""
    return {f"{a}-{b}": pooled([(x[a:b], bb[a:b]) for x, bb in seqs]) for a, b in BUCKETS}


@torch.no_grad()
def one_step(model, W, per_window: int, rng) -> tuple[dict, dict]:
    """Predicted vs real rates at `per_window` random continuation positions of every window."""
    ctx = model.cfg.ctx
    cats, conts, tgts, bbs = [], [], [], []
    for w in W:
        acts = np.vstack([w.cond_actions, w.real_actions])
        msgs = np.vstack([w.cond_msgs, w.real_msgs])
        book = np.vstack([w.cond_book, w.real_book])
        c, x, _ = token_features(acts, msgs, book, w.feat_state)
        n0 = len(w.cond_actions)
        for j in rng.choice(len(w.real_actions), per_window, replace=False):
            i = n0 + int(j)                       # target event; context = tokens i-ctx .. i-1
            cats.append(c[i - ctx:i])
            conts.append(x[i - ctx:i])
            tgts.append(acts[i])
            bbs.append(book[i - 1])
    tgt_np, bb = np.array(tgts), np.array(bbs)
    pred_num, real_runs = dict.fromkeys(RATES, 0.0), [(tgt_np, bb)]
    on_all = masks(tgt_np, bb)
    for s in range(0, len(tgts), 256):
        sl = slice(s, s + 256)
        cat = torch.from_numpy(np.stack(cats[sl])).long()
        cont = torch.from_numpy(np.stack(conts[sl])).float()
        tgt = torch.from_numpy(tgt_np[sl]).long()
        h = model.trunk(cat, cont)[:, -1]
        typ, side = tgt[:, TYPE], tgt[:, SIDE]
        ts = (typ.clamp(1, 5) - 1) * 2 + (side > 0).long()
        c1, c2, c3, c4 = model._cond(h, ts, tgt[:, PRICE], tgt[:, QPOS], tgt[:, SIZE])
        B = len(tgt)
        p_ts = torch.softmax(model.head_ts(h).float(), -1)
        p100 = model.head_size.log_prob(c3, torch.full_like(typ, 100)).exp()
        neg = torch.arange(-40, 0)
        p_neg = model.head_p.log_prob(c1[:, None].expand(-1, 40, -1), neg[None].expand(B, 40)).exp().sum(1)
        pdt0 = model.head_dt.log_prob(c4, torch.zeros_like(typ)).exp()
        pred = {"submit_100": p100, "inside_when_wide": p_neg, "dt0": pdt0,
                "trade": p_ts[:, 6:8].sum(1), "pointer_cancel": p_neg}
        for k in RATES:
            on = torch.from_numpy(on_all[k][0][sl])
            pred_num[k] += float(pred[k][on].sum())
    pred_rate = {k: pred_num[k] / max(int(on_all[k][0].sum()), 1) for k in RATES}
    return pred_rate, pooled(real_runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/v2.pt")
    ap.add_argument("--tickers", nargs="+", default=["GOOG", "INTC"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gen", type=int, default=1000)
    ap.add_argument("--per-window", type=int, default=20)
    a = ap.parse_args()
    gen = ModelGenerator.load(a.model, device="cpu", name=Path(a.model).stem)
    model = gen.model.eval()
    out = {"model": a.model, "n_windows": a.n}
    for t in a.tickers:
        day = Day.load(LOB / STEM.format(t=t, d=TEST))
        W = make_windows(day, a.n, 500, a.gen, seed=0, feat_states=True)
        del day
        pred, real1 = one_step(model, W, a.per_window, np.random.default_rng(1))
        G = gen.generate(W, a.gen)
        fake = [(g.actions, np.vstack([w.cond_book[-1:], g.book[:-1]])) for w, g in zip(W, G)]
        real = [(w.real_actions, np.vstack([w.cond_book[-1:], w.real_book[:-1]])) for w in W]
        trades_in_ctx = float(np.mean([(w.cond_actions[-model.cfg.ctx:, TYPE] == EXEC).sum() for w in W]))
        out[t] = {"one_step": {"model": pred, "real": real1},
                  "rollout": {"model": by_bucket(fake), "real": by_bucket(real)},
                  "visible_trades_in_context": trades_in_ctx}
        print(f"\n**{t}** ({a.n} windows; the model's {model.cfg.ctx}-event context holds "
              f"{trades_in_ctx:.1f} visible trades on average)\n")
        print("| rate | real | one-step model | " + " | ".join(f"run {k}" for k in out[t]["rollout"]["model"])
              + " | real by bucket |")
        print("|---|---|---|" + "---|" * (len(BUCKETS) + 1))
        for k in RATES:
            rb = out[t]["rollout"]
            print(f"| {k} | {real1[k]:.3f} | {pred[k]:.3f} | "
                  + " | ".join(f"{rb['model'][b][k]:.3f}" for b in rb["model"])
                  + " | " + " / ".join(f"{rb['real'][b][k]:.2f}" for b in rb["real"]) + " |", flush=True)
    Path("results/diagnose_rollout.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
