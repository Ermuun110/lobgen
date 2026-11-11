"""Train the event transformer (step 4). Runs locally (CPU / Apple MPS) or in Colab (CUDA).

Train Mon-Wed, validate Thu, Fri stays untouched for the detective. Atoms for the spike-aware
heads are fit on the training days only. Writes the best-validation checkpoint and a JSON log.

  uv run --group model python scripts/train.py --out checkpoints/full.pt
  # ablations (step 5):
  uv run --group model python scripts/train.py --no-atoms --out checkpoints/no_atoms.pt
  uv run --group model python scripts/train.py --no-book  --out checkpoints/no_book.pt
  uv run --group model python scripts/train.py --ctx 32   --out checkpoints/ctx32.pt
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from lobgen.model import EventData, ModelConfig, fit_atoms, train

STEM = "{t}_{d}_34200000_57600000_10.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lob-dir", default="data/lobster")
    ap.add_argument("--tickers", nargs="+", default=["GOOG", "INTC"])
    ap.add_argument("--train-dates", nargs="+", default=["2025-12-08", "2025-12-09", "2025-12-10"])
    ap.add_argument("--val-dates", nargs="+", default=["2025-12-11"])
    ap.add_argument("--out", default="checkpoints/full.pt")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-mix", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-atoms", action="store_true", help="ablation: pure continuous heads")
    ap.add_argument("--no-book", action="store_true", help="ablation: hide the book from the input")
    ap.add_argument("--atom-share", type=float, default=0.005)
    ap.add_argument("--max-atoms", type=int, default=48)
    ap.add_argument("--no-regime", action="store_true",
                    help="ablation: no day-so-far input features (v1 inputs)")
    ap.add_argument("--flow", action="store_true",
                    help="v3: also the order-flow memory inputs (trade-sign averages over trades)")
    ap.add_argument("--market", action="store_true",
                    help="v4: also clock-time own-mid and ETF (QQQ, SMH) inputs; implies --flow")
    ap.add_argument("--market-dir", default="data/market")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disk-features", action="store_true",
                    help="keep the input features in memory-mapped temp files, not RAM (same values;"
                         " v4's ~9 GB of inputs do not fit a 12.7 GB Colab runtime otherwise)")
    a = ap.parse_args()

    lob = Path(a.lob_dir)
    paths = lambda dates: [lob / STEM.format(t=t, d=d) for t in a.tickers for d in dates]  # noqa: E731
    mdir = a.market_dir if a.market else None
    print(f"loading + featurizing {len(a.tickers)} tickers x {len(a.train_dates) + len(a.val_dates)} days"
          " (a few minutes, no output until done)", flush=True)
    tmp = tempfile.TemporaryDirectory(prefix="lobgen_feat_") if a.disk_features else None
    fdir = tmp.name if tmp else None
    tr = EventData.from_npz(paths(a.train_dates), a.ctx, market_dir=mdir, out_dir=fdir)
    va = EventData.from_npz(paths(a.val_dates), a.ctx, market_dir=mdir, out_dir=fdir)
    acts = np.concatenate([p[3][::10] for p in tr.parts])  # 10% is plenty to find >=0.5% values
    atoms = fit_atoms(acts, a.atom_share, a.max_atoms)
    print("atoms:", json.dumps(atoms))
    cfg = ModelConfig(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, ctx=a.ctx,
                      dropout=a.dropout, n_mix=a.n_mix, use_atoms=not a.no_atoms,
                      use_book=not a.no_book, atom_share=a.atom_share, max_atoms=a.max_atoms,
                      atoms=atoms, encoding=tr.encoding, regime=not a.no_regime,
                      flow=a.flow or a.market, market=a.market)
    if va.encoding != tr.encoding:
        raise ValueError("train and validation days use different action encodings")
    res = train(cfg, tr, va, a.out, steps=a.steps, batch=a.batch, lr=a.lr,
                eval_every=a.eval_every, eval_batches=a.eval_batches, device=a.device, seed=a.seed)
    log = Path(a.out).with_suffix(".json")
    log.write_text(json.dumps({"args": vars(a), "atoms": atoms, **res}, indent=1) + "\n")
    print("best val nats/event", res["best_val_event"], "->", a.out)
    if tmp:
        del tr, va  # drop the memory maps before removing their files
        tmp.cleanup()


if __name__ == "__main__":
    main()
