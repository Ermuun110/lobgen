"""Generate notebooks/train_colab.ipynb with the lobgen package and scripts embedded verbatim.

The notebook rebuilds the LOBSTER days from the filtered ITCH streams already in Drive
(MyDrive/lobgen/filtered, written by fetch_itch_colab.ipynb), so nothing large is uploaded
from this machine. Re-run this script after changing any embedded file.

Run from repo root:  uv run python scripts/make_train_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ["__init__", "itch", "book", "lobster", "validate", "market", "sim", "features",
           "data", "baselines", "detective", "plots", "model", "shock", "cli"]
SCRIPTS = ["build_actions", "train", "run_detective", "shock_test", "track_test"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(True)}


files = {f"src/lobgen/{m}.py": (ROOT / f"src/lobgen/{m}.py").read_text() for m in PACKAGE}
files |= {f"scripts/{s}.py": (ROOT / f"scripts/{s}.py").read_text() for s in SCRIPTS}

cells = [
    md("""
# Train the order-flow transformer (runs in Colab)

**First run `fetch_itch_colab.ipynb`** (CPU runtime): it adds the market ETF quotes
(`MyDrive/lobgen/market/`) that `v4` reads.

1. Builds the five LOBSTER days for GOOG and INTC from `MyDrive/lobgen/filtered/` (or reuses them
   from Drive) and checks
   that the simulator reproduces every row exactly (same check as on the laptop).
2. A 300-step smoke run checks that GPU training gives finite losses.
3. Trains `v2` (v1's size, with the v2 action encoding and day-so-far features), `full`
   (full size), `v3` (v2 plus order-flow memory inputs) and `v4` (v3 plus the market: the
   stock's own clock-time moves and the QQQ / SMH quotes, the channel news arrives through),
   and optionally the ablations (no day-so-far features, no atoms, no book, short context).
   Checkpoints go to `MyDrive/lobgen/checkpoints/`; finished runs are skipped.
4. Runs the detective on the test day for the baselines and every model (one ticker at a time), saving
   `MyDrive/lobgen/results/detective.json` after each run.
5. Shock test for v3 and v4 (reaction to large market orders; `shock.json`) and tracking test
   (given the market's real path, does the generated stock follow the real one? plus a news
   scenario; `track.json`). v2 and the baselines are already done on the laptop.

**Run:** Runtime -> Change runtime type -> GPU (A100 or L4 if available, T4 works). Then
Runtime -> Run all and approve the Drive prompt. **Resumable:** if the session drops, run all
again: the prepared data is reused from Drive, an interrupted training run continues from its last
evaluation (at most 500 steps lost), and finished runs and detective entries are skipped. To stop
early: interrupt the cell, then Runtime -> Disconnect and delete runtime.
"""),
    code("""
DRIVE = "/content/drive/MyDrive/lobgen"
REPO = "/content/lobgen"
DATES = ["2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12"]
TICKERS = ["GOOG", "INTC"]
STEPS = 20_000
RUNS = {  # name -> extra train.py flags
    "v2": "--d-model 192 --layers 4 --heads 6 --ctx 128",   # v1's size: new encoding + day-so-far
    "full": "",                                             # 5.3M params, ctx 256
    # v4 first: it is the headline model. v3 (= v4 without the market inputs) is the ablation that
    # says whether a v4 gain comes from the market/news channel or from the order-flow memory.
    "v4": "--d-model 192 --layers 4 --heads 6 --ctx 128 --market",  # v3 + market (news channel)
    "v3": "--d-model 192 --layers 4 --heads 6 --ctx 128 --flow",  # v2 + order-flow memory
}
SHOCK_MODELS = ["v4", "v3"]   # shock tests (v2 and baselines: done on the laptop)
TRACK_MODELS = ["v4"]         # tracking test: only models that read the market (v3 cannot track
                              # it, and v4 with the market frozen is already the no-news control)
ETFS = ["QQQ", "SMH"]
RUN_SEED_CHECK = False  # True: v4 once more with another seed (+~35 min training + detective). The
                        # detective CIs cover evaluation noise only; this measures training noise, so
                        # a v4-vs-v3 gap can be told apart from seed luck. Trained last.
if RUN_SEED_CHECK:
    RUNS |= {"v4_seed1": RUNS["v4"] + " --seed 1"}
RUN_ABLATIONS = False   # True: also the full-size ablations below (~1 h each on an A100)
if RUN_ABLATIONS:
    RUNS |= {"no_regime": "--no-regime", "no_atoms": "--no-atoms", "no_book": "--no-book",
             "ctx32": "--ctx 32"}
"""),
    code("""
from google.colab import drive
drive.mount("/content/drive", timeout_ms=600_000)   # approve the popup: tick "Select all"
import os, subprocess, torch
for d in (f"{DRIVE}/checkpoints", f"{DRIVE}/results", f"{REPO}/src/lobgen", f"{REPO}/scripts"):
    os.makedirs(d, exist_ok=True)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch runtime to GPU")
!pip -q install sortedcontainers numba
"""),
]
for path, src in files.items():
    cells.append(code(f"%%writefile {{REPO}}/{path}\n".replace("{REPO}", "/content/lobgen") + src))
cells += [
    code("""
# Prepared data. Reused from Drive (~1 min) when it was built by exactly this code; otherwise
# rebuilt from the filtered ITCH files (~20-30 min), checked, and saved to Drive for next time.
import hashlib, shutil
os.makedirs(f"{REPO}/data/lobster", exist_ok=True)
missing = [d for d in DATES if not os.path.exists(f"{DRIVE}/filtered/{d}.itch.gz")]
assert not missing, f"not in {DRIVE}/filtered: {missing} (run fetch_itch_colab.ipynb first)"
if not os.path.exists(f"{REPO}/data/filtered"):
    os.symlink(f"{DRIVE}/filtered", f"{REPO}/data/filtered")
os.chdir(REPO)
os.environ["PYTHONPATH"] = f"{REPO}/src"
BUILD_SRC = ["itch", "book", "lobster", "validate", "sim", "cli"]   # everything the data depends on
key = hashlib.sha256(b"".join(open(f, "rb").read() for f in
      [f"src/lobgen/{m}.py" for m in BUILD_SRC] + ["scripts/build_actions.py"])).hexdigest()[:12]
CACHE = f"{DRIVE}/prepared_{key}"
if os.path.exists(f"{CACHE}/DONE"):
    shutil.copytree(CACHE, "data/lobster", dirs_exist_ok=True)
    shutil.copy(f"{CACHE}/build_actions.log", "/content/build_actions.log")
    print("prepared data copied from", CACHE)
else:
    for d in DATES:
        for t in TICKERS:
            out = f"data/lobster/{t}_{d}_34200000_57600000_10.npz"
            if not os.path.exists(out):
                subprocess.run(["python", "-m", "lobgen.cli", "convert", f"data/filtered/{d}.itch.gz",
                                "--ticker", t, "--date", d], check=True, stdout=subprocess.DEVNULL)
            print("converted", t, d, flush=True)
    # Encode every day into simulator actions and verify the exact round trip.
    !cd {REPO} && python -u scripts/build_actions.py --workers 1 2>&1 | tee /content/build_actions.log | cut -c1-160
    assert "ALL EXACT" in open("/content/build_actions.log").read(), "round trip failed: stop here"
    for old in os.listdir(DRIVE):            # data built by older code is no longer needed
        if old.startswith("prepared_"):
            shutil.rmtree(f"{DRIVE}/{old}")
    shutil.copytree("data/lobster", CACHE, dirs_exist_ok=True)
    shutil.copy("/content/build_actions.log", CACHE)
    open(f"{CACHE}/DONE", "w").write("ok")
    print("prepared data saved to", CACHE)
assert "ALL EXACT" in open("/content/build_actions.log").read()
"""),
    code("""
# Market quotes (from fetch_itch_colab.ipynb): small, copied next to the data.
missing = [f"{e}_{d}.npz" for d in DATES for e in ETFS if not os.path.exists(f"{DRIVE}/market/{e}_{d}.npz")]
assert not missing, f"not in {DRIVE}/market: {missing} - run fetch_itch_colab.ipynb first"
shutil.copytree(f"{DRIVE}/market", f"{REPO}/data/market", dirs_exist_ok=True)
print("market quotes:", sorted(os.listdir(f"{REPO}/data/market")))
"""),
    code("""
# Smoke test (~3 min, one INTC day): 300 steps of a small model on the GPU; losses must be finite.
import json, math
!cd {REPO} && python scripts/train.py --tickers INTC --train-dates 2025-12-10 --val-dates 2025-12-11 --steps 300 --eval-every 100 --eval-batches 5 --d-model 64 --layers 2 --heads 2 --ctx 64 --market --disk-features --out /content/smoke.pt
hist = json.load(open("/content/smoke.json"))["history"]
print(hist[-1])
assert all(math.isfinite(v) for h in hist for k, v in h.items() if k != "step"), "NaN/inf loss: stop"
assert hist[-1]["val_event"] < hist[0]["val_event"], "validation loss did not improve"
"""),
    code("""
# Training. A run is skipped if its final checkpoint (<name>_last.pt) is already in Drive, and
# continues from <name>_state.pt (saved every 500 steps) if it was interrupted.
# <name>.pt = best validation step, <name>_last.pt = final weights, <name>.json = log.
# Loading + featurizing the days takes a few minutes before the first step: that is not a hang.
# --disk-features: the inputs (~8 GB for v3/v4) go to memory-mapped files on the runtime's disk,
# not RAM (identical values); in RAM they need ~14 GB and the 12.7 GB runtime kills the run
# (it shows only "^C").
# If a run fails or is interrupted, this cell raises, so "Run all" stops here instead of going on.
for name, flags in RUNS.items():
    if os.path.exists(f"{DRIVE}/checkpoints/{name}_last.pt"):
        print("done already:", name)
        continue
    print("training", name, flags, flush=True)
    !cd {REPO} && python -u scripts/train.py --steps {STEPS} {flags} --disk-features --out {DRIVE}/checkpoints/{name}.pt
    assert os.path.exists(f"{DRIVE}/checkpoints/{name}_last.pt"), \\
        f"{name} did not finish (error or interrupt above). Re-run this cell to resume it."
"""),
    code("""
# Detective on the test day (Fri), merged into MyDrive/lobgen/results/detective.json after every
# run. The floors (same-day, other days) do not depend on the action encoding and are already
# computed on the laptop; replay and the baselines do, so they are re-run here.
import shutil
os.makedirs(f"{REPO}/results", exist_ok=True)
if os.path.exists(f"{DRIVE}/results/detective.json"):
    shutil.copy(f"{DRIVE}/results/detective.json", f"{REPO}/results/detective.json")

def done(name, t):
    p = f"{REPO}/results/detective.json"
    return os.path.exists(p) and name in json.load(open(p)).get(t, {})

def save():   # to Drive after every (generator, ticker): an interrupted session loses one at most
    for f in os.listdir(f"{REPO}/results"):
        shutil.copy(f"{REPO}/results/{f}", f"{DRIVE}/results/")

def step(ck):
    return torch.load(ck, map_location="cpu", weights_only=False)["step"]

for name in ["replay", "history", "pooled", "markov"]:
    for t in TICKERS:
        if not done(name, t):
            !cd {REPO} && python scripts/run_detective.py --tickers {t} --only {name}
            save()
            assert done(name, t), f"detective failed for {name} {t}: see the output above"
for name in list(RUNS) + ["full_last"]:
    ck = f"{DRIVE}/checkpoints/{name}.pt"
    assert os.path.exists(ck), f"missing {ck}: run the training cell first"
    best = ck.replace("_last.pt", ".pt")
    if name.endswith("_last") and os.path.exists(best) and step(best) == step(ck):
        print(name, "skipped: the best checkpoint is the final step, same weights")
        continue
    for t in TICKERS:
        if not done(name, t):
            !cd {REPO} && python scripts/run_detective.py --tickers {t} --only {name} --model {ck} --model-name {name}
            save()
            assert done(name, t), f"detective failed for {name} {t}: see the output above"
print("detective done")
"""),
    code("""
# Shock test (~5 min per model and ticker on a GPU) and tracking test (~15 min for v4 on GOOG).
for f in ("shock.json", "track.json"):
    if os.path.exists(f"{DRIVE}/results/{f}"):
        shutil.copy(f"{DRIVE}/results/{f}", f"{REPO}/results/{f}")

def has(f, name):
    p = f"{REPO}/results/{f}"
    return os.path.exists(p) and all(name in json.load(open(p)).get(t, {}) for t in TICKERS)

for name in SHOCK_MODELS:
    assert os.path.exists(f"{DRIVE}/checkpoints/{name}.pt"), f"missing {name}.pt: train it first"
for name in SHOCK_MODELS:
    ck = f"{DRIVE}/checkpoints/{name}.pt"
    if not has("shock.json", name):
        !cd {REPO} && python -u scripts/shock_test.py --baselines --model {name}={ck}
        save()
        assert has("shock.json", name), f"shock test for {name} failed: see the output above"
for name in TRACK_MODELS:
    ck = f"{DRIVE}/checkpoints/{name}.pt"
    if not has("track.json", name):
        !cd {REPO} && python -u scripts/track_test.py --model {name}={ck}
        save()
        assert has("track.json", name), f"tracking test for {name} failed: see the output above"
print("done: download MyDrive/lobgen/results/ and MyDrive/lobgen/checkpoints/ (v3*, v4*)")
"""),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}, "colab": {"provenance": []},
                 "accelerator": "GPU"},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks/train_colab.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(out)
