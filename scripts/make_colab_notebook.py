"""Generate notebooks/fetch_itch_colab.ipynb with the lobgen modules it needs embedded verbatim.

Run from repo root:  uv run python scripts/make_colab_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "itch", "book", "lobster", "market"]
FILES = {m: (ROOT / f"src/lobgen/{m}.py").read_text() for m in MODULES}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(True)}


cells = [
    md("""
# Fetch + filter Nasdaq ITCH days (runs in Colab)

Downloads full-day TotalView-ITCH 5.0 files from Nasdaq's public server and keeps only what
the project needs, in your Google Drive:
* `MyDrive/lobgen/filtered/<date>.itch.gz`: the stocks (GOOG, INTC), full order flow.
* `MyDrive/lobgen/filtered_market/<date>.itch.gz`: the market ETFs (QQQ, SMH), and from them
  `MyDrive/lobgen/market/<ETF>_<date>.npz`: best bid and ask after every change. This is how
  news reaches the model (src/lobgen/market.py).

Raw files are deleted after filtering. A day is downloaded only if something is missing for it,
so with the stock files already in Drive this run fetches only the ETFs (~15 min per day: ~10
download, ~5 filter), then builds the quote series (a few minutes per ETF-day).

**Run:** Runtime -> Run all (a CPU runtime is enough). Approve the Drive prompt. Keep the tab
open until the last cell prints `done`. Safe to re-run: finished work is skipped.
"""),
    code("""
DAYS = {  # date -> file on emi.nasdaq.com
    "2025-12-08": "S120825-v50.txt.gz",
    "2025-12-09": "S120925-v50.txt.gz",
    "2025-12-10": "S121025-v50.txt.gz",
    "2025-12-11": "S121125-v50.txt.gz",
    "2025-12-12": "S121225-v50.txt.gz",
    # "2026-06-12": "S061226-v50.txt.gz",  # optional: out-of-time test day, 17.9 GB
}
TICKERS = ["GOOG", "INTC"]
MARKET = ["QQQ", "SMH"]          # must match lobgen.market.ETFS
BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
DRIVE = "/content/drive/MyDrive/lobgen"
OUT = f"{DRIVE}/filtered"
OUT_MKT = f"{DRIVE}/filtered_market"
MKT = f"{DRIVE}/market"
"""),
    code("""
from google.colab import drive
drive.mount("/content/drive")
import os
for d in (OUT, OUT_MKT, MKT, "/content/pkg/lobgen"):
    os.makedirs(d, exist_ok=True)
!pip -q install numba sortedcontainers
"""),
]
for m, src in FILES.items():
    cells.append(code(f"%%writefile /content/pkg/lobgen/{m}.py\n" + src))
cells += [
    code("""
# Speed check: 15 s of one file. If this is under ~5 MB/s, total time is large; tell Claude.
import subprocess
r = subprocess.run(["curl", "-s", "-m", "15", "-o", "/dev/null", "-w", "%{speed_download}",
                    "-r", "0-2000000000", BASE + DAYS["2025-12-12"]], capture_output=True, text=True)
mbps = float(r.stdout or 0) / 1e6
total_gb = 51
print(f"{mbps:.1f} MB/s  ->  ~{total_gb * 1000 / max(mbps, 0.01) / 60:.0f} min for all 5 days")
"""),
    code("""
import gzip, json, shutil, subprocess, sys, time
sys.path.insert(0, "/content/pkg")
from lobgen.itch import filter_file

def content_length(url):
    h = subprocess.run(["curl", "-sI", url], capture_output=True, text=True).stdout
    return int([l for l in h.splitlines() if l.lower().startswith("content-length")][-1].split(":")[1])

def filter_to(raw, tickers, final, date, fname, size, t0, t1):
    local = f"/content/{date}.itch"
    rep = filter_file(raw, tickers, local)
    t2 = time.time()
    with open(local, "rb") as src, gzip.open(final + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.rename(final + ".part", final)
    rep.update(date=date, file=fname, tickers=tickers, compressed_bytes=size,
               download_min=(t1 - t0) / 60, filter_min=(t2 - t1) / 60)
    with open(final.replace(".itch.gz", ".filter.json"), "w") as f:
        json.dump(rep, f, indent=2)
    os.remove(local)
    print(date, tickers, f"filter {rep['filter_min']:.1f} min,",
          f"kept {rep['kept_bytes'] / 1e6:.1f} MB of {rep['raw_bytes'] / 1e9:.1f} GB", flush=True)

for date, fname in DAYS.items():
    jobs = [(t, f) for t, f in ((TICKERS, f"{OUT}/{date}.itch.gz"), (MARKET, f"{OUT_MKT}/{date}.itch.gz"))
            if not os.path.exists(f)]
    if not jobs:
        print(date, "already done"); continue
    url, raw = BASE + fname, "/content/raw.gz"
    size = content_length(url)
    t0 = time.time()
    # curl's own progress bar doesn't reach the cell output, so poll the file size.
    p = subprocess.Popen(["curl", "-sS", "-C", "-", "--retry", "20", "--retry-delay", "5",
                          "--retry-all-errors", "-o", raw, url])
    while p.poll() is None:
        time.sleep(30)
        have = os.path.getsize(raw) if os.path.exists(raw) else 0
        rate = have / max(time.time() - t0, 1) / 1e6
        print(f"  {date}: {have / 1e9:5.2f} / {size / 1e9:.2f} GB  ({rate:.0f} MB/s)", flush=True)
    if p.returncode:
        raise RuntimeError(f"{date}: curl exited {p.returncode}")
    got = os.path.getsize(raw)
    assert got == size, f"{date}: downloaded {got} of {size} bytes"
    t1 = time.time()
    print(date, f"download {(t1 - t0) / 60:.1f} min", flush=True)
    for tickers, final in jobs:
        filter_to(raw, tickers, final, date, fname, size, t0, t1)
    os.remove(raw)
print("filtering done")
"""),
    code("""
# Best bid / ask series of each ETF (what the model reads as "the market"), ~1-5 min per ETF-day.
import numpy as np
from lobgen.market import ETFS, bbo_series, save_bbo
assert tuple(MARKET) == ETFS, (MARKET, ETFS)
for date in DAYS:
    for etf in MARKET:
        out = f"{MKT}/{etf}_{date}.npz"
        if os.path.exists(out):
            print(etf, date, "already done"); continue
        t0 = time.time()
        s = bbo_series(f"{OUT_MKT}/{date}.itch.gz", etf)
        save_bbo(s, "/content/tmp.npz")
        shutil.copy("/content/tmp.npz", out)
        spread = (s["ask"] - s["bid"])[(s["ask"] > 0) & (s["bid"] > 0)] / 100
        print(etf, date, f"{len(s['t']):,} quote changes, median spread {np.median(spread):.1f} ticks,",
              f"{os.path.getsize(out) / 1e6:.1f} MB, {(time.time() - t0) / 60:.1f} min", flush=True)
print("done: download MyDrive/lobgen/market/ to data/market/ on the laptop")
"""),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}, "colab": {"provenance": []}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = ROOT / "notebooks/fetch_itch_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1) + "\n")
print(out)
