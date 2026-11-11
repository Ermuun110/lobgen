"""LOB-Bench noise floor: score REAL data from another day as if it were generated.

No model can be expected to beat real data from a different day, so these numbers are the
floor every later model is read against. Test day = Fri 2025-12-12; "generated" windows are
taken from Mon and Thu at the same time of day.

Run: uv run --group bench python scripts/noise_floor.py
"""

import json
from pathlib import Path

from lobgen.baselines import RealOtherDay
from lobgen.bench import score_dirs, write_bench_dirs
from lobgen.data import Day, make_windows

LOB = Path("data/lobster")
BENCH = Path("data/bench")
OUT = Path("results")
TEST, OTHERS = "2025-12-12", ["2025-12-08", "2025-12-11"]
STEM = "{t}_{d}_34200000_57600000_10.npz"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results = {}
    for t in ("GOOG", "INTC"):
        real = Day.load(LOB / STEM.format(t=t, d=TEST))
        windows = make_windows(real, 128, cond_len=500, gen_len=500, seed=0)
        for d in OTHERS:
            name = f"{t}/real_{d}"
            other = Day.load(LOB / STEM.format(t=t, d=d), actions=False)
            gen = RealOtherDay(other).generate(windows, 500)
            bdir = write_bench_dirs(BENCH / name, real, windows, gen, 500)
            results[name] = score_dirs(bdir)
            s = results[name]["_summary"]
            print(f"{name}: L1 mean {s['l1']['mean'][0]:.3f}  Wasserstein mean {s['wasserstein']['mean'][0]:.3f}")
    (OUT / "noise_floor.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
