"""External checks on filtered ITCH days (things internal invariants cannot catch).

1. Closing-cross price from our stream == official close (catches wrong file/date mapping,
   wrong symbol/locate, wrong price scaling, truncated streams).
2. Hidden-execution ('P') side flag is always 'B' (confirms type-5 direction is unusable).

Run: uv run python scripts/check_external.py
"""

import csv
import sys
from pathlib import Path

from lobgen.itch import decode, iter_messages, load_stream

ref = {(r["date"], r["ticker"]): float(r["close"])
       for r in csv.DictReader(l for l in open("checks/official_closes.csv") if not l.startswith("#"))}
bad = 0
for date in sorted({d for d, _ in ref}):
    path = Path(f"data/filtered/{date}.itch.gz")
    loc2sym, close, sides = {}, {}, {}
    for raw in iter_messages(load_stream(path)):
        t, loc, ts, f = decode(raw)
        if t == "R":
            loc2sym[loc] = f[0].decode().strip()
        elif t == "Q" and f[4] == b"C":
            close[loc2sym[loc]] = f[2] / 1e4
        elif t == "P":
            sides.setdefault(loc2sym[loc], set()).add(f[1].decode())
    for (d, tkr), official in ref.items():
        if d != date:
            continue
        ok = abs(close.get(tkr, -1) - official) < 1e-9
        bad += not ok
        print(f"{date} {tkr}: closing cross {close.get(tkr)}  official {official}  {'OK' if ok else 'MISMATCH'}"
              f"  | hidden side flags {sorted(sides.get(tkr, []))}")
print("all match" if not bad else f"{bad} mismatches")
sys.exit(1 if bad else 0)
