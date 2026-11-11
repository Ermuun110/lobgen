"""lobgen command line.

  lobgen filter  RAW.gz --tickers GOOG INTC --out data/filtered/2025-12-12.itch
  lobgen convert data/filtered/2025-12-12.itch --ticker GOOG --date 2025-12-12
  lobgen validate data/lobster/GOOG_2025-12-12_34200000_57600000_10.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .itch import filter_file
from .lobster import convert, write_lobster
from .validate import check_book_transitions, check_messages, summary


def _dump(obj, path: Path | None = None) -> None:
    text = json.dumps(obj, indent=2, default=str)
    print(text)
    if path:
        path.write_text(text + "\n")


def cmd_filter(a) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = filter_file(a.raw, a.tickers, out, allow_truncated=a.allow_truncated)
    _dump(rep, out.with_suffix(".filter.json"))


def cmd_convert(a) -> None:
    msgs, book, rep = convert(a.filtered, a.ticker, a.date, a.levels, a.start, a.end)
    paths = write_lobster(msgs, book, a.out_dir, a.ticker, a.date, a.levels, a.start, a.end,
                          csv=a.csv)
    d = rep.as_dict()
    d["events"] = len(msgs)
    d["files"] = [str(p) for p in paths]
    _dump(d, paths[0].with_suffix(".convert.json"))


def cmd_validate(a) -> None:
    z = np.load(a.npz)
    msgs, book, levels = z["messages"], z["book"], int(z["levels"])
    res = {
        "messages": check_messages(msgs),
        "book_transitions": check_book_transitions(msgs, book, levels),
        "summary": summary(msgs, book),
    }
    _dump(res, Path(a.npz).with_suffix(".validate.json"))
    ok = (res["book_transitions"]["violations"] == 0
          and not any(res["messages"].values()))
    raise SystemExit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(prog="lobgen")
    sub = p.add_subparsers(required=True)

    f = sub.add_parser("filter", help="stream a gzipped ITCH day, keep chosen tickers")
    f.add_argument("raw")
    f.add_argument("--tickers", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--allow-truncated", action="store_true",
                   help="stop cleanly at EOF of a partial download (testing only)")
    f.set_defaults(fn=cmd_filter)

    c = sub.add_parser("convert", help="filtered ITCH -> LOBSTER csv + npz")
    c.add_argument("filtered")
    c.add_argument("--ticker", required=True)
    c.add_argument("--date", required=True, help="YYYY-MM-DD")
    c.add_argument("--levels", type=int, default=10)
    c.add_argument("--start", type=int, default=34_200, help="window start, s after midnight")
    c.add_argument("--end", type=int, default=57_600, help="window end, s after midnight")
    c.add_argument("--out-dir", default="data/lobster")
    c.add_argument("--csv", action="store_true",
                   help="also write full-day LOBSTER csv pair (~2.4 GB per liquid stock-day)")
    c.set_defaults(fn=cmd_convert)

    v = sub.add_parser("validate", help="invariant checks + summary stats")
    v.add_argument("npz")
    v.set_defaults(fn=cmd_validate)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
