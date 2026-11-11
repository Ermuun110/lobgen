"""Write LOB-Bench input folders from windows + generated runs, and run its scoring.

LOB-Bench (third_party/lob_bench) expects, per stock:
  data_cond/{T}_{date}_message_real_id_{i}.csv            conditioning context (cond_len msgs)
  data_real/{T}_{date}_message_real_id_{i}.csv            the true continuation (gen_len msgs)
  data_gen/{T}_{date}_message_real_id_{i}_gen_id_{k}.csv  a model's continuation
plus matching *_orderbook_* files (10 levels). Its public sample uses 128 sequences of 500+500.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .data import Day, Generated, Window
from .lobster import write_lobster_csv

LOB_BENCH_DIR = Path(__file__).resolve().parents[2] / "third_party" / "lob_bench"


def write_bench_dirs(out_dir: str | Path, day: Day, windows: list[Window],
                     generated: list[Generated], gen_len: int = 500) -> Path:
    out = Path(out_dir)
    for sub in ("data_real", "data_cond", "data_gen"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    stem = f"{day.ticker}_{day.date}"
    for w, g in zip(windows, generated):
        for sub, m, b, suffix in (("data_cond", w.cond_msgs, w.cond_book, ""),
                                  ("data_real", w.real_msgs[:gen_len], w.real_book[:gen_len], ""),
                                  ("data_gen", g.msgs[:gen_len], g.book[:gen_len], "_gen_id_0")):
            write_lobster_csv(m, b,
                              out / sub / f"{stem}_message_real_id_{w.i}{suffix}.csv",
                              out / sub / f"{stem}_orderbook_real_id_{w.i}{suffix}.csv")
    return out


def _import_lob_bench():
    if str(LOB_BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(LOB_BENCH_DIR))
    import data_loading, metrics, run_bench, scoring  # noqa: E401
    return data_loading, metrics, run_bench, scoring


def score_dirs(bench_dir: str | Path, conditional: bool = False) -> dict:
    """Run LOB-Bench's default unconditional (or conditional) scores on one folder.

    Returns {score_name: {metric: value}} plus '_summary' with LOB-Bench's own aggregate
    (mean / median / IQM across scores, 99% bootstrap CI).
    """
    data_loading, metrics, run_bench, scoring = _import_lob_bench()
    d = Path(bench_dir)
    loader = data_loading.Simple_Loader(real_data_path=str(d / "data_real"),
                                        gen_data_path=str(d / "data_gen"),
                                        cond_data_path=str(d / "data_cond"))
    for s in loader:
        s.materialize()
    cfg = run_bench.DEFAULT_SCORING_CONFIG_COND if conditional else run_bench.DEFAULT_SCORING_CONFIG
    scores, _dfs, _plots = scoring.run_benchmark(loader, cfg, default_metric=run_bench.DEFAULT_METRICS)
    out = {name: {m: float(v[0]) for m, v in per.items()} for name, per in scores.items()}
    summ = scoring.summary_stats(scores)
    out["_summary"] = {
        m: {stat: (float(val), [float(x) for x in ci])
            for stat, (val, ci) in zip(("mean", "median", "iqm"), triple)}
        for m, triple in summ.items()
    }
    return out
