"""Charts for results/. Needs the `bench` dependency group (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed colour per entity (validated categorical order, light mode); floors are neutral grey.
COLORS = {"markov": "#eb6834", "pooled": "#1baf7a", "history": "#eda100"}
# Trained models: the named runs keep one colour across every chart; any other model name takes
# the remaining slots in name order.
MODEL_COLORS = ["#2a78d6", "#6250d6", "#e87ba4", "#008300", "#e34948"]
NAMED_MODELS = {"full": MODEL_COLORS[0], "v1": MODEL_COLORS[1], "v2": MODEL_COLORS[2]}


def model_colors(names) -> dict:
    """Colour per trained-model name, stable across charts (colour follows the model)."""
    free = [c for c in MODEL_COLORS if c not in NAMED_MODELS.values()]
    others = sorted(n for n in names if n not in NAMED_MODELS)
    return {n: NAMED_MODELS[n] for n in sorted(names) if n in NAMED_MODELS} | dict(zip(others, free))
REFERENCES = {"same_day", "replay"}
OTHER_DAY_DASHES = [":", "-."]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_detective(res: dict, ticker: str, path: str | Path, metric: str = "auc") -> Path:
    """Detective AUC vs run length: one line per generator, grey floors, 95% CI bands.

    res: {generator name: {run length (int or str): {"auc": .., "auc_ci": [lo, hi], ...}}}
    Generators whose name starts with "real_" are drawn as the other-day floor; "replay"
    (a sanity check that is 0.5 by construction) is left off.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150)
    lows = [0.45]

    def series(name):
        r = res[name]
        L = np.array(sorted(int(k) for k in r))
        get = lambda k: r[str(k)] if str(k) in r else r[k]  # noqa: E731
        y = np.array([get(k)[metric] for k in L])
        ci = np.array([get(k)[f"{metric}_ci"] for k in L])
        return L, y, ci

    others = sorted(n for n in res if n.startswith("real_"))
    for name in (["same_day"] if "same_day" in res else []) + others:
        L, y, ci = series(name)
        if name == "same_day":
            style, label = "--", "real, neighbouring window (floor)"
        else:
            style, label = OTHER_DAY_DASHES[others.index(name) % 2], f"real, other day ({name[5:]})"
        ax.fill_between(L, ci[:, 0], ci[:, 1], color="#8a8984", alpha=0.10, lw=0)
        ax.plot(L, y, style, color="#6f6e69", lw=1.6, label=label)
        lows.append(float(ci[:, 0].min()))

    models = [n for n in res if n not in COLORS and n not in REFERENCES and not n.startswith("real_")]
    colors = COLORS | model_colors(models)
    for name, color in colors.items():
        if name not in res:
            continue
        L, y, ci = series(name)
        ax.fill_between(L, ci[:, 0], ci[:, 1], color=color, alpha=0.15, lw=0)
        ax.plot(L, y, "-", color=color, lw=2, marker="o", ms=4, label=name)
        lows.append(float(ci[:, 0].min()))

    ax.set_xscale("log")
    ax.set_ylim(min(lows) - 0.02, 1.02)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_xlabel("run length (events the detective sees)", color=MUTED)
    ax.set_ylabel("detective AUC  (0.5 = can't tell, 1 = always caught)", color=MUTED)
    ax.set_title(f"{ticker}: how fast generated order flow gets caught", color=TEXT, loc="left",
                 fontsize=11)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_facts(facts: dict, ticker: str, path: str | Path) -> Path:
    """Six stylized facts, real continuations vs references vs generators (small multiples).

    facts: {name: stylized_facts(...) dict}; "real" is the reference, "real_<date>" another day.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .facts import IMPACT_LAGS, RET_H, SIGN_LAGS, VOL_LAGS

    names = [n for n in facts if n != "real"]
    others = sorted(n for n in names if n.startswith("real_"))
    models = [n for n in names if n not in COLORS and not n.startswith("real_")]
    colors = COLORS | model_colors(models)

    def style(name):
        if name == "real":
            return dict(color=TEXT, lw=2.4, ls="-", label="real (same windows)", zorder=5)
        if name.startswith("real_"):
            return dict(color="#6f6e69", lw=1.6, ls=OTHER_DAY_DASHES[others.index(name) % 2],
                        label=f"real, other day ({name[5:]})")
        return dict(color=colors[name], lw=1.8, ls="-", label=name)

    order = ["real"] + others + [n for n in colors if n in facts]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), dpi=150)
    panels = [
        ("spread (share of events)", np.arange(1, 6), "spread", "spread, ticks (5 = 5+)", False),
        ("mid-price move std", np.array(RET_H), "ret_std", "horizon, events", True),
        ("volatility clustering: acf of |mid move|", VOL_LAGS, "vol_acf", "lag, events", False),
        ("trade-sign memory: acf of trade signs", SIGN_LAGS, "sign_acf", "lag, trades", False),
        ("price impact R(l), ticks", IMPACT_LAGS, "impact", "lag l, trades", False),
        ("depth profile (level 1 = 1)", np.arange(1, 11), "depth", "level", False),
    ]
    for ax, (title, x, key, xlabel, logy) in zip(axes.flat, panels):
        for name in order:
            f = facts[name]
            y = (np.add(f["depth_ask"], f["depth_bid"]) / 2) if key == "depth" else np.asarray(f[key])
            st = style(name)
            ax.plot(x[:len(y)], y, marker="o" if len(x) <= 10 else None, ms=3.5, **st)
        if key in ("vol_acf", "sign_acf", "impact", "ret_std"):
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if key in ("vol_acf", "sign_acf"):
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, color=TEXT, loc="left", fontsize=9.5)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7.5)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{ticker}: stylized facts of 1000-event runs, real vs generated", color=TEXT,
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shock(res: dict, ticker: str, path: str | Path, impact_ms: float | None = None) -> Path:
    """Shock test (scripts/shock_test.py): the reaction to a large market order, real vs
    generated. res: results/shock.json[ticker]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = res["_meta"]
    ev, ms = np.array(meta["events"]), np.array(meta["times_ms"])
    gens = [n for n in res if not n.startswith("_") and n not in ("real", "real_by_size")]
    models = [n for n in gens if n not in COLORS]
    colors = COLORS | model_colors(models)
    arr = lambda v: np.array([np.nan if x is None else x for x in v], dtype=float)  # noqa: E731

    def line(ax, x, s, color, label, real=False) -> bool:
        y, lo, hi = arr(s["mean"]), arr(s["lo"]), arr(s["hi"])
        ax.fill_between(x, lo, hi, color=color, alpha=0.18 if real else 0.12, lw=0)
        ax.plot(x, y, "-", color=color, lw=2.4 if real else 1.8, marker="o", ms=3.5,
                label=label, zorder=5 if real else 3)
        return bool(np.isfinite(y).any())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150)
    panels = [
        ("mid move after the order, ticks (+ = order's direction)", "move_t", ms, "time after the order, ms"),
        ("mid move, ticks, by events after the order", "move_ev", ev, "events after the order"),
        ("bid-ask spread, ticks", "spread_ev", ev, "events after the order"),
        ("follow-on aggressive volume in the order's direction, shares", "flow_ev", ev,
         "events after the order"),
        ("time to the k-th event, ms", "ms_ev", ev, "k (events after the order)"),
    ]
    for ax, (title, key, x, xlabel) in zip(axes.flat, panels):
        drawn = line(ax, x, res["real"][key], TEXT, "real (what followed)", real=True)
        for n in gens:
            if key in res[n]:
                drawn |= line(ax, x, res[n][key], colors.get(n, MUTED), n)
        if not drawn:   # e.g. runs too short to reach any clock horizon
            ax.text(0.5, 0.5, "no horizon covered", transform=ax.transAxes, ha="center",
                    color=MUTED, fontsize=8)
            continue
        ax.set_xscale("log")
        if key == "ms_ev":   # symlog: a generator whose events all share one ns still plots
            ax.set_yscale("symlog", linthresh=0.01)
        if key in ("move_t", "move_ev", "flow_ev"):
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, color=TEXT, loc="left", fontsize=9)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)

    # Impact vs order size: real orders binned by size vs injected orders (net of no-order runs).
    ax = axes.flat[5]
    rb = res["real_by_size"]
    if impact_ms is None:
        cover = [k for k, c in enumerate(res["real"]["move_t"]["cover"]) if c >= 0.95]
        j = min(cover[-1] if cover else 2, 3)
    else:
        j = int(np.argmin(np.abs(ms - impact_ms)))
    xs, ys, los, his = [], [], [], []
    for b in rb["bins"]:
        if "mean" in b and b["mean"][j] is not None:
            xs.append(b["median_size"])
            ys.append(b["mean"][j])
            los.append(b["lo"][j])
            his.append(b["hi"][j])
    if xs:
        ax.errorbar(xs, ys, yerr=[np.subtract(ys, los), np.subtract(his, ys)], color=TEXT, fmt="o-",
                    lw=2.2, ms=5, capsize=3, label="real orders of that size", zorder=5)
    for n in gens:
        inj = res[n].get("inject")
        if not inj:
            continue
        xs = [b["size"] for b in inj["by_size"]]
        s = [b["impact_t"] for b in inj["by_size"]]
        y = [np.nan if v["mean"][j] is None else v["mean"][j] for v in s]
        lo = [np.nan if v["lo"][j] is None else v["lo"][j] for v in s]
        hi = [np.nan if v["hi"][j] is None else v["hi"][j] for v in s]
        c = colors.get(n, MUTED)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(xs, y, "s--", color=c, lw=1.6, ms=4, label=f"{n}, injected")
    if ax.has_data():
        ax.set_xscale("log")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"price impact {ms[j]:g} ms after an order, ticks", color=TEXT, loc="left", fontsize=9)
    ax.set_xlabel("order size, shares", color=MUTED, fontsize=8)

    for ax in axes.flat:
        ax.grid(True, which="major", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7.5)
    h, lab = axes.flat[0].get_legend_handles_labels()
    h2, lab2 = axes.flat[5].get_legend_handles_labels()
    fig.legend(h + h2, lab + lab2, frameon=False, fontsize=8.5, loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{ticker}: reaction to {meta['n_shocks']} large market orders "
                 f"(>= {meta['min_size']:,} shares), test day {meta['test']}",
                 color=TEXT, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_track_case(case: dict, ticker: str, path: str | Path) -> Path:
    """Tracking case study (scripts/track_test.py): the real mid path around the test day's
    sharpest market drop, against generated runs (median and 10-90% band of the runs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.array(case["grid_s"])
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    names = list(case["runs"])
    base = sorted({n.split("@")[0] for n in names})
    colors = model_colors(base)
    for n in names:
        runs = np.array(case["runs"][n], dtype=float)
        if not len(runs):
            continue
        c = colors.get(n.split("@")[0], MUTED)
        frozen = n.endswith("@frozen")
        with np.errstate(all="ignore"):
            lo, med, hi = np.nanpercentile(runs, [10, 50, 90], 0)
        ax.fill_between(g, lo, hi, color=c, alpha=0.08 if frozen else 0.16, lw=0)
        ax.plot(g, med, ":" if frozen else "-", color=c, lw=1.8,
                label=f"{n.split('@')[0]}, market frozen at start (no news)" if frozen
                else f"{n} (median, 10-90% of runs)")
    q = np.array(case["qqq"], dtype=float)
    ax.plot(g, q * case["beta_qqq"], "--", color="#8a8984", lw=1.4,
            label=f"QQQ x beta {case['beta_qqq']:.2f} (training days)")
    ax.plot(g, np.array(case["real"], dtype=float), "-", color=TEXT, lw=2.4, label=f"real {ticker}",
            zorder=5)
    ax.axhline(0, color=MUTED, lw=0.8)
    hh, mm = divmod(int(case["t0_s"]) // 60, 60)
    ax.set_title(f"{ticker}: price path from {hh:02d}:{mm:02d} ET on the test day "
                 "(the day's sharpest market drop), real vs generated", color=TEXT, loc="left",
                 fontsize=10)
    ax.set_xlabel("seconds from the start of the run", color=MUTED, fontsize=8)
    ax.set_ylabel("mid move, bps", color=MUTED, fontsize=8)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
