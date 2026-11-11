"""Stylized facts of order flow: the checks a trader runs before trusting a simulator.

Each fact is computed from LOBSTER rows alone (messages + book snapshots), pooled over many
runs (windows) of equal length, so real continuations, other days and any generator are
measured the same way. Times are in events (or in trades, for trade-indexed facts); prices in
ticks.

  spread          distribution of the bid-ask spread (share of events at 1, 2, ..., 5+ ticks)
  returns         mid change over h events: std and excess kurtosis, h = 1, 10, 50, 200
  vol_clustering  autocorrelation of |mid change| per event, lags 1..100
  sign_memory     autocorrelation of trade signs (aggressor side of visible executions), lags
                  1..50 in trade time; real markets show slow, power-law-like decay
  impact          response function R(l) = E[eps_n (m_{n+l} - m_n)], ticks, l = 1..50 trades:
                  how far the mid moves in the direction of a trade (Bouchaud et al.)
  gaps            inter-arrival times: share of 0 ns and quantiles of log10 ns for the rest
  sizes           submit sizes: share of exactly 100 shares, of round lots, log10 quantiles
  mix             share of each event type
  depth           mean visible volume at levels 1..10, each side, relative to level 1
  lifetime        orders placed and deleted inside the run: log10 ns from submit to delete,
                  and the share of submits deleted inside the run

Hidden executions (type 5) are left out of trade signs: their ITCH side flag is always 'B'.
"""

from __future__ import annotations

import numpy as np

from .sim import TICK

Seq = tuple[np.ndarray, np.ndarray]
RET_H = (1, 10, 50, 200)
VOL_LAGS = np.arange(1, 101)
SIGN_LAGS = np.arange(1, 51)
IMPACT_LAGS = np.arange(1, 51)


def _mid(b: np.ndarray) -> np.ndarray:
    """Mid in ticks after each event, forward-filled over one-sided books."""
    two = (b[:, 1] > 0) & (b[:, 3] > 0)
    mid = np.where(two, (b[:, 0] + b[:, 2]) / (2 * TICK), np.nan)
    idx = np.where(~np.isnan(mid), np.arange(len(mid)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = mid[idx]
    first = np.flatnonzero(~np.isnan(mid))
    return np.where(np.isnan(out), mid[first[0]] if len(first) else 0.0, out)


def _acf(series: list[np.ndarray], lags: np.ndarray) -> np.ndarray:
    """Autocorrelation pooled over runs: products within each run, one global mean/variance."""
    allx = np.concatenate(series) if series else np.zeros(0)
    if len(allx) < 2 or allx.var() == 0:
        return np.full(len(lags), np.nan)
    mu, var = allx.mean(), allx.var()
    out = np.empty(len(lags))
    for k, lag in enumerate(lags):
        num = den = 0.0
        for x in series:
            if len(x) > lag:
                num += np.sum((x[:-lag] - mu) * (x[lag:] - mu))
                den += len(x) - lag
        out[k] = num / den / var if den else np.nan
    return out


def _kurt(x: np.ndarray) -> float:
    x = x - x.mean()
    v = np.mean(x ** 2)
    return float(np.mean(x ** 4) / v ** 2 - 3) if v > 0 else np.nan


def stylized_facts(seqs: list[Seq]) -> dict:
    mids = [_mid(b) for _, b in seqs]
    out: dict = {}

    spreads = np.concatenate([((b[:, 0] - b[:, 2]) / TICK)[(b[:, 1] > 0) & (b[:, 3] > 0)]
                              for _, b in seqs])
    out["spread"] = [float(np.mean(spreads == k)) for k in range(1, 5)] + [float(np.mean(spreads >= 5))]

    out["ret_std"], out["ret_kurt"] = [], []
    for h in RET_H:
        r = np.concatenate([m[h:] - m[:-h] for m in mids if len(m) > h])
        out["ret_std"].append(float(r.std()))
        out["ret_kurt"].append(_kurt(r))

    out["vol_acf"] = _acf([np.abs(np.diff(m)) for m in mids], VOL_LAGS).tolist()

    signs, resp_num, resp_den = [], np.zeros(len(IMPACT_LAGS)), np.zeros(len(IMPACT_LAGS))
    for (m, b), mid in zip(seqs, mids):
        tr = np.flatnonzero(m[:, 1] == 4)
        if len(tr) < 2:
            continue
        eps = -m[tr, 5].astype(float)
        signs.append(eps)
        before = np.where(tr > 0, mid[np.maximum(tr - 1, 0)], mid[0])   # mid just before trade n
        for k, lag in enumerate(IMPACT_LAGS):
            if len(tr) > lag:
                resp_num[k] += np.sum(eps[:-lag] * (before[lag:] - before[:-lag]))
                resp_den[k] += len(tr) - lag
    out["sign_acf"] = _acf(signs, SIGN_LAGS).tolist()
    out["impact"] = np.where(resp_den > 0, resp_num / np.maximum(resp_den, 1), np.nan).tolist()

    dts = np.concatenate([np.diff(m[:, 0]) for m, _ in seqs])
    pos = np.log10(dts[dts > 0])
    out["dt_zero"] = float(np.mean(dts == 0))
    out["ldt_q"] = np.quantile(pos, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if len(pos) else []

    allm = np.concatenate([m for m, _ in seqs])
    sub = allm[allm[:, 1] == 1, 3]
    out["size100"] = float(np.mean(sub == 100)) if len(sub) else np.nan
    out["size_round"] = float(np.mean(sub % 100 == 0)) if len(sub) else np.nan
    out["lsize_q"] = np.quantile(np.log10(sub), [0.1, 0.5, 0.9]).tolist() if len(sub) else []
    out["mix"] = [float(np.mean(allm[:, 1] == k)) for k in range(1, 6)]

    allb = np.concatenate([b for _, b in seqs]).astype(float)
    av, bv = allb[:, 1::4][:, :10], allb[:, 3::4][:, :10]
    out["depth_ask"] = (av.mean(0) / max(av[:, 0].mean(), 1e-9)).tolist()
    out["depth_bid"] = (bv.mean(0) / max(bv[:, 0].mean(), 1e-9)).tolist()

    life, n_sub, n_del = [], 0, 0
    for m, _ in seqs:
        born = {}
        for t, typ, ref, *_ in m.tolist():
            if typ == 1:
                born[ref] = t
                n_sub += 1
            elif typ == 3 and ref in born:
                life.append(t - born.pop(ref))
                n_del += 1
    lg = np.log10(np.array(life, dtype=float) + 1)
    out["life_q"] = np.quantile(lg, [0.25, 0.5, 0.75]).tolist() if len(lg) else []
    out["deleted_in_run"] = n_del / n_sub if n_sub else np.nan
    return out


# Scalar discrepancies (0 = identical) used for the summary table. Each compares one fact of a
# generator with the same fact of the real continuations of the same windows.
def discrepancies(real: dict, fake: dict) -> dict[str, float]:
    def l1(a, b):
        return float(np.nansum(np.abs(np.asarray(a, float) - np.asarray(b, float))))

    def rmse(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.sqrt(np.nanmean((a - b) ** 2)))

    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.nanmean(np.abs(np.log(np.maximum(b, 1e-12) / np.maximum(a, 1e-12)))))

    return {
        "spread (L1)": l1(real["spread"], fake["spread"]),
        "return std (log ratio)": rel(real["ret_std"], fake["ret_std"]),
        "vol clustering (RMSE)": rmse(real["vol_acf"], fake["vol_acf"]),
        "sign memory (RMSE)": rmse(real["sign_acf"], fake["sign_acf"]),
        "impact (RMSE, ticks)": rmse(real["impact"], fake["impact"]),
        "event mix (L1)": l1(real["mix"], fake["mix"]),
        "100-lot share (abs)": abs(real["size100"] - fake["size100"]),
        "gaps (L1 of quantiles)": l1(real["ldt_q"] + [real["dt_zero"]], fake["ldt_q"] + [fake["dt_zero"]]),
        "depth profile (L1)": l1(real["depth_ask"] + real["depth_bid"], fake["depth_ask"] + fake["depth_bid"]),
    }
