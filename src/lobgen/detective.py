"""The detective: a classifier that tells real from generated runs, at each run length.

For a run length L, every sequence (real continuation or generated one) is cut to its first
L events and summarised by ~70 features computed from the LOBSTER rows alone (messages + book
snapshots), so any generator's output can be judged, including files written by other code:

  event mix   share of each type x side
  timing      share of zero gaps, quantiles of log inter-arrival time, burstiness, duration
  sizes       log size by event type, share of round lots / exactly 100 shares
  prices      submit / cancel / execution price relative to the book just before the event
  book        spread, touch depth, imbalance, total depth, book span, mid-price path
  sequence    replace pairs (delete then submit, same side, same ns), side / type persistence,
              execution sweeps
  lifecycle   cancels and executions of orders submitted inside the run, time to delete

All features are relative (ticks from the book, logs of sizes and gaps): no absolute price,
no order id value. The detective sees only the continuation, never the conditioning context;
otherwise "real data from another day" would be caught by the jump at the window start and
could not serve as a noise floor.

Scoring: gradient-boosted trees, 5-fold cross-validation grouped by window (a window's real
and generated runs are always in the same fold), out-of-fold AUC and accuracy, with 95%
bootstrap intervals over windows. AUC 0.5 = indistinguishable, 1.0 = always caught.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

TICK = 100
Seq = tuple[np.ndarray, np.ndarray]  # (messages [L, 6], book [L, 4*levels])


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def _std(x) -> float:
    return float(np.std(x)) if len(x) > 1 else np.nan


def _q(x, qs) -> list[float]:
    return np.quantile(x, qs).tolist() if len(x) else [np.nan] * len(qs)


def _corr(a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def features(m: np.ndarray, b: np.ndarray, tick: int = TICK) -> dict[str, float]:
    t, typ, ref, size, price, d = (m[:, k] for k in range(6))
    f: dict[str, float] = {}
    for k in range(1, 6):
        for s, sname in ((1, "b"), (-1, "s")):
            f[f"share_t{k}{sname}"] = _mean((typ == k) & (d == s))

    dt = np.diff(t)
    ldt = np.log10(dt + 1.0)
    pos = ldt[dt > 0]
    f["dt_zero"] = _mean(dt == 0)
    for q, v in zip((10, 25, 50, 75, 90), _q(pos, [0.1, 0.25, 0.5, 0.75, 0.9])):
        f[f"ldt_q{q}"] = v
    f["ldt_mean"], f["ldt_std"] = _mean(ldt), _std(ldt)
    f["ldt_ac1"] = _corr(ldt[:-1], ldt[1:])
    f["log_duration"] = float(np.log10(t[-1] - t[0] + 1.0))

    sub, can, dele, exe, hid = (typ == k for k in (1, 2, 3, 4, 5))
    lsize = np.log10(np.maximum(size, 1))
    for name, mask in (("sub", sub), ("can", can), ("del", dele), ("exe", exe), ("hid", hid)):
        f[f"{name}_lsize"] = _mean(lsize[mask])
    f["sub_lsize_std"] = _std(lsize[sub])
    f["sub_size100"] = _mean(size[sub] == 100)
    f["sub_round"] = _mean(size[sub] % 100 == 0)
    f["exe_round"] = _mean(size[exe] % 100 == 0)

    # Prices relative to the own-side best just before the event (+ = deeper in the book).
    own = np.where(d[1:] > 0, b[:-1, 2], b[:-1, 0])
    own_ok = np.where(d[1:] > 0, b[:-1, 3], b[:-1, 1]) > 0
    rel = d[1:] * (own - price[1:]) / tick
    s1, c1, e1, h1 = (x[1:] & own_ok for x in (sub, can | dele, exe, hid))
    rs = rel[s1]
    f["sub_inside"], f["sub_at"] = _mean(rs < 0), _mean(rs == 0)
    f["sub_1"], f["sub_2_5"], f["sub_deep"] = _mean(rs == 1), _mean((rs >= 2) & (rs <= 5)), _mean(rs > 5)
    f["sub_rel"] = _mean(np.clip(rs, -5, 20))
    f["can_at"], f["can_rel"] = _mean(rel[c1] == 0), _mean(np.clip(rel[c1], 0, 20))
    f["exe_at"] = _mean(rel[e1] == 0)
    f["hid_rel"] = _mean(np.clip(rel[h1], -10, 10))

    ask, askv, bid, bidv = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    two = (askv > 0) & (bidv > 0)
    spread = (ask[two] - bid[two]) / tick
    f["spread_mean"], f["spread_std"] = _mean(spread), _std(spread)
    f["spread_min"] = float(spread.min()) if len(spread) else np.nan
    f["spread_max"] = float(spread.max()) if len(spread) else np.nan
    f["spread_1"], f["spread_2"] = _mean(spread == 1), _mean(spread == 2)
    f["spread_changes"] = _mean(np.diff(spread) != 0)
    la, lb = np.log1p(askv), np.log1p(bidv)
    f["l1_ask_mean"], f["l1_ask_std"] = _mean(la), _std(la)
    f["l1_bid_mean"], f["l1_bid_std"] = _mean(lb), _std(lb)
    tot = askv + bidv
    imb = ((bidv - askv) / np.where(tot > 0, tot, 1))[tot > 0]
    f["imb_mean"], f["imb_std"], f["imb_abs"] = _mean(imb), _std(imb), _mean(np.abs(imb))
    av, bv = b[:, 1::4], b[:, 3::4]
    f["depth_ask"], f["depth_bid"] = _mean(np.log1p(av.sum(1))), _mean(np.log1p(bv.sum(1)))
    f["levels_ask"], f["levels_bid"] = _mean((av > 0).sum(1)), _mean((bv > 0).sum(1))
    last = b.shape[1] // 4 - 1
    full_a, full_b = av[:, last] > 0, bv[:, last] > 0
    f["span_ask"] = _mean((b[full_a, 4 * last] - ask[full_a]) / tick)
    f["span_bid"] = _mean((bid[full_b] - b[full_b, 4 * last + 2]) / tick)
    mid = (ask[two] + bid[two]) / 2 / tick
    dmid = np.diff(mid)
    f["mid_changes"] = _mean(dmid != 0)
    f["mid_move"] = float(abs(mid[-1] - mid[0])) if len(mid) else np.nan
    f["mid_path"] = float(np.abs(dmid).sum()) if len(dmid) else np.nan
    f["mid_rv"] = float(np.sqrt((dmid ** 2).sum())) if len(dmid) else np.nan

    f["replace_pairs"] = _mean(dele[:-1] & sub[1:] & (dt == 0) & (d[:-1] == d[1:]))
    f["same_side"] = _mean(d[1:] == d[:-1])
    f["same_type"] = _mean(typ[1:] == typ[:-1])
    f["exec_sweep"] = _mean(exe[:-1] & exe[1:] & (dt == 0))

    sub_time = dict(zip(ref[sub].tolist(), t[sub].tolist()))
    cd = can | dele
    young_cd = np.array([r in sub_time for r in ref[cd].tolist()], dtype=bool)
    f["young_cancel"] = _mean(young_cd)
    f["young_exec"] = _mean(np.array([r in sub_time for r in ref[exe].tolist()], dtype=bool))
    ttd = [tt - sub_time[r] for r, tt in zip(ref[dele].tolist(), t[dele].tolist()) if r in sub_time]
    lttd = np.log10(np.array(ttd, dtype=float) + 1.0)
    f["ttd_q25"], f["ttd_q50"], f["ttd_q75"] = _q(lttd, [0.25, 0.5, 0.75])
    gone = set(ref[dele].tolist())
    f["sub_deleted"] = _mean(np.array([r in gone for r in ref[sub].tolist()], dtype=bool))
    return f


def feature_matrix(seqs: list[Seq], length: int) -> tuple[np.ndarray, list[str]]:
    rows = []
    for m, b in seqs:
        if len(m) < length:
            raise ValueError(f"sequence of {len(m)} events is shorter than run length {length}")
        rows.append(features(m[:length], b[:length]))
    names = list(rows[0])
    return np.array([[r[k] for k in names] for r in rows], dtype=np.float64), names


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          random_state=seed)


def _boot_ci(y, p, groups, n_boot: int, rng) -> tuple[list[float], list[float]]:
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ug))]
    aucs, accs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(len(ug), size=len(ug))])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        accs.append(np.mean((p[idx] > 0.5) == y[idx]))
    return (np.quantile(aucs, [0.025, 0.975]).tolist(), np.quantile(accs, [0.025, 0.975]).tolist())


def detect(real: list[Seq], fake: list[Seq], lengths, real_groups=None, fake_groups=None,
           folds: int = 5, seed: int = 0, n_boot: int = 500, importance: bool = True) -> dict:
    """Out-of-fold real-vs-fake AUC / accuracy at each run length.

    real[i] and fake[i] belong to window i unless groups are given; a group never spans folds.
    """
    real_groups = np.arange(len(real)) if real_groups is None else np.asarray(real_groups)
    fake_groups = np.arange(len(fake)) if fake_groups is None else np.asarray(fake_groups)
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))].astype(int)
    groups = np.r_[real_groups, fake_groups]
    rng = np.random.default_rng(seed)
    out = {}
    lengths = sorted(lengths)
    for L in lengths:
        Xr, names = feature_matrix(real, L)
        Xf, _ = feature_matrix(fake, L)
        X = np.vstack([Xr, Xf])
        p = np.empty(len(y))
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr, te in cv.split(X, y, groups):
                p[te] = _classifier(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        auc_ci, acc_ci = _boot_ci(y, p, groups, n_boot, rng)
        res = {"auc": float(roc_auc_score(y, p)), "auc_ci": auc_ci,
               "acc": float(np.mean((p > 0.5) == y)), "acc_ci": acc_ci,
               "n_real": len(real), "n_fake": len(fake)}
        if importance and L in (lengths[0], lengths[-1]):
            res["top_features"] = _importance(X, y, groups, names, seed)
        out[L] = res
    return out


def _importance(X, y, groups, names, seed, top: int = 10) -> list[tuple[str, float]]:
    """Permutation importance (drop in held-out AUC) on one 80/20 grouped split."""
    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                  .split(X, y, groups))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = _classifier(seed).fit(X[tr], y[tr])
        r = permutation_importance(clf, X[te], y[te], scoring="roc_auc", n_repeats=5,
                                   random_state=seed)
    order = np.argsort(-r.importances_mean)[:top]
    return [(names[k], round(float(r.importances_mean[k]), 4)) for k in order]
