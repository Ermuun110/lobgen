import numpy as np
import pytest

from lobgen.market import CLOCK_TAUS_S, ETFS, N_ETF_FEAT, NS, VOL_TAUS_S, Market, Series


def synthetic_market(t0=34_000 * NS, t1=58_000 * NS, n=4000, seed=0, date="2025-12-12") -> Market:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(len(ETFS)):
        t = np.sort(rng.integers(t0, t1, n))
        x = np.log(500.0 + k * 100) + np.cumsum(rng.normal(0, 2e-4, n))
        out.append(Series(t, x))
    return Market(out, date)


def _ema_exact(t, x, q, tau):
    """EMA at q of the step function x (value x[k] on [t[k], t[k+1])), started at x[0]."""
    E, last = x[0], t[0]
    k = 0
    while k + 1 < len(t) and t[k + 1] <= q:
        a = np.exp(-(t[k + 1] - last) / tau)
        E = a * E + (1 - a) * x[k]
        last = t[k + 1]
        k += 1
    a = np.exp(-(q - last) / tau)
    return a * E + (1 - a) * x[k], x[k]


def test_series_features_match_the_definition():
    rng = np.random.default_rng(1)
    t = np.sort(rng.integers(0, 600 * NS, 300)).astype(np.int64)
    x = np.log(100) + np.cumsum(rng.normal(0, 3e-4, 300))
    s = Series(t, x)
    q = np.sort(rng.integers(t[0], t[-1] + 5 * NS, 50))
    f = s.features(q)
    assert f.shape == (50, N_ETF_FEAT)
    for i, qi in enumerate(q):
        for j, tau in enumerate(CLOCK_TAUS_S):
            E, xv = _ema_exact(t, x, qi, tau * NS)
            np.testing.assert_allclose(f[i, j], np.clip((xv - E) * 1e3, -5, 5), atol=1e-6)
        for j, tau in enumerate(VOL_TAUS_S):
            k = np.searchsorted(t, qi, side="right") - 1
            jumps = np.abs(np.diff(x[:k + 1])) * 1e4
            R = np.sum(jumps * np.exp(-(qi - t[1:k + 1]) / (tau * NS)))
            np.testing.assert_allclose(f[i, len(CLOCK_TAUS_S) + j], np.log1p(R) / 3, atol=1e-6)
    assert (s.features(np.array([t[0] - 1])) == 0).all()      # before the first quote


def test_scenario_and_frozen():
    m = synthetic_market()
    s = m.series[0]
    t0 = 40_000 * NS
    sc = m.scenario(t0, -100, ramp_s=30, etfs=("QQQ",))
    q = np.array([t0 - NS, t0 + 15 * NS, t0 + 60 * NS, t0 + 3600 * NS])
    d = (sc.series[0].value(q) - s.value(q)) * 1e4
    np.testing.assert_allclose(d, [0, -50, -100, -100], atol=0.5)
    np.testing.assert_allclose(sc.series[1].value(q), m.series[1].value(q))   # SMH untouched
    fr = m.frozen(t0)
    np.testing.assert_allclose(fr.series[0].value(q[1:]), s.value(np.array([t0]))[0])
    # a frozen market forgets its moves: deviations decay to 0
    f = fr.features(np.array([t0 + 3600 * NS]))
    np.testing.assert_allclose(f[0, :len(CLOCK_TAUS_S)], 0, atol=1e-6)
    assert m.ret_bps(t0, t0 + 60 * NS) == pytest.approx(
        (s.value(np.array([t0 + 60 * NS]))[0] - s.value(np.array([t0]))[0]) * 1e4)


def test_frozen_and_shifted_equal_a_rebuilt_series():
    """The no-copy scenario paths give the features of an explicitly rebuilt series."""
    rng = np.random.default_rng(3)
    t = np.sort(rng.integers(0, 300 * NS, 400)).astype(np.int64)
    x = np.log(100) + np.cumsum(rng.normal(0, 3e-4, 400))
    s = Series(t, x)
    t0 = int(t[200]) + 7
    q = np.sort(rng.integers(t[0], t[-1] + 100 * NS, 200))
    keep = t <= t0
    np.testing.assert_allclose(s.frozen(t0).features(q), Series(t[keep], x[keep]).features(q),
                               atol=1e-9)
    step = 10_000_000
    sh = s.shifted(t0, -40, 2.0, step)
    tr = t0 + np.arange(201) * step
    t_all = np.union1d(t, tr)
    j = np.clip((t_all - t0) // step, 0, 200)            # the overlay is a 10 ms staircase
    x_all = s.value(t_all) + (t_all >= t0) * j / 200 * (-40e-4)
    rebuilt = Series(t_all, x_all)
    np.testing.assert_allclose(sh.value(q), rebuilt.value(q), atol=1e-12)
    f1, f2 = sh.features(q), rebuilt.features(q)
    np.testing.assert_allclose(f1, f2, atol=1e-6)


def test_from_bbo_keeps_the_last_quote_per_timestamp():
    # a replace of the best bid: delete (bid drops to 99) then add (back to 100), same ns
    s = {"t": np.array([1, 2, 2, 3, 4]), "bid": np.array([100, 99, 100, 100, -1]),
         "ask": np.array([102, 102, 102, 102, 102])}
    ser = Series.from_bbo(s)
    np.testing.assert_array_equal(ser.t, [1])        # the 99 bid at t=2 is transient
    np.testing.assert_allclose(np.exp(ser.x), [101])
