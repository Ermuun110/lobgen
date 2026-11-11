import numpy as np
import pytest

from lobgen.book import BUY, SELL
from lobgen.data import Day, make_windows
from lobgen.shock import aggressor_groups, all_orders, inject, measure, ref_point, summarize
from lobgen.sim import EXEC, SUBMIT, TICK, SimBook, Simulator
from lobgen.validate import check_book_transitions

NS = 1_000_000_000


def _toy_day(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    init = np.array([(i + 1, BUY if i % 2 else SELL,
                      1_000_000 + (1 if i % 2 == 0 else -1) * TICK * (1 + i // 2), 100)
                     for i in range(40)], dtype=np.int64)
    sim = Simulator(SimBook.from_orders(init), 35_000 * NS)
    acts = np.column_stack([rng.integers(0, 10**6, n), rng.choice([1, 1, 3, 4], n),
                            rng.choice([BUY, SELL], n), rng.integers(0, 4, n),
                            rng.integers(0, 3, n), rng.choice([100, 200, 7], n)])
    msgs, book, applied = sim.run(acts)
    return Day("TOY", "2025-12-12", msgs, book, applied, init, encoding=(16, True))


def test_aggressor_groups():
    m = np.array([
        [1, SUBMIT, 1, 100, 10, 1],
        [2, EXEC, 2, 50, 11, -1],    # one buy order hitting three asks
        [2, EXEC, 3, 50, 11, -1],
        [2, EXEC, 4, 10, 12, -1],
        [2, EXEC, 5, 10, 9, 1],      # same ns, other side: a new order
        [3, EXEC, 6, 10, 9, 1],      # later: a new order
    ])
    first, last = aggressor_groups(m)
    assert first.tolist() == [1, 4, 5] and last.tolist() == [3, 4, 5]


def test_inject_executes_in_price_time_order_and_keeps_a_valid_book():
    d = _toy_day()
    w = make_windows(d, 3, cond_len=64, gen_len=40, seed=0, margin_s=0)[1]
    ask_before = SimBook.from_orders(w.state)
    total = sum(lv.total for lv in ask_before.side[SELL].values())
    size = min(250, total - 1)
    nw, ref = inject(w, +1, size)
    k = len(nw.cond_msgs) - len(w.cond_msgs)
    new = nw.cond_msgs[-k:]
    assert (new[:, 1] == EXEC).all() and (new[:, 5] == SELL).all()
    assert new[:, 3].sum() == size == ref["size"]
    assert (np.diff(new[:, 4]) >= 0).all()                 # asks consumed from the best up
    assert len(set(new[:, 0])) == 1 and new[0, 0] == ref["t"]
    np.testing.assert_array_equal(nw.cond_book[-1], SimBook.from_orders(nw.state).snapshot(10))
    rep = check_book_transitions(nw.cond_msgs, nw.cond_book, 10)
    assert rep["violations"] == 0
    # the order is the first thing after the original context and the book moved against it
    assert ref["mid"] == pytest.approx((w.cond_book[-1][0] + w.cond_book[-1][2]) / (2 * TICK))
    assert nw.cond_book[-1][0] >= w.cond_book[-1][0]


def test_measure_on_a_known_path():
    t0 = 10 * NS
    ctx = np.array([1_000_000 + TICK, 100, 1_000_000 - TICK, 100] + [0] * 36)   # mid 10000
    rows, msgs = [], []
    for i, (ask, bid) in enumerate([(2, -1), (2, 0), (3, 1)]):        # in ticks from 1_000_000
        rows.append([1_000_000 + ask * TICK, 50, 1_000_000 + bid * TICK, 70] + [0] * 36)
        msgs.append([t0 + (i + 1) * 1_000_000, EXEC if i == 0 else SUBMIT, 9, 30, 0, SELL])
    ref = ref_point(ctx, t0, +1, 500)
    out = measure(np.array(msgs), np.array(rows), ctx, ref, events=(1, 2, 3, 4), times=(500_000, 2_000_000, 10**9))
    np.testing.assert_allclose(out["move_ev"][:3], [0.5, 1.0, 2.0])
    assert np.isnan(out["move_ev"][3])
    np.testing.assert_allclose(out["spread_ev"][:3], [3, 2, 2])
    np.testing.assert_allclose(out["flow_ev"][:3], [30, 30, 30])      # one buyer-initiated trade
    np.testing.assert_allclose(out["move_t"][:2], [0.0, 1.0])
    assert np.isnan(out["move_t"][2])                                  # 1 s is not covered
    np.testing.assert_allclose(out["ms_ev"][:3], [1, 2, 3])


def test_all_orders_signs_and_sizes():
    d = _toy_day()
    orders = all_orders(d)
    assert orders
    for o in orders[:50]:
        rows = d.msgs[o.first:o.last + 1]
        assert (rows[:, 1] == EXEC).all() and rows[:, 3].sum() == o.size
        assert o.sign == -rows[0, 5]


def test_summarize_drops_poorly_covered_horizons():
    x = np.array([[1.0, np.nan], [3.0, 2.0], [2.0, np.nan]])
    s = summarize(x, n_boot=50)
    assert s["mean"][0] == pytest.approx(2.0) and s["mean"][1] is None
