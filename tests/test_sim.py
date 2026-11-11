import numpy as np
import pytest

import itch_builder as b
from lobgen.book import BUY, SELL, EMPTY_ASK
from lobgen.lobster import convert, preopen_orders
from lobgen.sim import (CANCEL, DELETE, EXEC, HIDDEN, SUBMIT, TICK, SimBook, SimError,
                        Simulator, encode, states_at, verify_decode)
from lobgen.validate import check_book_transitions, check_messages

NS = 1_000_000_000
OPEN = 34_200 * NS
G = 5


def _day(tmp_path, msgs):
    p = tmp_path / "f.itch"
    p.write_bytes(b"".join([b.S(0, "O"), b.R(G, "GOOG")] + msgs))
    return p


def _busy_day(tmp_path):
    t = OPEN
    return _day(tmp_path, [
        b.A(G, OPEN - 9, 1, "B", 100, 1_000_000),      # pre-open queue at 100.00: refs 1, 2, 3
        b.A(G, OPEN - 8, 2, "B", 50, 1_000_000),
        b.A(G, OPEN - 7, 3, "B", 30, 1_000_000),
        b.U(G, OPEN - 6, 1, 4, 100, 1_000_000),         # replace: ref 4 goes to the back
        b.A(G, OPEN - 5, 5, "S", 70, 1_000_300),
        b.A(G, t + 1, 6, "S", 40, 1_000_200),           # submit inside spread (ask side)
        b.F(G, t + 2, 7, "B", 20, 999_800),             # submit 2 ticks deep
        b.E(G, t + 3, 2, 50),                           # exec: front of bid queue is ref 2
        b.X(G, t + 3, 4, 10),                           # partial cancel of newest bid
        b.U(G, t + 4, 3, 8, 30, 1_000_100),             # delete+submit, improves bid
        b.E(G, t + 5, 6, 15),                           # exec ask
        b.P(G, t + 6, 7, 1_000_150),                    # hidden at the midpoint (half tick)
        b.D(G, t + 7, 6),
        b.A(G, t + 8, 9, "B", 1, 1),                    # $0.0001 stub bid: off-grid
        b.D(G, t + 9, 9),
        b.C(G, t + 10, 8, 30, 1_000_000),               # exec at limit price
    ])


def test_preopen_orders_keeps_time_priority(tmp_path):
    init = preopen_orders(_busy_day(tmp_path), "GOOG")
    assert init.tolist() == [[2, BUY, 1_000_000, 50], [3, BUY, 1_000_000, 30],
                             [4, BUY, 1_000_000, 100], [5, SELL, 1_000_300, 70]]


def test_encode_decode_round_trip_exact(tmp_path):
    path = _busy_day(tmp_path)
    msgs, book, _ = convert(path, "GOOG", "2025-12-12", levels=3)
    init = preopen_orders(path, "GOOG")
    actions, inexact = encode(msgs, init, t0=OPEN, levels=3)
    assert inexact.tolist() == [False] * 9 + [True, True] + [False]  # stub bid + its delete
    res = verify_decode(msgs, book, init, actions, inexact, levels=3)
    assert res["message_mismatches"] == res["id_mismatches"] == res["book_mismatches"] == 0
    a = {i: actions[i].tolist() for i in range(len(actions))}
    # (dt, type, side, price field, qpos, size)
    assert a[0] == [1, SUBMIT, SELL, -1, 0, 40]          # 1 tick inside the 100.03 ask
    assert a[1] == [1, SUBMIT, BUY, 2, 0, 20]            # 2 ticks below the 100.00 bid
    assert a[2] == [1, EXEC, BUY, 0, 0, 0]               # level 0, front; 0 = whole order (50)
    # Live bids now, oldest first: 3, 4 (at 100.00), 7 (99.98). Cancels/deletes use pointers.
    assert a[3] == [0, CANCEL, BUY, -2, 0, 10]           # ref 4: second newest bid
    assert a[4][1:] == [DELETE, BUY, -3, 0, 30]          # ref 3: third newest bid
    assert a[7] == [1, HIDDEN, BUY, -1, 0, 7]            # 100.015 = bid 100.01 + half tick
    # pointer=0 is the v1 encoding: level index + queue position from the back.
    legacy, _ = encode(msgs, init, t0=OPEN, levels=3, pointer=0, whole_fill=False)
    assert legacy[2].tolist() == [1, EXEC, BUY, 0, 0, 50]
    assert legacy[3].tolist() == [0, CANCEL, BUY, 0, 0, 10]   # level 0, newest order (back)
    assert legacy[4].tolist()[1:] == [DELETE, BUY, 0, 1, 30]  # ref 3 is second from the back
    assert verify_decode(msgs, book, init, legacy, inexact, levels=3)["book_mismatches"] == 0
    p1, _ = encode(msgs, init, t0=OPEN, levels=3, pointer=2)  # rank 2 is past the limit
    assert p1[3].tolist()[3] == -2 and p1[4].tolist()[3:5] == [0, 1]


def test_states_at_matches_full_replay(tmp_path):
    path = _busy_day(tmp_path)
    msgs, book, _ = convert(path, "GOOG", "2025-12-12", levels=3)
    init = preopen_orders(path, "GOOG")
    for s, state in zip([0, 4, 9], states_at(msgs, init, [0, 4, 9])):
        bk = SimBook.from_orders(state)
        if s:
            assert bk.snapshot(3) == book[s - 1].tolist()
        rest, rest_inexact = encode(msgs[s:], state, t0=int(msgs[s - 1, 0]) if s else OPEN,
                                    levels=3)
        assert verify_decode(msgs[s:], book[s:], state, rest, rest_inexact, 3)["book_mismatches"] == 0


def _book(*orders):
    return SimBook.from_orders(np.array(orders, dtype=np.int64))


def test_pointer_actions_and_age_order():
    # Orders arrive across levels: arrival order != price order.
    bk = _book((1, BUY, 1_000_000, 100), (2, BUY, 999_900, 50), (3, BUY, 1_000_000, 30),
               (4, SELL, 1_000_200, 10))
    assert [bk.recency_rank(r) for r in (1, 2, 3)] == [2, 1, 0]
    assert bk.to_orders()[:, 0].tolist() == [1, 2, 3, 4]       # arrival order, not by price
    sim = Simulator(bk, OPEN, levels=2)
    msg, _ = sim.step(0, DELETE, BUY, -2, 0, 0)                  # second newest bid = ref 2
    assert msg[1:] == (DELETE, 2, 50, 999_900, BUY) and sim.last_action[3:5] == (-2, 0)
    msg, _ = sim.step(0, CANCEL, BUY, -9, 4, 10)                 # rank 8 of 2: clipped to oldest
    assert msg[1:] == (CANCEL, 1, 10, 1_000_000, BUY) and sim.repairs["rank"] == 1
    assert sim.last_action[3:5] == (-2, 0)
    msg, _ = sim.step(0, EXEC, BUY, -1, 0, 5)                    # executions never use pointers
    assert sim.repairs["level"] == 1 and msg[2] == 1             # level 0 front = ref 1
    rebuilt = SimBook.from_orders(sim.book.to_orders())
    assert [rebuilt.recency_rank(r) for r in (1, 3)] == [1, 0]


def test_repairs_and_strict():
    bk = _book((1, BUY, 1_000_000, 100), (2, BUY, 1_000_000, 50), (3, SELL, 1_000_200, 10))
    sim = Simulator(bk, OPEN, levels=2)
    msg, _ = sim.step(5, SUBMIT, BUY, -5, 0, 100)              # would cross the 100.02 ask
    assert msg[4] == 1_000_100 and sim.repairs["cross"] == 1
    msg, _ = sim.step(0, DELETE, BUY, 7, 9, 0)                  # level 7, qpos 9: clipped
    assert msg[1:] == (DELETE, 1, 100, 1_000_000, BUY)          # deepest level, oldest order
    assert sim.repairs["level"] == 1 and sim.repairs["qpos"] == 1
    msg, _ = sim.step(0, EXEC, BUY, 0, 0, 0)                    # 0 = the whole front order:
    assert msg[3:5] == (100, 1_000_100)                         # the 100 bid placed above
    assert sim.last_action[5] == 0 and not sim.repairs["size"]
    msg, _ = sim.step(0, EXEC, SELL, 0, 0, 999)                 # size clipped to the order
    assert msg[3] == 10 and sim.repairs["size"] == 1 and sim.last_action[5] == 0
    assert sim.step(0, CANCEL, SELL, 0, 0, 1) is None           # ask side now empty
    assert sim.skipped == 1
    with pytest.raises(SimError, match="cross"):
        Simulator(_book((1, BUY, 1_000_000, 1), (2, SELL, 1_000_100, 1)), OPEN,
                  strict=True).step(0, SUBMIT, SELL, -1, 0, 1)


def test_empty_side_reference_price():
    sim = Simulator(_book((1, BUY, 1_000_000, 100)), OPEN, levels=1)
    msg, snap = sim.step(1, SUBMIT, SELL, 0, 0, 5)              # no asks: 1 tick above bid
    assert msg[4] == 1_000_100 and snap == [1_000_100, 5, 1_000_000, 100]
    sim.step(0, DELETE, SELL, 0, 0, 0)
    sim.step(0, DELETE, BUY, 0, 0, 0)
    msg, _ = sim.step(1, SUBMIT, BUY, 0, 0, 1)                  # both empty: last bid seen
    assert msg[4] == 1_000_000


def test_random_actions_always_give_a_valid_book():
    rng = np.random.default_rng(0)
    bk = _book(*[(i + 1, BUY if i % 2 else SELL,
                  1_000_000 + (1 if i % 2 == 0 else -1) * TICK * (1 + i // 2), 100)
                 for i in range(20)])
    init = bk.to_orders()
    sim = Simulator(bk, OPEN, levels=10)
    n = 5000
    acts = np.column_stack([
        rng.integers(-5, 10**6, n), rng.integers(1, 6, n), rng.choice([BUY, SELL], n),
        rng.integers(-8, 12, n), rng.integers(-2, 30, n), rng.integers(-3, 500, n)])
    msgs, book, applied = sim.run(acts)
    assert len(msgs) == n - sim.skipped and sum(sim.repairs.values()) > 0
    assert check_book_transitions(msgs, book, 10)["violations"] == 0
    assert not any(check_messages(msgs).values())
    two = (book[:, 1] > 0) & (book[:, 3] > 0)
    assert (book[two, 0] > book[two, 2]).all()                  # never crossed or locked
    assert (book[~(book[:, 1] > 0), 0] == EMPTY_ASK).all()
    assert set(np.unique(msgs[:, 1])) <= {1, 2, 3, 4, 5}
    # Applied (post-repair) actions are valid and deterministic: replaying them reproduces the
    # output without a single repair.
    again = Simulator(SimBook.from_orders(init), OPEN, levels=10, strict=True)
    m2, b2, a2 = again.run(applied)
    np.testing.assert_array_equal(m2, msgs)
    np.testing.assert_array_equal(b2, book)
    np.testing.assert_array_equal(a2, applied)
    # A cancel can be written two ways (pointer or level + queue); encode() picks one and its
    # actions decode to exactly the same rows.
    enc, inexact = encode(msgs, init, t0=OPEN)
    assert not inexact.any() and (enc[:, 3][np.isin(enc[:, 1], (2, 3))] < 0).any()
    res = verify_decode(msgs, book, init, enc, inexact, levels=10)
    assert res["message_mismatches"] == res["id_mismatches"] == res["book_mismatches"] == 0
