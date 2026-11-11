"""Order-book simulator: the "game engine" that turns generator actions into a valid LOBSTER stream.

A generator never writes LOBSTER rows itself. It emits *actions* expressed relative to the
current book; the simulator resolves each action against an order-level book (one FIFO queue
per price) and returns the LOBSTER message row and orderbook snapshot it produces. Whatever a
generator does, its output is a proper book: never crossed, no cancel of an order that does not
exist, level volumes that add up, one message per row.

Action = 6 int64 fields (columns of an action array, see the DT..SIZE constants):
  DT     ns since the previous event (>= 0).
  TYPE   LOBSTER type: 1 submit | 2 partial cancel | 3 delete | 4 visible execution |
         5 hidden execution. 6 cross / 7 halt pass through unchanged and are never generated.
  SIDE   1 buy | -1 sell. For 2-4 it is the resting order's side, as in LOBSTER.
  PRICE  1:   ticks from the own-side best price, + deeper (more passive), - inside the spread.
         2-4: index of an *existing* price level on the own side, 0 = best. Referring to levels
              (not tick distances) means a cancel can never point at an empty price.
         2-3: a negative value -(r+1) is instead a *pointer*: the r-th newest live order on the
              own side (r = 0 the newest), wherever it rests. encode() uses it whenever the
              target is among the PTR_RANKS newest orders. Real deletes are aimed this way (GOOG
              pulls a ladder of its own 100-share orders a few events after placing them: 44%
              of deletes hit one of the 5 newest orders), which a (level, queue position) action
              can only express indirectly through a moving book.
         5:   half-ticks from the own-side best (midpoint executions sit on half ticks).
         6/7: raw LOBSTER price (pass-through).
  QPOS   2-3: target order's place in its level's queue counted from the BACK (0 = newest);
              0 for pointer actions;
         4:   counted from the FRONT (0 = oldest), i.e. price-time priority. With these
              conventions 0 is the mode for every type: on 2025-12-12 52-90% of cancels and
              deletes hit the newest order and 93% (INTC) / 99% (GOOG) of executions the oldest.
  SIZE   shares. A delete (3) ignores it and removes the whole order. For an execution (4),
         0 means "the whole resting order" (with whole_fill, encode() writes 0 whenever the
         execution consumes the order: 78-79% of executions on 2025-12-12). A generator then
         does not need to know the size of an order it cannot see to produce a full fill.

Own-side best when that side is empty: one tick inside the opposite best; if both sides are
empty, the last own-side reference price used.

Invalid actions are repaired (and counted in Simulator.repairs) rather than rejected:
a submit that would cross is moved to one tick inside the opposite best, an out-of-range level
or queue index (or pointer rank) is clipped, a size larger than the target order is clipped. The one action that
cannot be repaired (2-4 on an empty side) returns None. strict=True raises instead; encode()
runs strict on real data, so real data never needs a repair.

encode() maps real LOBSTER rows to actions using the same rules, and decoding those actions
reproduces the rows exactly (tests + scripts/check_sim_replay.py over every converted day).
"""

from __future__ import annotations

from collections import Counter
from itertools import islice

import numpy as np
from sortedcontainers import SortedDict, SortedList

from .book import BUY, EMPTY_ASK, EMPTY_BID, SELL, BookError

TICK = 100            # $0.01 in LOBSTER price units ($ x 10000)
HALF_TICK = TICK // 2
DT, TYPE, SIDE, PRICE, QPOS, SIZE = range(6)
N_FIELDS = 6
SUBMIT, CANCEL, DELETE, EXEC, HIDDEN, CROSS, HALT = 1, 2, 3, 4, 5, 6, 7
MODELED_TYPES = (SUBMIT, CANCEL, DELETE, EXEC, HIDDEN)
GEN_REF0 = 1 << 40    # refs for generated orders; ITCH refs on these days are < 2**31
PTR_RANKS = 16        # cancels / deletes of the 16 newest orders on a side use a pointer


class SimError(RuntimeError):
    pass


class _Level:
    __slots__ = ("total", "q")

    def __init__(self) -> None:
        self.total = 0
        self.q: dict[int, int] = {}  # ref -> shares; insertion order = time priority


class SimBook:
    """Order-level book with a FIFO queue per price level, plus each side's orders by age."""

    def __init__(self) -> None:
        self.orders: dict[int, tuple[int, int]] = {}  # ref -> (side, price)
        self.side: dict[int, SortedDict] = {BUY: SortedDict(), SELL: SortedDict()}
        self.seq: dict[int, int] = {}                 # ref -> arrival number
        self.age: dict[int, SortedList] = {BUY: SortedList(), SELL: SortedList()}  # live seqs
        self._by_seq: dict[int, int] = {}             # seq -> ref
        self._next_seq = 0

    @classmethod
    def from_orders(cls, rows: np.ndarray) -> "SimBook":
        """rows: [K, 4] (ref, side, price, shares) in arrival order (oldest first), which is
        also queue order within each price."""
        b = cls()
        for ref, side, price, shares in np.asarray(rows).tolist():
            b.add(ref, side, price, shares)
        return b

    def to_orders(self) -> np.ndarray:
        """Live orders in arrival order: from_orders(to_orders()) rebuilds the same queues and
        the same pointer ranks."""
        rows = [(ref, s, p, sh) for s in (BUY, SELL) for p, lv in self.side[s].items()
                for ref, sh in lv.q.items()]
        rows.sort(key=lambda r: self.seq[r[0]])
        return np.array(rows, dtype=np.int64).reshape(-1, 4)

    def add(self, ref: int, side: int, price: int, shares: int) -> None:
        if ref in self.orders:
            raise BookError(f"duplicate order ref {ref}")
        if shares <= 0:
            raise BookError(f"non-positive add size {shares} for ref {ref}")
        levels = self.side[side]
        lv = levels.get(price)
        if lv is None:
            lv = levels[price] = _Level()
        lv.q[ref] = shares
        lv.total += shares
        self.orders[ref] = (side, price)
        s = self._next_seq
        self._next_seq += 1
        self.seq[ref] = s
        self._by_seq[s] = ref
        self.age[side].add(s)

    def reduce(self, ref: int, shares: int) -> int:
        """Remove `shares` from an order; returns what is left of it."""
        try:
            side, price = self.orders[ref]
        except KeyError:
            raise BookError(f"unknown order ref {ref}") from None
        lv = self.side[side][price]
        have = lv.q[ref]
        if not 0 < shares <= have:
            raise BookError(f"reduce {shares} from ref {ref} holding {have}")
        lv.total -= shares
        if shares < have:
            lv.q[ref] = have - shares
            return have - shares
        del lv.q[ref]
        del self.orders[ref]
        if not lv.q:
            del self.side[side][price]
        s = self.seq.pop(ref)
        del self._by_seq[s]
        self.age[side].remove(s)
        return 0

    def apply_message(self, typ: int, ref: int, size: int, price: int, d: int) -> None:
        """Apply a real LOBSTER message by order ref (fast replay, no encoding)."""
        if typ == SUBMIT:
            self.add(ref, d, price, size)
        elif typ in (CANCEL, EXEC):
            self.reduce(ref, size)
        elif typ == DELETE:
            self.reduce(ref, self.shares(ref))

    def shares(self, ref: int) -> int:
        side, price = self.orders[ref]
        return self.side[side][price].q[ref]

    def best(self, side: int) -> int | None:
        levels = self.side[side]
        if not levels:
            return None
        return levels.peekitem(-1)[0] if side == BUY else levels.peekitem(0)[0]

    def n_levels(self, side: int) -> int:
        return len(self.side[side])

    def level_price(self, side: int, k: int) -> int:
        levels = self.side[side]
        return levels.peekitem(-1 - k)[0] if side == BUY else levels.peekitem(k)[0]

    def level_index(self, side: int, price: int) -> int:
        levels = self.side[side]
        i = levels.index(price)
        return len(levels) - 1 - i if side == BUY else i

    def queue_pos(self, ref: int, from_front: bool = False) -> int:
        """Place of `ref` in its level's queue, from the back (0 = newest) or the front."""
        side, price = self.orders[ref]
        q = self.side[side][price].q
        for k, r in enumerate(iter(q) if from_front else reversed(q)):
            if r == ref:
                return k
        raise BookError(f"ref {ref} missing from its queue")

    def n_orders(self, side: int) -> int:
        return len(self.age[side])

    def recency_rank(self, ref: int) -> int:
        """How many live orders on the same side arrived after `ref` (0 = the newest)."""
        side = self.orders[ref][0]
        ages = self.age[side]
        return len(ages) - 1 - ages.index(self.seq[ref])

    def ref_by_rank(self, side: int, r: int) -> int:
        return self._by_seq[self.age[side][-1 - r]]

    def ref_at(self, side: int, price: int, qpos: int, from_front: bool = False) -> int:
        q = self.side[side][price].q
        return next(islice(iter(q) if from_front else reversed(q), qpos, None))

    def queue_len(self, side: int, price: int) -> int:
        return len(self.side[side][price].q)

    def snapshot(self, levels: int) -> list[int]:
        """LOBSTER orderbook row: askP1, askV1, bidP1, bidV1, askP2, ... (4 * levels ints)."""
        asks, bids = self.side[SELL], self.side[BUY]
        na, nb = len(asks), len(bids)
        row: list[int] = []
        for i in range(levels):
            if i < na:
                p, lv = asks.peekitem(i)
                row += (p, lv.total)
            else:
                row += (EMPTY_ASK, 0)
            if i < nb:
                p, lv = bids.peekitem(nb - 1 - i)
                row += (p, lv.total)
            else:
                row += (EMPTY_BID, 0)
        return row


class Simulator:
    """Applies actions to a SimBook and emits LOBSTER rows."""

    def __init__(self, book: SimBook, t_ns: int, levels: int = 10, next_ref: int = GEN_REF0,
                 strict: bool = False) -> None:
        self.book = book
        self.t = int(t_ns)
        self.levels = levels
        self.next_ref = next_ref
        self.strict = strict
        self.repairs: Counter = Counter()
        self.skipped = 0
        self.last_action: tuple | None = None  # the action as applied, after repairs
        self._anchor = {BUY: book.best(BUY), SELL: book.best(SELL)}

    def _repair(self, why: str) -> None:
        if self.strict:
            raise SimError(why)
        self.repairs[why] += 1

    def ref_price(self, side: int) -> int:
        """Own-side best, else one tick inside the opposite best, else the last value returned."""
        b = self.book.best(side)
        if b is None:
            o = self.book.best(-side)
            b = o - side * TICK if o is not None else self._anchor[side]
            if b is None:
                raise SimError("both sides empty and no anchor price")
        self._anchor[side] = b
        return b

    def step(self, dt: int, typ: int, side: int, p: int, qpos: int, size: int,
             ref: int | None = None, snapshot: bool = True):
        """Apply one action. Returns (message tuple, snapshot list | None), or None if the
        action targets an empty side. `ref` forces the order id of a submit (replay)."""
        bk = self.book
        if dt < 0:
            self._repair("dt")
            dt = 0
        if side not in (BUY, SELL):
            self._repair("side")
            side = BUY if side > 0 else SELL
        t = self.t + dt

        if typ == SUBMIT:
            if size <= 0:
                self._repair("size")
                size = 1
            ref_px = self.ref_price(side)
            price = ref_px - side * p * TICK
            opp = bk.best(-side)
            if opp is not None and side * (price - opp) >= 0:
                self._repair("cross")
                price = opp - side * TICK
            if price <= 0:
                self._repair("price")
                price = TICK
            p = side * (ref_px - price) // TICK
            if ref is None:
                ref = self.next_ref
                self.next_ref += 1
            bk.add(ref, side, price, size)
            msg = (t, SUBMIT, ref, size, price, side)

        elif typ in (CANCEL, DELETE, EXEC):
            n = bk.n_levels(side)
            if n == 0:
                self._repair("empty_side")
                self.skipped += 1
                return None
            if p < 0 and typ != EXEC:                      # pointer: r-th newest order
                r, n_live = -p - 1, bk.n_orders(side)
                if r >= n_live:
                    self._repair("rank")
                    r = n_live - 1
                    p = -r - 1
                qpos = 0
                target = bk.ref_by_rank(side, r)
                price = bk.orders[target][1]
            else:
                if not 0 <= p < n:
                    self._repair("level")
                    p = min(max(p, 0), n - 1)
                price = bk.level_price(side, p)
                qlen = bk.queue_len(side, price)
                if not 0 <= qpos < qlen:
                    self._repair("qpos")
                    qpos = min(max(qpos, 0), qlen - 1)
                target = bk.ref_at(side, price, qpos, from_front=typ == EXEC)
            have = bk.side[side][price].q[target]
            if typ == DELETE or (typ == EXEC and size == 0):   # 0 = whole order
                size = have
            elif not 0 < size <= have:
                self._repair("size")
                size = min(max(size, 1), have)
            whole = typ == EXEC and size == have
            bk.reduce(target, size)
            msg = (t, typ, target, size, price, side)

        elif typ == HIDDEN:
            if size <= 0:
                self._repair("size")
                size = 1
            ref_px = self.ref_price(side)
            price = ref_px - side * p * HALF_TICK
            if price <= 0:
                self._repair("price")
                price = HALF_TICK
                p = side * (ref_px - price) // HALF_TICK
            msg = (t, HIDDEN, 0, size, price, side)

        elif typ in (CROSS, HALT):
            msg = (t, typ, 0, size, p, side)

        else:
            self._repair("type")
            self.skipped += 1
            return None

        self.t = t
        a_size = 0 if typ == EXEC and whole else size
        self.last_action = (dt, typ, side, p, qpos if typ in (CANCEL, DELETE, EXEC) else 0, a_size)
        return msg, (bk.snapshot(self.levels) if snapshot else None)

    def run(self, actions: np.ndarray, snapshot: bool = True):
        """Apply an [N, 6] action array, skipping inapplicable ones.

        Returns (messages [M, 6], book [M, 4*levels] or None, applied actions [M, 6]), M <= N.
        """
        msgs, books, applied = [], [], []
        for a in np.asarray(actions).tolist():
            out = self.step(*a, snapshot=snapshot)
            if out is not None:
                msgs.append(out[0])
                applied.append(self.last_action)
                if snapshot:
                    books.append(out[1])
        m = np.array(msgs, dtype=np.int64).reshape(-1, 6)
        b = np.array(books, dtype=np.int64).reshape(-1, 4 * self.levels) if snapshot else None
        return m, b, np.array(applied, dtype=np.int64).reshape(-1, N_FIELDS)


def encode(msgs: np.ndarray, init_orders: np.ndarray, t0: int | None = None,
           levels: int = 10, pointer: int = PTR_RANKS,
           whole_fill: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Real LOBSTER messages -> actions, replaying them through a strict Simulator.

    init_orders is the order-level book before msgs[0] (lobster.preopen_orders, or
    states_at for a later start), oldest order first; t0 the time of the event before msgs[0]
    (default msgs[0]). A cancel / delete whose target is among the `pointer` newest orders on
    its side is encoded as a pointer (pointer=0: always level + queue position). whole_fill:
    an execution that consumes the whole order gets size 0. (pointer=0, whole_fill=False) is
    the v1 encoding.
    Returns (actions [N, 6] int64, inexact [N] bool). A row is inexact when its price is off
    the tick grid the action can express (1-6 sub-dollar stub quotes per day, e.g. a $0.0001
    bid, which may be legally sub-penny) or it touches an order whose submit was inexact; decoding reproduces every other
    row exactly, which this function checks as it goes.
    """
    m = np.asarray(msgs)
    sim = Simulator(SimBook.from_orders(init_orders), int(m[0, 0]) if t0 is None else t0,
                    levels, strict=True)
    bk = sim.book
    actions = np.empty((len(m), N_FIELDS), dtype=np.int64)
    inexact = np.zeros(len(m), dtype=bool)
    off_grid_refs: set[int] = set()
    for i, (t, typ, ref, size, price, d) in enumerate(m.tolist()):
        qpos = 0
        if typ in (SUBMIT, HIDDEN):
            unit = TICK if typ == SUBMIT else HALF_TICK
            num = d * (sim.ref_price(d) - price)
            p, r = divmod(num, unit)
            if r:
                p = int(round(num / unit))
                inexact[i] = True
                if typ == SUBMIT:
                    off_grid_refs.add(ref)
        elif typ in (CANCEL, DELETE, EXEC):
            side_b, price_b = bk.orders[ref]
            if side_b != d:
                raise SimError(f"row {i}: side {d} but ref {ref} rests on side {side_b}")
            r = bk.recency_rank(ref) if typ != EXEC and pointer else pointer
            if r < pointer:
                p = -r - 1
            else:
                p = bk.level_index(side_b, price_b)
                qpos = bk.queue_pos(ref, from_front=typ == EXEC)
            inexact[i] = ref in off_grid_refs
        else:
            p = price
        a_size = 0 if typ == EXEC and whole_fill and size == bk.shares(ref) else size
        a = (t - sim.t, typ, d, p, qpos, a_size)
        actions[i] = a
        sim.strict = not inexact[i]  # a rounded stub price (e.g. $0.0001 -> $0) may need a repair
        out = sim.step(*a, ref=ref if typ == SUBMIT else None, snapshot=False)
        if not inexact[i] and out[0] != (t, typ, ref, size, price, d):
            raise SimError(f"row {i}: decode gave {out[0]}, real {(t, typ, ref, size, price, d)}")
    return actions, inexact


def states_at(msgs: np.ndarray, init_orders: np.ndarray, starts) -> list[np.ndarray]:
    """Order-level book just before each row index in `starts` (ascending), as [K, 4] arrays."""
    starts = list(starts)
    if starts != sorted(starts):
        raise ValueError("starts must be ascending")
    bk = SimBook.from_orders(init_orders)
    out, j = [], 0
    m = np.asarray(msgs)
    for s in starts:
        for _t, typ, ref, size, price, d in m[j:s].tolist():
            bk.apply_message(typ, ref, size, price, d)
        j = s
        out.append(bk.to_orders())
    return out


def verify_decode(msgs: np.ndarray, book: np.ndarray, init_orders: np.ndarray,
                  actions: np.ndarray, inexact: np.ndarray, levels: int = 10) -> dict:
    """Decode `actions` with fresh order ids and compare every row with the real data.

    Messages are compared on all columns but the order id (generated ids are new); ids are
    checked to map one-to-one onto real ids. Book snapshots are compared on every row.
    """
    m = np.asarray(msgs)
    sim = Simulator(SimBook.from_orders(init_orders), int(m[0, 0] - actions[0, DT]), levels,
                    strict=True)
    real_to_gen: dict[int, int] = {}
    gen_to_real: dict[int, int] = {}
    bad_msg = bad_ref = bad_book = 0
    first_bad = None
    for i, a in enumerate(np.asarray(actions).tolist()):
        sim.strict = not inexact[i]
        msg, snap = sim.step(*a)
        t, typ, ref, size, price, d = m[i].tolist()
        if not inexact[i]:
            if (msg[0], msg[1], msg[3], msg[4], msg[5]) != (t, typ, size, price, d):
                bad_msg += 1
            if ref and (real_to_gen.setdefault(ref, msg[2]) != msg[2]
                        or gen_to_real.setdefault(msg[2], ref) != ref):
                bad_ref += 1
        if snap != book[i].tolist():
            bad_book += 1
        if first_bad is None and (bad_msg or bad_ref or bad_book):
            first_bad = i
    return {"rows": len(m), "inexact_rows": int(np.sum(inexact)), "message_mismatches": bad_msg,
            "id_mismatches": bad_ref, "book_mismatches": bad_book, "first_bad_row": first_bad,
            "repairs": dict(sim.repairs)}
