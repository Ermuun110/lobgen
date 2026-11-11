"""Converted days, evaluation windows, and the generator interface every generator implements.

A *window* is a point in a real day: the `cond_len` real events before it (what a generator
may condition on), the order-level book at that moment, and the `gen_len` real events after it
(what the generator's output is compared with). Generators get the context and the book state,
never the real continuation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FeatState, states_before
from .sim import MODELED_TYPES, TYPE, SimBook, Simulator, states_at

NS = 1_000_000_000
OPEN_S, CLOSE_S = 34_200, 57_600


@dataclass
class Day:
    ticker: str
    date: str
    msgs: np.ndarray
    book: np.ndarray | None                # None when loaded with book=False
    actions: np.ndarray | None = None      # [N, 6] simulator actions (scripts/build_actions.py)
    init_orders: np.ndarray | None = None  # order-level book before msgs[0], oldest first
    encoding: tuple = (0, False)           # encoding of `actions`: sim.encode (pointer, whole_fill)

    @classmethod
    def load(cls, npz: str | Path, actions: bool = True, book: bool = True) -> "Day":
        """book=False skips the snapshot array (~3 GB for a GOOG day) when only actions are needed."""
        p = Path(npz)  # GOOG_2025-12-12_34200000_57600000_10.npz
        ticker, date = p.name.split("_")[:2]
        z = np.load(p)
        day = cls(ticker, date, z["messages"], z["book"] if book else None)
        a = p.with_name(p.stem + "_actions.npz")
        if actions:
            if not a.exists():
                raise FileNotFoundError(f"{a} missing: run scripts/build_actions.py")
            za = np.load(a)
            day.actions, day.init_orders = za["actions"], za["init_orders"]
            day.encoding = (int(za["pointer"]) if "pointer" in za.files else 0,
                            bool(za["whole_fill"]) if "whole_fill" in za.files else False)
        return day

    def time_of_day_ns(self) -> np.ndarray:
        return self.msgs[:, 0]

    def model_rows(self, margin_s: int = 300) -> np.ndarray:
        """Row mask: modeled event types, away from the open/close crosses."""
        t = self.msgs[:, 0]
        return (np.isin(self.msgs[:, 1], MODELED_TYPES)
                & (t >= (OPEN_S + margin_s) * NS) & (t < (CLOSE_S - margin_s) * NS))


@dataclass
class Window:
    i: int
    start: int              # day row index of the first continuation event
    t_prev: int             # time of the last context event (ns)
    state: np.ndarray       # order-level book just before `start`, [K, 4]
    cond_msgs: np.ndarray
    cond_book: np.ndarray
    cond_actions: np.ndarray
    real_msgs: np.ndarray
    real_book: np.ndarray
    real_actions: np.ndarray
    encoding: tuple = (0, False)        # action encoding of cond_actions / real_actions
    feat_state: FeatState | None = None  # model feature stream just before cond_msgs[0]
    market: object = None                # market.Market the run is conditioned on (v4 models)


def pick_windows(day: Day, n: int, cond_len: int, gen_len: int, rng: np.random.Generator,
                 margin_s: int = 300) -> np.ndarray:
    """n non-overlapping window starts (index of first continuation msg), away from open/close."""
    t = day.time_of_day_ns()
    lo = max(cond_len, int(np.searchsorted(t, (OPEN_S + margin_s) * NS)))
    hi = min(len(t) - gen_len, int(np.searchsorted(t, (CLOSE_S - margin_s) * NS)))
    span = cond_len + gen_len
    if (hi - lo) < n * span:
        raise ValueError(f"{day.ticker} {day.date}: not enough events for {n} windows")
    # Stratified: one start per equal slice of the day keeps time-of-day coverage even.
    edges = np.linspace(lo, hi - span, n + 1).astype(np.int64)
    return np.array([rng.integers(edges[k], max(edges[k] + 1, edges[k + 1] - span))
                     for k in range(n)]) + cond_len


def make_windows(day: Day, n: int, cond_len: int = 500, gen_len: int = 1000, seed: int = 0,
                 margin_s: int = 300, feat_states: bool = True) -> list[Window]:
    """feat_states: also run the model's feature stream from the open to each context start
    (what a model generator needs; a few seconds per day)."""
    starts = pick_windows(day, n, cond_len, gen_len, np.random.default_rng(seed), margin_s)
    return windows_at(day, starts, cond_len, gen_len, feat_states)


def windows_at(day: Day, starts, cond_len: int = 500, gen_len: int = 1000,
               feat_states: bool = True) -> list[Window]:
    """Windows whose continuation starts at the given day rows (ascending; e.g. the row right
    after a large market order, for the shock test)."""
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) and (starts[0] < cond_len or starts[-1] + gen_len > len(day.msgs)):
        raise ValueError("a window runs past the start or end of the day")
    states = states_at(day.msgs, day.init_orders, starts)
    fs = (states_before(day.actions, day.msgs, day.book, (starts - cond_len).tolist())
          if feat_states else [None] * len(starts))
    out = []
    for i, (s, st, f) in enumerate(zip(starts.tolist(), states, fs)):
        c, e = slice(s - cond_len, s), slice(s, s + gen_len)
        out.append(Window(i, s, int(day.msgs[s - 1, 0]), st,
                          day.msgs[c], day.book[c], day.actions[c],
                          day.msgs[e], day.book[e], day.actions[e], day.encoding, f))
    return out


@dataclass
class Generated:
    msgs: np.ndarray
    book: np.ndarray
    repairs: Counter = field(default_factory=Counter)
    skipped: int = 0
    actions: np.ndarray | None = None  # the actions as applied (after repairs), [N, 6]


class Generator:
    """Produces a continuation for each window. Subclasses set `name` and override generate()
    (or, for per-event samplers, subclass ActionGenerator)."""

    name = "generator"

    def generate(self, windows: list[Window], n_events: int, seed: int = 0) -> list[Generated]:
        raise NotImplementedError


class ActionGenerator(Generator):
    """A generator that emits one action at a time and lets the simulator build the book.

    Subclasses implement reset(window, rng) and sample(sim, rng) -> action tuple, and may use
    observe(applied_action) to track their own history.
    """

    max_skips = 1000  # inapplicable actions (2-4 on an empty side) are redrawn

    def reset(self, w: Window, rng: np.random.Generator) -> None:
        pass

    def sample(self, sim: Simulator, rng: np.random.Generator) -> tuple:
        raise NotImplementedError

    def observe(self, action: tuple) -> None:
        pass

    def rollout(self, w: Window, n_events: int, rng: np.random.Generator,
                levels: int = 10) -> Generated:
        sim = Simulator(SimBook.from_orders(w.state), w.t_prev, levels)
        self.reset(w, rng)
        msgs, books, acts = [], [], []
        while len(msgs) < n_events:
            out = sim.step(*self.sample(sim, rng))
            if out is None:
                if sim.skipped > self.max_skips:
                    raise RuntimeError(f"{self.name}: window {w.i} stuck after {sim.skipped} skips")
                continue
            msgs.append(out[0])
            books.append(out[1])
            acts.append(sim.last_action)
            self.observe(sim.last_action)
        return Generated(np.array(msgs, dtype=np.int64), np.array(books, dtype=np.int64),
                         sim.repairs, sim.skipped, np.array(acts, dtype=np.int64))

    def generate(self, windows: list[Window], n_events: int, seed: int = 0) -> list[Generated]:
        return [self.rollout(w, n_events, np.random.default_rng([seed, w.i])) for w in windows]


def modeled(actions: np.ndarray) -> np.ndarray:
    return actions[np.isin(actions[:, TYPE], MODELED_TYPES)]
