"""Reference generators (the "perfect" end of the scale) and dumb baselines (the "clearly bad" end).

References, built from real data:
  Replay        the window's real continuation re-played through the simulator. Output equals
                the real data except for order ids, so a detective must score exactly chance;
                a check on the simulator and the detective, not a floor.
  RealOtherDay  the real continuation of another day at the same time of day. The noise floor:
                a perfect model of "a day like this one" cannot beat it.

Baselines, all driving the simulator so their output is a valid book:
  HistoryBootstrap  each event is a random draw from the window's own conditioning events.
  PooledIID         each event is a random draw from all events of the training days.
  Markov            first-order Markov chain over a 40-class summary of the previous event
                    (type x side x price bucket); the next event is a random training event that
                    followed the same class, so all its fields (dt, price, size...) come jointly.
"""

from __future__ import annotations

import numpy as np

from .data import ActionGenerator, Day, Generated, Generator, Window, modeled
from .sim import CANCEL, DELETE, EXEC, PRICE, SIDE, SUBMIT, TYPE, Simulator

N_CLASSES = 40


class Replay(ActionGenerator):
    name = "replay"

    def reset(self, w: Window, rng) -> None:
        self._it = iter(w.real_actions.tolist())

    def sample(self, sim: Simulator, rng) -> tuple:
        return next(self._it)


class RealOtherDay(Generator):
    def __init__(self, other: Day) -> None:
        self.other = other
        self.name = f"real_{other.date}"

    def generate(self, windows: list[Window], n_events: int, seed: int = 0) -> list[Generated]:
        t = self.other.time_of_day_ns()
        out = []
        for w in windows:
            j = min(int(np.searchsorted(t, w.real_msgs[0, 0])), len(t) - n_events)
            out.append(Generated(self.other.msgs[j:j + n_events].copy(),
                                 self.other.book[j:j + n_events].copy()))
        return out


class _PoolSampler(ActionGenerator):
    """Draws whole action rows from a pool; draws are made in blocks for speed."""

    _block = 4096

    def _draw(self, pool: np.ndarray, rng) -> tuple:
        if self._k >= len(self._idx):
            self._idx = rng.integers(len(pool), size=self._block).tolist()
            self._k = 0
        j = self._idx[self._k]
        self._k += 1
        return tuple(pool[j].tolist())


class HistoryBootstrap(_PoolSampler):
    name = "history"

    def reset(self, w: Window, rng) -> None:
        self.pool = modeled(w.cond_actions)
        self._idx, self._k = [], 0

    def sample(self, sim: Simulator, rng) -> tuple:
        return self._draw(self.pool, rng)


def train_pool(days: list[Day], max_events: int = 3_000_000, seed: int = 0) -> list[np.ndarray]:
    """Modeled actions of each training day (open/close margins dropped), subsampled in
    contiguous blocks to at most max_events in total so consecutive pairs survive."""
    per_day = max_events // max(len(days), 1)
    rng = np.random.default_rng(seed)
    out = []
    for d in days:
        a = d.actions[d.model_rows()]
        if len(a) > per_day:
            s = int(rng.integers(0, len(a) - per_day))
            a = a[s:s + per_day]
        out.append(a)
    return out


class PooledIID(_PoolSampler):
    name = "pooled"

    def __init__(self, train: list[np.ndarray]) -> None:
        self.pool = np.concatenate(train)

    def reset(self, w: Window, rng) -> None:
        self._idx, self._k = [], 0

    def sample(self, sim: Simulator, rng) -> tuple:
        return self._draw(self.pool, rng)


def event_class(a: np.ndarray) -> np.ndarray:
    """40 classes: type (5) x side (2) x price bucket (4).

    Submit bucket: inside the spread / at the own best / 1 tick deeper / deeper.
    Cancel, delete, execution bucket: level 0 / 1 / 2+, or (cancel, delete) a pointer to one
    of the newest orders. Hidden executions: one bucket.
    """
    a = np.atleast_2d(a)
    typ, side, p = a[:, TYPE], a[:, SIDE], a[:, PRICE]
    bucket = np.where(typ == SUBMIT, np.clip(p, -1, 2) + 1,
                      np.where(np.isin(typ, [CANCEL, DELETE, EXEC]),
                               np.where(p < 0, 3, np.clip(p, 0, 2)), 0))
    return ((np.clip(typ, 1, 5) - 1) * 2 + (side > 0)) * 4 + bucket


class Markov(ActionGenerator):
    name = "markov"

    def __init__(self, train: list[np.ndarray]) -> None:
        prev = np.concatenate([event_class(a[:-1]) for a in train])
        nxt = np.concatenate([a[1:] for a in train])
        order = np.argsort(prev, kind="stable")
        self.next = nxt[order]
        self.bounds = np.searchsorted(prev[order], np.arange(N_CLASSES + 1))
        self.pool = nxt

    def reset(self, w: Window, rng) -> None:
        self.c = int(event_class(w.cond_actions[-1])[0])

    def sample(self, sim: Simulator, rng) -> tuple:
        lo, hi = self.bounds[self.c], self.bounds[self.c + 1]
        if hi == lo:  # class never seen in training: fall back to the marginal
            return tuple(self.pool[rng.integers(len(self.pool))].tolist())
        return tuple(self.next[rng.integers(lo, hi)].tolist())

    def observe(self, action: tuple) -> None:
        self.c = int(event_class(np.array(action))[0])
