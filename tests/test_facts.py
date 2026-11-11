import numpy as np

from lobgen.book import BUY, SELL
from lobgen.facts import discrepancies, stylized_facts
from lobgen.sim import TICK, SimBook, Simulator

NS = 1_000_000_000


def _runs(n_runs=6, n=400, seed=0):
    rng = np.random.default_rng(seed)
    init = np.array([(i + 1, BUY if i % 2 else SELL,
                      1_000_000 + (1 if i % 2 == 0 else -1) * TICK * (1 + i // 2), 100)
                     for i in range(40)], dtype=np.int64)
    out = []
    for _ in range(n_runs):
        sim = Simulator(SimBook.from_orders(init), 35_000 * NS)
        acts = np.column_stack([rng.integers(0, 10**6, n), rng.choice([1, 1, 3, 4], n),
                                rng.choice([BUY, SELL], n), rng.integers(-1, 4, n),
                                rng.integers(0, 3, n), rng.choice([100, 200, 0], n)])
        m, b, _ = sim.run(acts)
        out.append((m, b))
    return out


def test_facts_are_well_formed_and_zero_against_themselves():
    runs = _runs()
    f = stylized_facts(runs)
    assert abs(sum(f["spread"]) - 1) < 1e-9 and abs(sum(f["mix"]) - 1) < 1e-9
    assert len(f["vol_acf"]) == 100 and len(f["sign_acf"]) == 50 and len(f["impact"]) == 50
    assert all(np.isfinite(f["ret_std"])) and f["depth_ask"][0] == 1.0
    assert all(v == 0 for v in discrepancies(f, f).values())
    g = stylized_facts(_runs(seed=1))
    assert sum(discrepancies(f, g).values()) > 0
