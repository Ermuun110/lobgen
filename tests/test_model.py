import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lobgen.book import BUY, SELL  # noqa: E402
from lobgen.data import Day, make_windows  # noqa: E402
from lobgen.features import BatchState, states_before  # noqa: E402
from lobgen.model import (EventData, LOBModel, ModelConfig, ModelGenerator, SpikeHead,  # noqa: E402
                          choose_atoms, fit_atoms, token_features)
from lobgen.sim import TICK, SimBook, Simulator  # noqa: E402
from lobgen.validate import check_book_transitions  # noqa: E402

NS = 1_000_000_000


def _head(atoms, lo, seed=0):
    torch.manual_seed(seed)
    h = SpikeHead(16, atoms, n_mix=4, lo=lo, hidden=32)
    k = len(atoms) + 1 + 2 * 4
    with torch.no_grad():  # components spread over ~2 orders of magnitude, moderate widths
        h.net[-1].bias.normal_(0, 1.5)
        h.net[-1].bias[k:] -= 1.0
    return h


@pytest.mark.parametrize("atoms,lo,span", [([0, 100], 0, 20_000), ([], 1, 20_000),
                                           ([-1, 0, 3], None, 20_000)])
def test_spike_head_is_a_distribution_over_integers(atoms, lo, span):
    h = _head(atoms, lo)
    x = torch.randn(3, 16)
    start = lo if lo is not None else -span
    y = torch.arange(start, span + 1)
    lp = h.log_prob(x[:, None, :].expand(3, len(y), 16), y[None].expand(3, len(y)))
    total = lp.exp().sum(1)
    # Whatever mass sits beyond +-span is in the far tail of a logistic in log space.
    assert torch.all((total - 1).abs() < 2e-3), total


def test_spike_head_sampler_matches_its_likelihood():
    h = _head([0, 100], 0, seed=1)
    x = torch.randn(1, 16).expand(40_000, 16)
    s = h.sample(x).numpy()
    vals, counts = np.unique(s, return_counts=True)
    top = vals[np.argsort(-counts)[:6]]
    lp = h.log_prob(x[:len(top)], torch.tensor(top))
    emp = np.array([np.mean(s == v) for v in top])
    np.testing.assert_allclose(emp, lp.exp().detach().numpy(), atol=0.012)


def test_choose_atoms_and_fit():
    v = np.array([0] * 50 + [100] * 30 + list(range(1, 21)))
    assert choose_atoms(v, 0.05, 8) == [0, 100]
    assert choose_atoms(v, 0.05, 1) == [0]


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


def test_token_features_chunking_is_consistent():
    d = _toy_day(500)
    c, x, s = token_features(d.actions, d.msgs, d.book)
    c1, x1, s1 = token_features(d.actions[:200], d.msgs[:200], d.book[:200])
    c2, x2, s2 = token_features(d.actions[200:], d.msgs[200:], d.book[200:], s1)
    np.testing.assert_array_equal(c, np.vstack([c1, c2]))
    np.testing.assert_allclose(x, np.vstack([x1, x2]))
    assert np.isfinite(x).all() and s.prev_mid == s2.prev_mid
    for f in ("num", "den", "tr_num", "tr_den", "since", "m_num", "m_den"):
        np.testing.assert_allclose(getattr(s, f), getattr(s2, f), err_msg=f)
    # day-so-far means are means: the event-type shares sum to 1 at every half-life
    from lobgen.features import N_BASE, N_REG
    for h in range(2):
        np.testing.assert_allclose(x[:, N_BASE + h * N_REG:N_BASE + h * N_REG + 5].sum(1), 1, atol=1e-5)
    st = states_before(d.actions, d.msgs, d.book, [0, 137, 500])
    assert st[0].den.sum() == 0
    np.testing.assert_allclose(st[2].num, s.num)
    _, x3, _ = token_features(d.actions[137:], d.msgs[137:], d.book[137:], st[1])
    np.testing.assert_allclose(x3, x[137:], atol=1e-6)


def test_flow_features_match_a_plain_loop():
    """Order-flow memory columns against a direct re-implementation over the messages."""
    from lobgen.features import MID_HALF_LIVES, N_ALL, N_CONT, TRADE_HALF_LIVES, _mids
    d = _toy_day(800)
    _, x, _ = token_features(d.actions, d.msgs, d.book)
    T = len(TRADE_HALF_LIVES)
    num, den = np.zeros((T, 2)), np.zeros(T)
    mnum, mden = np.zeros(len(MID_HALF_LIVES)), np.zeros(len(MID_HALF_LIVES))
    since, last_mid, n_trades = 1e4, np.nan, 0
    mids = _mids(d.book)
    for i, m in enumerate(d.msgs):
        since += 1
        if m[1] == 4:
            since, n_trades = 0, n_trades + 1
            eps, ls = -float(m[5]), np.log1p(m[3]) / 5
            for k, h in enumerate(TRADE_HALF_LIVES):
                g = 0.5 ** (1 / h)
                den[k] = g * den[k] + 1
                num[k] = g * num[k] + [eps, eps * ls]
        last_mid = mids[i] if not np.isnan(mids[i]) else last_mid
        if not np.isnan(last_mid):
            for k, h in enumerate(MID_HALF_LIVES):
                g = 0.5 ** (1 / h)
                mden[k], mnum[k] = g * mden[k] + 1, g * mnum[k] + last_mid
        f = x[i, N_CONT:N_ALL]
        exp_sign = num[:, 0] / den if n_trades else np.zeros(T)
        np.testing.assert_allclose(f[:T], exp_sign, atol=1e-5)
        np.testing.assert_allclose(f[2 * T], np.log1p(since) / 10, atol=1e-5)
        if not np.isnan(last_mid):
            np.testing.assert_allclose(f[2 * T + 1:], np.clip((last_mid - mnum / mden) / 10, -3, 3),
                                       atol=1e-4)
    assert n_trades > 20


def test_batch_state_steps_match_the_sequential_stream():
    from test_market import synthetic_market
    d = _toy_day(300)
    mk = synthetic_market()
    _, x, _ = token_features(d.actions, d.msgs, d.book, market=mk)
    st = states_before(d.actions, d.msgs, d.book, [100, 100])   # two identical streams
    bs = BatchState(st, [mk, mk])
    for i in range(100, 300):
        _, xb = bs.step(d.actions[[i, i]], d.msgs[[i, i]], d.book[[i, i]])
        np.testing.assert_allclose(xb[0], x[i], atol=1e-6)
        np.testing.assert_allclose(xb[1], x[i], atol=1e-6)


def test_own_clock_features_match_a_plain_loop():
    from lobgen.features import N_ALL, _mids
    from lobgen.market import CLOCK_TAUS_S
    d = _toy_day(600)
    _, x, _ = token_features(d.actions, d.msgs, d.book)
    mids = _mids(d.book)
    E, last_x, last_t = None, np.nan, None
    for i, m in enumerate(d.msgs):
        if E is not None and m[0] > last_t:
            a = np.exp(-(m[0] - last_t) / (np.array(CLOCK_TAUS_S) * NS))
            E = a * E + (1 - a) * last_x
        if not np.isnan(mids[i]):
            if E is None:
                E = np.full(len(CLOCK_TAUS_S), np.log(mids[i]))
            last_x = np.log(mids[i])
        last_t = m[0]
        want = np.zeros(len(CLOCK_TAUS_S)) if E is None else np.clip((last_x - E) * 1e3, -5, 5)
        np.testing.assert_allclose(x[i, N_ALL:N_ALL + len(CLOCK_TAUS_S)], want, atol=1e-5)
    assert np.abs(x[:, N_ALL:N_ALL + len(CLOCK_TAUS_S)]).max() > 0.01


def _tiny(day, ctx=32, regime=True, flow=False, market=False):
    atoms = fit_atoms(day.actions, 0.02, 8)
    return LOBModel(ModelConfig(d_model=32, n_layers=1, n_heads=2, ctx=ctx, n_mix=3, atoms=atoms,
                                encoding=(16, True), regime=regime, flow=flow, market=market))


@pytest.mark.parametrize("regime,flow,market", [(True, False, False), (False, False, False),
                                                (True, True, False), (True, True, True)])
def test_model_trains_one_step_and_samples(regime, flow, market):
    from test_market import synthetic_market
    d = _toy_day()
    mk = synthetic_market() if market else None
    data = EventData([EventData.featurize(d, margin_s=0, market=mk)], ctx=32)
    model = _tiny(d, regime=regime, flow=flow, market=market)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    batch = data.batch(4, rng, "cpu")
    first = model.nll(*batch)
    for _ in range(30):
        loss = model.nll(*batch)["event"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert set(first) == {"ts", "p", "q", "size", "dt", "event"}
    assert model.nll(*batch)["event"].item() < first["event"].item()
    a = model.sample_next(torch.randn(5, 32))
    assert a.shape == (5, 6) and set(a[:, 1].tolist()) <= {1, 2, 3, 4, 5}


@pytest.mark.parametrize("flow,market", [(False, False), (True, False), (True, True)])
def test_generator_context_features_match_training_features(flow, market):
    """Rollout buffers must hold exactly the features the model was trained on."""
    from test_market import synthetic_market
    d = _toy_day()
    mk = synthetic_market() if market else None
    w = make_windows(d, 2, cond_len=64, gen_len=40, seed=0, margin_s=0)
    for x in w:
        x.market = mk
    model = _tiny(d, flow=flow, market=market).eval()
    seen = []
    orig = model.trunk
    model.trunk = lambda cat, cont: (seen.append((cat.clone(), cont.clone())), orig(cat, cont))[1]
    gen = ModelGenerator(model, batch=2).generate(w, 40)
    cat, cont = seen[-1]  # context at the last step: 32 tokens, the last 31 generated
    for b, (win, g) in enumerate(zip(w, gen)):
        # Training features of the same stream: the real day from its first row, then the run.
        acts = np.vstack([d.actions[:win.start], g.actions])
        msgs = np.vstack([d.msgs[:win.start], g.msgs])
        book = np.vstack([d.book[:win.start], g.book])
        c, x, _ = token_features(acts, msgs, book, market=mk)
        np.testing.assert_array_equal(cat[b].numpy(), c[-40 - 1:-1][-32:])
        np.testing.assert_allclose(cont[b].numpy(), x[-40 - 1:-1][-32:], atol=1e-6)


def test_generated_actions_replay_to_the_same_output():
    from lobgen.baselines import HistoryBootstrap
    d = _toy_day()
    w = make_windows(d, 2, cond_len=64, gen_len=40, seed=0, margin_s=0)
    for gen in (ModelGenerator(_tiny(d).eval(), batch=2), HistoryBootstrap()):
        for win, g in zip(w, gen.generate(w, 40)):
            sim = Simulator(SimBook.from_orders(win.state), win.t_prev, strict=True)
            m, b, a = sim.run(g.actions)
            np.testing.assert_array_equal(m, g.msgs)
            np.testing.assert_array_equal(a, g.actions)


def test_model_refuses_windows_with_another_encoding():
    d = _toy_day()
    w = make_windows(d, 2, cond_len=64, gen_len=10, seed=0, margin_s=0)
    for x in w:
        x.encoding = (0, False)
    with pytest.raises(ValueError, match="encoding"):
        ModelGenerator(_tiny(d).eval(), batch=2).generate(w, 10)


def test_model_generator_output_is_a_valid_book():
    d = _toy_day()
    w = make_windows(d, 2, cond_len=64, gen_len=40, seed=0, margin_s=0)
    gen = ModelGenerator(_tiny(d).eval(), batch=2).generate(w, 40)
    for g in gen:
        assert g.msgs.shape == (40, 6) and g.book.shape == (40, 40)
        assert check_book_transitions(g.msgs, g.book, 10)["violations"] == 0
        two = (g.book[:, 1] > 0) & (g.book[:, 3] > 0)
        assert (g.book[two, 0] > g.book[two, 2]).all()
        assert (np.diff(g.msgs[:, 0]) >= 0).all() and g.msgs[0, 0] >= w[0].t_prev


def test_train_saves_best_and_last(tmp_path):
    from lobgen.model import load, train
    d = _toy_day()
    data = EventData([EventData.featurize(d, margin_s=0)], ctx=32)
    cfg = _tiny(d).cfg
    res = train(cfg, data, data, tmp_path / "m.pt", steps=4, batch=4, warmup=1, eval_every=2,
                eval_batches=1, device="cpu", log=lambda s: None)
    assert (tmp_path / "m.pt").exists() and (tmp_path / "m_last.pt").exists()
    assert len(res["history"]) == 2
    assert torch.load(tmp_path / "m_last.pt", weights_only=False)["step"] == 4
    assert not (tmp_path / "m_state.pt").exists()      # removed once the run is finished
    load(tmp_path / "m_last.pt", device="cpu")


def test_interrupted_training_resumes_bit_identically(tmp_path):
    from lobgen.model import train
    d = _toy_day()
    data = EventData([EventData.featurize(d, margin_s=0)], ctx=32)
    cfg = _tiny(d).cfg
    kw = dict(steps=6, batch=4, warmup=1, eval_every=2, eval_batches=1, device="cpu")
    ref = train(cfg, data, data, tmp_path / "ref.pt", log=lambda s: None, **kw)

    class Interrupt(Exception):
        pass

    def crash_at_step_4(s):
        if s.startswith("step=4 "):
            raise Interrupt                  # mid-run, after the step-2 state was written
    with pytest.raises(Interrupt):
        train(cfg, data, data, tmp_path / "m.pt", log=crash_at_step_4, **kw)
    assert (tmp_path / "m_state.pt").exists() and not (tmp_path / "m_last.pt").exists()
    lines = []
    res = train(cfg, data, data, tmp_path / "m.pt", log=lines.append, **kw)
    assert lines[0] == "resumed from m_state.pt at step 2"
    a = torch.load(tmp_path / "ref_last.pt", weights_only=False)["state"]
    b = torch.load(tmp_path / "m_last.pt", weights_only=False)["state"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    strip = lambda h: [{k: v for k, v in r.items() if k != "minutes"} for r in h]  # noqa: E731
    assert strip(res["history"]) == strip(ref["history"])
    assert not (tmp_path / "m_state.pt").exists()

    # A state file from other settings is ignored, not silently continued.
    torch.save({**torch.load(tmp_path / "ref_last.pt", weights_only=False), "run": {"steps": 99}},
               tmp_path / "x_state.pt")
    lines = []
    train(cfg, data, data, tmp_path / "x.pt", log=lines.append, **kw)
    assert "different run" in lines[0]


def test_featurize_on_disk_matches_ram_and_trains_identically(tmp_path):
    """--disk-features (memory-mapped files) must not change a single value or training step."""
    from lobgen.model import train
    d = _toy_day()
    ram = EventData.featurize(d, margin_s=0)
    disk = EventData.featurize(d, margin_s=0, out_dir=tmp_path, chunk=37)   # many chunks
    small = EventData.featurize(d, margin_s=0, chunk=37)
    for a, b, c in zip(ram[1:], disk[1:], small[1:]):
        assert isinstance(b, np.memmap)
        assert a.dtype == b.dtype == c.dtype and np.array_equal(a, b) and np.array_equal(a, c)
    cfg = _tiny(d).cfg
    kw = dict(steps=4, batch=4, warmup=1, eval_every=2, eval_batches=1, device="cpu",
              log=lambda s: None)
    r1 = train(cfg, EventData([ram], 32), EventData([ram], 32), tmp_path / "a.pt", **kw)
    r2 = train(cfg, EventData([disk], 32), EventData([disk], 32), tmp_path / "b.pt", **kw)
    strip = lambda h: [{k: v for k, v in r.items() if k != "minutes"} for r in h]  # noqa: E731
    assert strip(r1["history"]) == strip(r2["history"])
