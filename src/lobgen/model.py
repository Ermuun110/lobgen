"""Autoregressive transformer over simulator actions, with spike-aware output heads.

Input token for event i = the action as applied (type, side, price field, queue position,
size, dt) + the book right after it (spread, 10 levels of volume and price gaps, imbalance,
mid move) + time of day. Everything is relative to the book, so one model serves any stock.

Output for event i+1, factorised so that every field sees the fields chosen before it:
    (type, side)  ->  price field  ->  queue position  ->  size  ->  dt
(type, side) is a 10-way softmax. The four integer fields share one generic head, SpikeHead:

    P(x) = sum_j w_j [x == a_j]  +  w_c * sum_m pi_m * Logistic(T(x) bin | mu_m, s_m)

  * atoms a_j: every value holding >= `atom_share` of the training data for that field (up to
    `max_atoms`), picked automatically - e.g. dt = 0 ns (17-32% of events), size = 100 shares
    (40-78% of submits), price field = 0 (at the touch / level 0). Nothing is hand-chosen per
    field; the same rule runs on every field.
  * continuous part: mixture of logistics on T(x) = sign(x) log(1 + |x|), integrated over each
    integer's bin, so the likelihood is exact for integer data spanning nine orders of
    magnitude (dt from 1 ns to minutes).
A continuous density alone smears a point mass over its neighbours (the problem "atom
smoothing" works around in Jane Street's post); atoms make each spike exact. use_atoms=False
removes them (ablation).

Masks: queue position is only scored for cancels / deletes / executions; size is not scored
for deletes (the simulator sets it to the order's size).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Day, Generated, Generator, Window
from .features import (BOOK_COLS, FLOW_BOOK, N_ALL, N_BASE, N_CAT, N_CONT, N_FULL,  # noqa: F401
                       N_TS, OWN_COLS, P_HI, P_LO, Q_MAX, REGIME_BOOK, SIZE_BUCKETS, SPREAD_MAX,
                       BatchState,
                       FeatState, N_REG, HALF_LIVES, size_bucket, token_features, ts_decode,
                       ts_index)
from .sim import (CANCEL, DELETE, DT, EXEC, PRICE, QPOS, SIDE, SIZE, SUBMIT, TICK, TYPE,
                  SimBook, Simulator)

FIELDS = ("ts", "p", "q", "size", "dt")


# ----------------------------------------------------------------------------- atoms

def choose_atoms(values: np.ndarray, min_share: float, max_atoms: int) -> list[int]:
    v, c = np.unique(np.asarray(values), return_counts=True)
    share = c / c.sum()
    order = np.argsort(-share)
    return [int(v[k]) for k in order[:max_atoms] if share[k] >= min_share]


# ----------------------------------------------------------------------------- heads

def _T(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _T_inv(u: torch.Tensor) -> torch.Tensor:
    return torch.sign(u) * torch.expm1(torch.abs(u))


class SpikeHead(nn.Module):
    """Atoms + discretised mixture of logistics on sign(x)log(1+|x|), for one integer field."""

    def __init__(self, d: int, atoms: list[int], n_mix: int = 8, lo: int | None = None,
                 hidden: int = 256) -> None:
        super().__init__()
        self.register_buffer("atoms", torch.tensor(atoms, dtype=torch.long), persistent=False)
        self.K, self.M, self.lo = len(atoms), n_mix, lo
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, self.K + 1 + 3 * n_mix))

    def _params(self, x: torch.Tensor):
        o = self.net(x).float()
        K, M = self.K, self.M
        comp = F.log_softmax(o[..., :K + 1], -1)
        mix = F.log_softmax(o[..., K + 1:K + 1 + M], -1)
        mu = o[..., K + 1 + M:K + 1 + 2 * M]
        log_s = o[..., K + 1 + 2 * M:].clamp(-7.0, 3.0)
        return comp, mix, mu, log_s

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        comp, mix, mu, log_s = self._params(x)
        yf = y.float()
        s = log_s.exp()
        hi = (_T(yf + 0.5).unsqueeze(-1) - mu) / s
        lo = (_T(yf - 0.5).unsqueeze(-1) - mu) / s
        at_lo = torch.zeros_like(y, dtype=torch.bool) if self.lo is None else (y <= self.lo)
        log_hi = -F.softplus(-hi)                                # log sigmoid(hi)
        delta = torch.sigmoid(hi) - torch.sigmoid(lo)
        mid = (_T(yf).unsqueeze(-1) - mu) / s
        ay = yf.abs()
        width = torch.where(ay == 0, torch.full_like(ay, 2 * math.log1p(0.5)),
                            torch.log1p(1.0 / (ay + 0.5)))       # bin width in T-space, stable
        log_mid = mid - log_s - 2 * F.softplus(mid) + width.log().unsqueeze(-1)
        lp = torch.where(delta > 1e-5, delta.clamp(min=1e-12).log(), log_mid)
        lp = torch.where(at_lo.unsqueeze(-1), log_hi, lp)        # bin extends to -inf at the bound
        cont = torch.logsumexp(mix + lp, -1) + comp[..., self.K]
        if self.K == 0:
            return cont
        match = y.unsqueeze(-1) == self.atoms
        atom = torch.where(match, comp[..., :self.K], torch.full_like(comp[..., :self.K], -1e30))
        return torch.logsumexp(torch.stack([torch.logsumexp(atom, -1), cont], -1), -1)

    @torch.no_grad()
    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        comp, mix, mu, log_s = self._params(x)
        dev = x.device
        c = torch.distributions.Categorical(logits=comp / temperature).sample().cpu()
        m = torch.distributions.Categorical(logits=mix / temperature).sample()
        # float64 on CPU: MPS has no float64, and dt spans 1 ns .. minutes.
        mu_m = mu.gather(-1, m.unsqueeze(-1)).squeeze(-1).cpu().double()
        s_m = log_s.gather(-1, m.unsqueeze(-1)).squeeze(-1).cpu().double().exp()
        u01 = torch.rand(mu_m.shape, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
        u = (mu_m + s_m * temperature * (u01.log() - (-u01).log1p())).clamp(-30, 30)
        val = torch.round(_T_inv(u))
        if self.lo is not None:
            val = val.clamp(min=self.lo)
        val = val.clamp(-1e12, 1e12).long()
        if self.K:
            atoms = self.atoms.cpu()
            val = torch.where(c < self.K, atoms[c.clamp(max=self.K - 1)], val)
        return val.to(dev)


# ----------------------------------------------------------------------------- model

@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ctx: int = 256
    dropout: float = 0.1
    n_mix: int = 8
    use_atoms: bool = True
    use_book: bool = True          # ablation: False zeroes every book feature in the input
    atom_share: float = 0.005
    max_atoms: int = 32
    atoms: dict = field(default_factory=dict)  # field -> list of atom values (fit on train data)
    encoding: tuple = (0, False)   # action encoding trained on: (pointer ranks, whole_fill)
    regime: bool = False           # day-so-far input features (features.py); False = v1 inputs
    flow: bool = False             # order-flow memory inputs (features.py, v3); needs regime
    market: bool = False           # clock-time own + ETF inputs (market.py, v4); needs flow

    def __post_init__(self) -> None:
        if self.flow and not self.regime:
            raise ValueError("flow features come after the day-so-far ones: set regime=True")
        if self.market and not self.flow:
            raise ValueError("market features come after the flow ones: set flow=True")

    @property
    def n_cont(self) -> int:
        if self.market:
            return N_FULL
        return N_ALL if self.flow else N_CONT if self.regime else N_BASE

    @property
    def size_lo(self) -> int:
        return 0 if self.encoding[1] else 1      # whole_fill: 0 = the whole resting order


def _book_cols(n_cont: int) -> list[int]:
    cols = list(BOOK_COLS)
    if n_cont > N_BASE:  # day-so-far spread and mid-move averages are book information too
        cols += [N_BASE + h * N_REG + k for h in range(len(HALF_LIVES)) for k in REGIME_BOOK]
    if n_cont > N_CONT:  # the mid's distance from its moving average
        cols += FLOW_BOOK
    if n_cont > N_ALL:   # the mid's clock-time moves
        cols += OWN_COLS
    return cols


class LOBModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.e_ts = nn.Embedding(N_TS, d)
        self.e_p = nn.Embedding(P_LO + P_HI + 1, d)
        self.e_q = nn.Embedding(Q_MAX + 1, d)
        self.e_spread = nn.Embedding(SPREAD_MAX + 1, d)
        self.e_cont = nn.Linear(cfg.n_cont, d)
        self._book = _book_cols(cfg.n_cont)
        self.pos = nn.Parameter(torch.zeros(cfg.ctx, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.trunk_layers = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.o_ts = nn.Embedding(N_TS, d)
        self.o_p = nn.Embedding(P_LO + P_HI + 1, d)
        self.o_q = nn.Embedding(Q_MAX + 1, d)
        self.o_size = nn.Embedding(SIZE_BUCKETS, d)
        self.head_ts = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_TS))
        A = cfg.atoms if cfg.use_atoms else {}
        self.head_p = SpikeHead(d, A.get("p", []), cfg.n_mix, lo=None)
        self.head_q = SpikeHead(d, A.get("q", []), cfg.n_mix, lo=0)
        self.head_size = SpikeHead(d, A.get("size", []), cfg.n_mix, lo=cfg.size_lo)
        self.head_dt = SpikeHead(d, A.get("dt", []), cfg.n_mix, lo=0)
        mask = torch.triu(torch.full((cfg.ctx, cfg.ctx), float("-inf")), diagonal=1)
        self.register_buffer("causal", mask, persistent=False)

    def trunk(self, cat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        cont = cont[..., :self.cfg.n_cont]
        if not self.cfg.use_book:
            cont = cont.clone()
            cont[..., self._book] = 0
            cat = cat.clone()
            cat[..., 3] = 0
        T = cat.shape[1]
        x = (self.e_ts(cat[..., 0]) + self.e_p(cat[..., 1]) + self.e_q(cat[..., 2])
             + self.e_spread(cat[..., 3]) + self.e_cont(cont.float()) + self.pos[:T])
        x = self.trunk_layers(x, mask=self.causal[:T, :T], is_causal=True)
        return self.norm(x)

    def _cond(self, h, ts, p, q, size):
        c1 = h + self.o_ts(ts)
        c2 = c1 + self.o_p(torch.clamp(p, -P_LO, P_HI) + P_LO)
        c3 = c2 + self.o_q(torch.clamp(q, 0, Q_MAX))
        c4 = c3 + self.o_size(size_bucket(size))
        return c1, c2, c3, c4

    def nll(self, cat, cont, target) -> dict[str, torch.Tensor]:
        """Mean negative log-likelihood (nats) per field; target = next actions [B, T, 6]."""
        h = self.trunk(cat, cont)
        typ, side = target[..., TYPE], target[..., SIDE]
        ts = (typ.clamp(1, 5) - 1) * 2 + (side > 0).long()
        p, q, size, dt = target[..., PRICE], target[..., QPOS], target[..., SIZE], target[..., DT]
        c1, c2, c3, c4 = self._cond(h, ts, p, q, size)
        out = {"ts": F.cross_entropy(self.head_ts(h).float().flatten(0, 1), ts.flatten())}
        out["p"] = -self.head_p.log_prob(c1, p).mean()
        mq = (typ == EXEC) | (((typ == CANCEL) | (typ == DELETE)) & (p >= 0))
        out["q"] = -(self.head_q.log_prob(c2, q) * mq).sum() / mq.sum().clamp(min=1)
        ms = typ != DELETE
        out["size"] = -(self.head_size.log_prob(c3, size) * ms).sum() / ms.sum().clamp(min=1)
        out["dt"] = -self.head_dt.log_prob(c4, dt).mean()
        # nats per event: masked fields weighted by how often they are scored
        out["event"] = (out["ts"] + out["p"] + out["q"] * mq.float().mean()
                        + out["size"] * ms.float().mean() + out["dt"])
        return out

    @torch.no_grad()
    def sample_next(self, h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """h [B, d] -> actions [B, 6] (dt, type, side, price field, qpos, size)."""
        ts = torch.distributions.Categorical(logits=self.head_ts(h).float() / temperature).sample()
        c1 = h + self.o_ts(ts)
        p = self.head_p.sample(c1, temperature)
        c2 = c1 + self.o_p(torch.clamp(p, -P_LO, P_HI) + P_LO)
        q = self.head_q.sample(c2, temperature)
        typ = ts // 2 + 1
        level_q = (typ == EXEC) | (((typ == CANCEL) | (typ == DELETE)) & (p >= 0))
        q = torch.where(level_q, q, torch.zeros_like(q))
        c3 = c2 + self.o_q(torch.clamp(q, 0, Q_MAX))
        size = self.head_size.sample(c3, temperature)
        c4 = c3 + self.o_size(size_bucket(size))
        dt = self.head_dt.sample(c4, temperature)
        side = torch.where(ts % 2 == 1, 1, -1)
        return torch.stack([dt, typ, side, p, q, size], -1)


def fit_atoms(actions: np.ndarray, share: float, max_atoms: int) -> dict[str, list[int]]:
    typ = actions[:, TYPE]
    return {
        "p": choose_atoms(actions[:, PRICE], share, max_atoms),
        "q": choose_atoms(actions[np.isin(typ, [CANCEL, DELETE, EXEC]), QPOS], share, max_atoms),
        "size": choose_atoms(actions[typ != DELETE, SIZE], share, max_atoms),
        "dt": choose_atoms(actions[:, DT], share, max_atoms),
    }


# ----------------------------------------------------------------------------- data

class EventData:
    """Features + next-action targets for a set of days; samples random ctx-long windows.

    Each day contributes the rows between open+margin and close-margin. Sampling picks a ticker
    uniformly, then a day in proportion to its length, then a uniform start.
    """

    def __init__(self, parts: list[tuple], ctx: int, encoding: tuple = (0, False)) -> None:
        self.ctx = ctx
        self.parts = parts  # (ticker, cat int16, cont float16, actions int64)
        self.tickers = sorted({p[0] for p in parts})
        self.encoding = tuple(encoding)  # action encoding of every part

    @staticmethod
    def featurize(d: Day, margin_s: int = 300, market=None, out_dir=None,
                  chunk: int = 1_000_000) -> tuple:
        """(ticker, cat int16, cont float16, actions int64) for the modeled rows of a day.
        out_dir: write the arrays to .npy files there and return them memory-mapped (read-only),
        so the training set does not have to fit in RAM (all of v4's inputs are ~9 GB). The
        values are the same either way; chunking only bounds the float64 temporaries."""
        rows = np.flatnonzero(d.model_rows(margin_s))
        if len(rows) and rows[-1] - rows[0] + 1 != len(rows):
            raise ValueError(f"{d.ticker} {d.date}: non-modeled rows inside the session")
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        n = r1 - r0
        names = {k: Path(out_dir) / f"{d.ticker}_{d.date}_{k}.npy" for k in ("cat", "cont", "act")} \
            if out_dir is not None else None

        def alloc(k, dtype, shape):  # filled in place: no concatenation copy at the end
            if names is None:
                return np.empty(shape, dtype)
            return np.lib.format.open_memmap(names[k], mode="w+", dtype=dtype, shape=shape)

        cat_out = alloc("cat", np.int16, (n, N_CAT))
        cont_out = alloc("cont", np.float16, (n, N_FULL))
        act_out = alloc("act", d.actions.dtype, (n, d.actions.shape[1]))
        act_out[:] = d.actions[r0:r1]
        # The stream starts at the first row of the day, so the day-so-far features of row r0
        # already include the open; rows before r0 are featurised and dropped.
        st = FeatState()
        for s in range(0, r1, chunk):
            e = min(s + chunk, r1)
            cat, cont, st = token_features(d.actions[s:e], d.msgs[s:e], d.book[s:e], st, market)
            k = max(r0 - s, 0)
            if k < e - s:
                o = s + k - r0
                cat_out[o:o + e - s - k] = cat[k:].astype(np.int16)
                cont_out[o:o + e - s - k] = cont[k:].astype(np.float16)
            del cat, cont
        if names is None:
            return d.ticker, cat_out, cont_out, act_out
        for a in (cat_out, cont_out, act_out):
            a.flush()
        del cat_out, cont_out, act_out
        return (d.ticker, *(np.load(names[k], mmap_mode="r") for k in ("cat", "cont", "act")))

    @classmethod
    def from_npz(cls, paths, ctx: int, margin_s: int = 300, market_dir=None,
                 out_dir=None) -> "EventData":
        """Load, featurise and free one day at a time (a GOOG day's book array is ~3 GB).
        market_dir: also the ETF inputs, from market/<ETF>_<date>.npz files there.
        out_dir: keep the features in memory-mapped files there instead of RAM (featurize)."""
        from .market import Market
        parts, encodings, markets = [], set(), {}
        for p in paths:
            d = Day.load(p)
            mk = None
            if market_dir is not None:
                mk = markets.get(d.date) or markets.setdefault(d.date, Market.load(d.date, market_dir))
            parts.append(cls.featurize(d, margin_s, mk, out_dir))
            encodings.add(d.encoding)
            del d
        if len(encodings) != 1:
            raise ValueError(f"days use different action encodings {encodings}: rerun build_actions.py")
        return cls(parts, ctx, encodings.pop())

    def all_actions(self) -> np.ndarray:
        return np.concatenate([p[3] for p in self.parts])

    def n_events(self) -> int:
        return sum(len(p[1]) for p in self.parts)

    def batch(self, B: int, rng: np.random.Generator, device) -> tuple[torch.Tensor, ...]:
        cats, conts, tgts = [], [], []
        for _ in range(B):
            tk = self.tickers[rng.integers(len(self.tickers))]
            cand = [p for p in self.parts if p[0] == tk]
            w = np.array([len(p[1]) for p in cand], dtype=float)
            _, cat, cont, act = cand[rng.choice(len(cand), p=w / w.sum())]
            s = int(rng.integers(0, len(cat) - self.ctx - 1))
            cats.append(cat[s:s + self.ctx])
            conts.append(cont[s:s + self.ctx])
            tgts.append(act[s + 1:s + 1 + self.ctx])
        f = lambda x, dt: torch.from_numpy(np.stack(x)).to(device=device, dtype=dt)  # noqa: E731
        return f(cats, torch.long), f(conts, torch.float32), f(tgts, torch.long)


# ----------------------------------------------------------------------------- training

def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: LOBModel, data: EventData, n_batches: int, B: int, device, seed: int = 123) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    acc: dict[str, float] = {}
    for _ in range(n_batches):
        out = model.nll(*data.batch(B, rng, device))
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + float(v) / n_batches
    model.train()
    return acc


def train(cfg: ModelConfig, train_data: EventData, val_data: EventData, out_path: str | Path,
          steps: int = 20_000, batch: int = 64, lr: float = 3e-4, warmup: int = 500,
          eval_every: int = 500, eval_batches: int = 20, device=None, seed: int = 0,
          resume: bool = True, log=print) -> dict:
    """Writes <out>.pt (best validation step), <out>_last.pt (final weights) and, at every
    evaluation, <out>_state.pt: everything needed to continue (weights, optimizer, schedule,
    scaler, random states). If that file exists and is from the same run, training continues
    from it, so an interrupted run loses at most `eval_every` steps; on CPU the result is
    bit-identical to an uninterrupted run. The state file is removed when the run finishes."""
    device = pick_device(device)
    out_path = Path(out_path)
    state_path = out_path.with_name(out_path.stem + "_state" + out_path.suffix)
    torch.manual_seed(seed)
    model = LOBModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    rng = np.random.default_rng(seed)
    best, history, start, minutes, val = float("inf"), [], 1, 0.0, None
    run = {"cfg": asdict(cfg), "steps": steps, "batch": batch, "lr": lr, "warmup": warmup,
           "seed": seed, "train_events": train_data.n_events(), "val_events": val_data.n_events()}
    if resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st["run"] != run:
            log(f"{state_path.name} is from a different run (other settings or data): starting over")
        else:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            rng.bit_generator.state = st["rng"]
            torch.set_rng_state(st["torch_rng"])
            if device.type == "cuda" and st["cuda_rng"] is not None \
                    and len(st["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(st["cuda_rng"])
            best, history, val = st["best"], st["history"], st["val"]
            start, minutes = st["step"] + 1, st["minutes"]
            log(f"resumed from {state_path.name} at step {st['step']}")
    t0 = time.time() - 60 * minutes
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params {n_params / 1e6:.2f}M  train events {train_data.n_events():,}  "
        f"val events {val_data.n_events():,}  device {device}")
    for step in range(start, steps + 1):
        cat, cont, tgt = train_data.batch(batch, rng, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss = model.nll(cat, cont, tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss["event"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % eval_every == 0 or step == steps:
            val = evaluate(model, val_data, eval_batches, batch, device)
            rec = {"step": step, "train_event": loss["event"].detach().item(),
                   "minutes": round((time.time() - t0) / 60, 2)}
            rec |= {f"val_{k}": round(v, 4) for k, v in val.items()}
            history.append(rec)
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if val["event"] < best:
                best = val["event"]
                save(model, out_path, {"step": step, "val": val, "history": history})
            _atomic_save({
                "run": run, "step": step, "minutes": (time.time() - t0) / 60, "best": best,
                "val": val, "history": history, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }, state_path)
    # Also keep the final weights: price NLL on the fixed validation batches swings by ~0.15
    # nats between evaluations (a few far-from-touch prices), so "best" can be an early step.
    save(model, out_path.with_name(out_path.stem + "_last" + out_path.suffix),
         {"step": steps, "val": val, "history": history})
    state_path.unlink(missing_ok=True)
    return {"best_val_event": best, "history": history}


def _atomic_save(obj, path: str | Path) -> None:
    """Write to a temporary file, then rename: an interruption never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save(model: LOBModel, path: str | Path, extra: dict) -> None:
    _atomic_save({"cfg": asdict(model.cfg), "state": model.state_dict(), **extra}, path)


def load(path: str | Path, device=None) -> LOBModel:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = LOBModel(ModelConfig(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    return model.to(pick_device(device)).eval()


# ----------------------------------------------------------------------------- generation

def attach_market(g, date: str, market_dir: str = "data/market"):
    """Give a market model (cfg.market) the ETF quotes of `date`; other generators unchanged."""
    if isinstance(g, ModelGenerator) and g.model.cfg.market:
        from .market import Market
        g.market = Market.load(date, market_dir)
    return g


class ModelGenerator(Generator):
    """Batched rollouts: one forward pass per event for a batch of windows, each window with
    its own simulator. Actions the simulator cannot apply (cancel on an empty side) are
    redrawn from the same distribution."""

    def __init__(self, model: LOBModel, name: str = "model", temperature: float = 1.0,
                 batch: int = 250, max_redraws: int = 50, market=None) -> None:
        self.model, self.name, self.temperature = model, name, temperature
        self.batch, self.max_redraws = batch, max_redraws
        self.market = market  # market.Market for windows without their own (market models)

    @classmethod
    def load(cls, path, device=None, name: str = "model", **kw) -> "ModelGenerator":
        return cls(load(path, device), name=name, **kw)

    @torch.no_grad()
    def generate(self, windows: list[Window], n_events: int, seed: int = 0) -> list[Generated]:
        out: list[Generated] = []
        for k in range(0, len(windows), self.batch):
            out += self._chunk(windows[k:k + self.batch], n_events, seed + k)
        return out

    def _chunk(self, ws: list[Window], n_events: int, seed: int) -> list[Generated]:
        model, ctx = self.model, self.model.cfg.ctx
        dev = next(model.parameters()).device
        torch.manual_seed(seed)
        B = len(ws)
        bad = {w.encoding for w in ws} - {tuple(model.cfg.encoding)}
        if bad:
            raise ValueError(f"{self.name} was trained on action encoding {model.cfg.encoding} "
                             f"but the windows use {bad}: rebuild the actions to match")
        cats, conts, states, sims, mks = [], [], [], [], []
        for w in ws:
            if len(w.cond_actions) < ctx:
                raise ValueError(f"window {w.i}: {len(w.cond_actions)} context events < ctx {ctx}")
            if w.feat_state is None:
                raise ValueError(f"window {w.i} has no feature state: make_windows(feat_states=True)")
            mk = (w.market if w.market is not None else self.market) if model.cfg.market else None
            if model.cfg.market and mk is None:
                raise ValueError(f"{self.name} reads the market: set window.market or "
                                 "ModelGenerator.market (market.Market of the day)")
            mks.append(mk)
            # Features of the whole context, continuing the day's stream exactly as in training.
            c, x, st = token_features(w.cond_actions, w.cond_msgs, w.cond_book, w.feat_state, mk)
            cats.append(c[-ctx:])
            conts.append(x[-ctx:])
            states.append(st)
            sims.append(Simulator(SimBook.from_orders(w.state), w.t_prev))
        stream = BatchState(states, mks)
        cat = torch.from_numpy(np.stack(cats)).to(dev)
        cont = torch.from_numpy(np.stack(conts)).to(dev)
        msgs = [[] for _ in range(B)]
        books = [[] for _ in range(B)]
        acts_out = np.empty((B, n_events, 6), dtype=np.int64)
        for step in range(n_events):
            h = model.trunk(cat, cont)[:, -1]
            acts = model.sample_next(h, self.temperature).cpu().numpy()
            applied = np.empty((B, 6), dtype=np.int64)
            snaps = np.empty((B, 4 * sims[0].levels), dtype=np.int64)
            new_msgs = np.empty((B, 6), dtype=np.int64)
            for b in range(B):
                res, tries = sims[b].step(*acts[b].tolist()), 0
                while res is None:
                    tries += 1
                    if tries > self.max_redraws:
                        raise RuntimeError(f"{self.name}: window {ws[b].i} cannot place an action")
                    a = model.sample_next(h[b:b + 1], self.temperature)[0].cpu().numpy()
                    res = sims[b].step(*a.tolist())
                msg, snap = res
                msgs[b].append(msg)
                books[b].append(snap)
                applied[b] = sims[b].last_action
                acts_out[b, step] = applied[b]
                snaps[b] = snap
                new_msgs[b] = msg
            # One vectorised step for the B new rows: each row continues its own window's stream.
            new_c, new_x = stream.step(applied, new_msgs, snaps)
            cat = torch.cat([cat[:, 1:], torch.from_numpy(new_c).to(dev)[:, None]], 1)
            cont = torch.cat([cont[:, 1:], torch.from_numpy(new_x).to(dev)[:, None]], 1)
        return [Generated(np.array(msgs[b], dtype=np.int64), np.array(books[b], dtype=np.int64),
                          sims[b].repairs, sims[b].skipped, acts_out[b]) for b in range(B)]
