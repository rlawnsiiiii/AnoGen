"""A tiny numpy dilated-CNN ε-network with manual backprop.

Exists so the testbed can check, with a *trained nonlinear* denoiser, what
the exact Gaussian estimators predict: a causal backbone puts artefacts at
the window start, a bidirectional one with the same parameter count does
not. Causal vs "same" padding is the only difference between the two
variants. Gradient-checked in tests/test_testbed.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_grad(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


class Conv1d:
    """(B, Cin, L) -> (B, Cout, L); causal (left pad) or same (centred) padding."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int, causal: bool, rng: np.random.Generator):
        self.k, self.d, self.causal = int(k), int(dilation), bool(causal)
        scale = np.sqrt(2.0 / (cin * k))
        self.W = rng.normal(scale=scale, size=(cout, cin, k))
        self.b = np.zeros(cout)
        self.cache: tuple | None = None

    def _pads(self) -> tuple[int, int]:
        tot = (self.k - 1) * self.d
        return (tot, 0) if self.causal else (tot // 2, tot - tot // 2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        lp, rp = self._pads()
        xp = np.pad(x, ((0, 0), (0, 0), (lp, rp)))
        B, C, L = x.shape
        # taps: (B, Cin*k, L), row index c*k + j
        taps = np.stack([xp[:, :, j * self.d : j * self.d + L] for j in range(self.k)], axis=2).reshape(B, C * self.k, L)
        self.cache = (taps, x.shape)
        wr = self.W.reshape(self.W.shape[0], -1)
        return np.matmul(wr, taps) + self.b[None, :, None]

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        taps, shape = self.cache  # type: ignore[misc]
        B, C, L = shape
        gW = np.matmul(gy, taps.transpose(0, 2, 1)).sum(axis=0).reshape(self.W.shape)
        gb = gy.sum(axis=(0, 2))
        wr = self.W.reshape(self.W.shape[0], -1)
        gtaps = np.matmul(wr.T, gy).reshape(B, C, self.k, L)
        lp, rp = self._pads()
        gxp = np.zeros((B, C, L + lp + rp))
        for j in range(self.k):
            gxp[:, :, j * self.d : j * self.d + L] += gtaps[:, :, j, :]
        return gxp[:, :, lp : lp + L], gW, gb


def time_features(t: np.ndarray, dim: int = 16, n_times: int = 200) -> np.ndarray:
    half = dim // 2
    freqs = np.exp(-np.log(1000.0) * np.arange(half) / half)
    a = np.asarray(t, dtype=np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(a), np.cos(a)], axis=1)


@dataclass
class TinyEpsNet:
    """in-conv -> residual dilated blocks (+ time shift) -> 1x1 out."""

    hidden: int = 32
    k: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    causal: bool = True
    seed: int = 0
    tdim: int = 16
    params: dict = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.inp = Conv1d(1, self.hidden, self.k, 1, self.causal, rng)
        self.blocks = [Conv1d(self.hidden, self.hidden, self.k, d, self.causal, rng) for d in self.dilations]
        self.Wt = [rng.normal(scale=0.1, size=(self.tdim, self.hidden)) for _ in self.dilations]
        self.out = Conv1d(self.hidden, 1, 1, 1, self.causal, rng)
        self.out.W *= 0.1
        self.adam: dict = {}

    def layers(self) -> list:
        return [self.inp, *self.blocks, self.out]

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """x: (B, L) -> eps_hat (B, L)."""
        tf = time_features(t, self.tdim)
        self._tf = tf
        h = self.inp.forward(x[:, None, :])
        self._pre = []
        self._h_in = []
        for blk, wt in zip(self.blocks, self.Wt, strict=True):
            self._h_in.append(h)
            pre = blk.forward(h) + (tf @ wt)[:, :, None]
            self._pre.append(pre)
            h = h + silu(pre)
        return self.out.forward(h)[:, 0, :]

    def backward(self, g_out: np.ndarray) -> tuple[np.ndarray, dict]:
        """Returns (d loss / d x, parameter grads) for upstream grad (B, L)."""
        grads: dict = {}
        gh, gW, gb = self.out.backward(g_out[:, None, :])
        grads["out"] = (gW, gb)
        for i in reversed(range(len(self.blocks))):
            gpre = gh * silu_grad(self._pre[i])
            grads[f"wt{i}"] = self._tf.T @ gpre.sum(axis=2)
            gx_blk, gW, gb = self.blocks[i].backward(gpre)
            grads[f"blk{i}"] = (gW, gb)
            gh = gh + gx_blk
        gx, gW, gb = self.inp.backward(gh)
        grads["inp"] = (gW, gb)
        return gx[:, 0, :], grads

    def step(self, grads: dict, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999) -> None:
        self.adam["t"] = self.adam.get("t", 0) + 1
        tt = self.adam["t"]
        pairs = [("inp", self.inp), ("out", self.out)] + [(f"blk{i}", b) for i, b in enumerate(self.blocks)]
        targets: list[tuple[str, object, str]] = []
        for name, layer in pairs:
            targets += [(name + ".W", layer, "W"), (name + ".b", layer, "b")]
        for i in range(len(self.Wt)):
            targets.append((f"wt{i}", self.Wt, i))
        for key, obj, attr in targets:
            base = key.split(".")[0]
            g = grads[base][0 if key.endswith(".W") else 1] if base in ("inp", "out") or base.startswith("blk") else grads[base]
            p = getattr(obj, attr) if isinstance(attr, str) else obj[attr]
            m, v = self.adam.get(key + "m", np.zeros_like(p)), self.adam.get(key + "v", np.zeros_like(p))
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            self.adam[key + "m"], self.adam[key + "v"] = m, v
            upd = lr * (m / (1 - b1**tt)) / (np.sqrt(v / (1 - b2**tt)) + 1e-8)
            if isinstance(attr, str):
                setattr(obj, attr, p - upd)
            else:
                obj[attr] = p - upd


class NetDenoiser:
    """Adapter so ``gauss.guided_ddim`` can sample with a TinyEpsNet."""

    def __init__(self, net: TinyEpsNet, schedule, length: int, mean: float = 0.0):
        self.net, self.schedule, self.length, self.mean = net, schedule, int(length), float(mean)
        self._last: tuple | None = None

    def x0_hat(self, xt: np.ndarray, t: int) -> np.ndarray:
        ab = float(self.schedule.alpha_bar[int(t)])
        eps = self.net.forward(xt - np.sqrt(ab) * self.mean, np.full(len(xt), int(t)))
        self._last = (int(t), ab)
        return (xt - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab)

    def vjp(self, g_x0: np.ndarray, t: int) -> np.ndarray:
        """d f / d x_t for the *last* x0_hat call at this t."""
        ab = float(self.schedule.alpha_bar[int(t)])
        # x0 = (x - sqrt(1-ab) eps(x)) / sqrt(ab)  ->  J^T g = (g - sqrt(1-ab) Jeps^T g) / sqrt(ab)
        g_eps_in, _ = self.net.backward(g_x0)
        return (g_x0 - np.sqrt(1.0 - ab) * g_eps_in) / np.sqrt(ab)


def train_eps_net(
    net: TinyEpsNet,
    sample_fn,
    schedule,
    *,
    steps: int = 2000,
    batch: int = 64,
    lr: float = 2e-3,
    mean: float = 0.0,
    seed: int = 0,
    log_every: int = 200,
) -> list[float]:
    """ε-MSE training on fresh draws from ``sample_fn(n, rng) -> (n, L)``."""
    rng = np.random.default_rng(seed)
    hist = []
    n_times = len(schedule.alpha_bar)
    for s in range(1, steps + 1):
        x0 = sample_fn(batch, rng) - mean
        t = rng.integers(0, n_times, batch)
        ab = schedule.alpha_bar[t][:, None]
        eps = rng.standard_normal(x0.shape)
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        pred = net.forward(xt, t)
        g = 2.0 * (pred - eps) / pred.size
        _, grads = net.backward(g)
        lr_s = lr * min(1.0, s / 100) * (0.5 * (1 + np.cos(np.pi * s / steps)) * 0.9 + 0.1)
        net.step(grads, lr=lr_s)
        hist.append(float(np.mean((pred - eps) ** 2)))
        if log_every and s % log_every == 0:
            print(f"  step {s}: eps-mse {np.mean(hist[-log_every:]):.4f}", flush=True)
    return hist
