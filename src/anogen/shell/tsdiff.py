"""TSDiff-style score: residual S4-D blocks (Kollovieh et al., arXiv:2307.11494).

S4-D kernel follows Gu et al. (diagonal SSM, Vandermonde + FFT conv).
Same call signature as UNet1D: (x, t, channel_idx) -> ε.
"""

from __future__ import annotations

import math
from typing import Any

from anogen.shell.diffusion import _require_torch, sinusoidal_embedding, torch_available

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if torch_available():

    class S4DKernel(nn.Module):
        """Diagonal SSM kernel K_t = C̄ Ā^t, Ā = exp(Δ A)."""

        def __init__(self, d_model: int, d_state: int = 64) -> None:
            super().__init__()
            h, n = d_model, d_state
            log_dt = torch.rand(h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            self.log_dt = nn.Parameter(log_dt)
            c = torch.randn(h, n, dtype=torch.cfloat)
            self.C = nn.Parameter(torch.view_as_real(c))
            self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(h, n)))
            self.register_buffer("A_imag", math.pi * torch.arange(1, n + 1, dtype=torch.float32))

        def forward(self, length: int) -> "torch.Tensor":
            dt = self.log_dt.float().exp()
            c = torch.view_as_complex(self.C.float())
            a = -self.log_A_real.float().exp() + 1j * self.A_imag.float()
            dt_a = a * dt.unsqueeze(-1)
            # ZOH: C̄ = C (e^{ΔA} − 1) / A
            c_bar = c * (torch.exp(dt_a) - 1.0) / a
            steps = torch.arange(length, device=dt.device, dtype=dt.dtype)
            k = (c_bar.unsqueeze(-1) * torch.exp(dt_a.unsqueeze(-1) * steps)).sum(dim=1)
            return k.real

    class S4D(nn.Module):
        def __init__(self, d_model: int, d_state: int = 64) -> None:
            super().__init__()
            self.kernel = S4DKernel(d_model, d_state)
            self.D = nn.Parameter(torch.randn(d_model))
            self.out = nn.Conv1d(d_model, d_model, 1)

        def forward(self, u: "torch.Tensor") -> "torch.Tensor":
            # u: (B, H, L) — FFT in fp32 for stability
            length = u.size(-1)
            orig = u.dtype
            u32 = u.float()
            k = self.kernel(length)
            nfft = 2 * length
            y = torch.fft.irfft(torch.fft.rfft(u32, n=nfft) * torch.fft.rfft(k, n=nfft), n=nfft)
            y = y[..., :length] + u32 * self.D.float().unsqueeze(-1)
            return self.out(y.to(orig))

    class ResidualS4Block(nn.Module):
        def __init__(self, hidden: int, d_state: int, time_dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.s4 = S4D(hidden, d_state)
            self.time = nn.Linear(time_dim, 2 * hidden)
            self.ff = nn.Sequential(
                nn.Conv1d(hidden, hidden, 1),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 1),
            )

        def forward(self, x: "torch.Tensor", temb: "torch.Tensor") -> "torch.Tensor":
            h = self.norm(x.transpose(1, 2)).transpose(1, 2)
            h = self.s4(h)
            scale, shift = self.time(temb).unsqueeze(-1).chunk(2, dim=1)
            h = h * (1.0 + scale) + shift
            return x + self.ff(h)

    class TSDiffBackbone(nn.Module):
        """Residual S4-D ε-predictor. x is (B, 1, W)."""

        def __init__(
            self,
            hidden: int = 64,
            n_channels: int = 6,
            n_layers: int = 6,
            d_state: int = 64,
            time_dim: int = 128,
        ) -> None:
            super().__init__()
            self.time_dim = time_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim),
            )
            self.in_conv = nn.Conv1d(1, hidden, 1)
            self.ch_emb = nn.Embedding(n_channels, hidden)
            self.blocks = nn.ModuleList(
                [ResidualS4Block(hidden, d_state, time_dim) for _ in range(n_layers)]
            )
            self.out_norm = nn.LayerNorm(hidden)
            self.out = nn.Conv1d(hidden, 1, 1)

        def forward(
            self, x: "torch.Tensor", t: "torch.Tensor", channel_idx: "torch.Tensor"
        ) -> "torch.Tensor":
            temb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
            h = self.in_conv(x) + self.ch_emb(channel_idx).unsqueeze(-1)
            for block in self.blocks:
                h = block(h, temb)
            h = self.out_norm(h.transpose(1, 2)).transpose(1, 2)
            return self.out(h)

else:  # pragma: no cover

    class TSDiffBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
