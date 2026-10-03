"""Refactored guided_ddim must reproduce the locked sampler bit for bit by default."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anogen.shell.diffusion import DiffusionSchedule, FlipEnsemble, build_denoiser, wrap_denoiser  # noqa: E402
from anogen.shell.encoders import ShellEncoder  # noqa: E402
from anogen.shell.scaler import ChannelMinMax  # noqa: E402
from anogen.shell.steer import (  # noqa: E402
    chunked_guided_ddim,
    guided_ddim,
    shift_time_embedding,
    shift_time_embedding_np,
)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _legacy_guided_ddim import legacy_guided_ddim  # noqa: E402

W = 64


def _setup(bidirectional=False):
    torch.manual_seed(0)
    model = build_denoiser(backbone="tsdiff", hidden=8, n_channels=2, n_layers=2, d_state=4, bidirectional=bidirectional)
    enc = ShellEncoder(W, hidden=4, emb=8, time_emb=2, pool="time")
    sched = DiffusionSchedule.linear(50)
    rng = np.random.default_rng(0)
    x0 = (0.5 + 0.1 * rng.standard_normal((6, W))).astype(np.float32)
    ch = np.array([0, 1, 0, 1, 0, 1])
    with torch.no_grad():
        ref = enc.encode(torch.from_numpy(x0).unsqueeze(1)).detach() + 0.1
    scaler = ChannelMinMax(lo=np.array([0.0, 0.1]), hi=np.array([1.0, 0.9]))
    return model, enc, sched, x0, ch, ref, scaler


CASES = [
    dict(lam=0.3, normalize_grad=True),
    dict(lam=0.3, normalize_grad=False, n_correct=2),
    dict(lam=0.3, normalize_grad=True, lam_anom=1.0, anom_energy_kind="proto", anom_proto_index=torch.tensor([0, 1, 2, 3, 4, 5])),
    dict(lam=0.3, normalize_grad=True, lam_anom=1.0, lam_rare=0.5),
    dict(lam=0.3, normalize_grad=True, lam_parent=0.5, parent_leash="gm"),
    dict(lam=0.3, normalize_grad=True, apply_final_grad=False),
    dict(lam=0.0, mode="off", normalize_grad=True, lam_anom=1.0),
    dict(lam=0.3, normalize_grad=True, start_from_noise=True),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_defaults_reproduce_legacy(case):
    model, enc, sched, x0, ch, ref, scaler = _setup()
    kw = dict(CASES[case])
    common = dict(ref=ref, Q_q=0.5, tau=1.0, nu=0.3, c_max=1.0, ddim_steps=12, scaler=scaler, device="cpu")
    if "lam_anom" in kw:
        kw["ref_anom"] = ref[:6]
    if "lam_rare" in kw:
        kw["ref_rare"] = ref[3:]
    torch.manual_seed(11)
    a, ha = legacy_guided_ddim(model, enc, x0, ch, sched, **common, **kw)
    torch.manual_seed(11)
    b, hb = guided_ddim(model, enc, x0, ch, sched, **common, **kw)
    assert np.array_equal(a, b)
    assert np.array_equal(ha, hb)


def test_final_grad_scale_generalizes_apply_final_grad():
    model, enc, sched, x0, ch, ref, scaler = _setup()
    common = dict(ref=ref, Q_q=0.5, tau=1.0, nu=0.3, lam=0.3, c_max=1.0, ddim_steps=8, normalize_grad=True, device="cpu")
    torch.manual_seed(3)
    off, _ = guided_ddim(model, enc, x0, ch, sched, apply_final_grad=False, **common)
    torch.manual_seed(3)
    zero, _ = guided_ddim(model, enc, x0, ch, sched, final_grad_scale=0.0, **common)
    torch.manual_seed(3)
    half, _ = guided_ddim(model, enc, x0, ch, sched, final_grad_scale=0.5, **common)
    torch.manual_seed(3)
    full, _ = guided_ddim(model, enc, x0, ch, sched, final_grad_scale=1.0, **common)
    assert np.array_equal(off, zero)
    assert np.allclose(half, 0.5 * (zero + full), atol=1e-6)


def test_x0_space_burnin_and_shift_run():
    model, enc, sched, x0, ch, ref, scaler = _setup()
    pre = (0.5 + 0.1 * np.random.default_rng(1).standard_normal((6, 16))).astype(np.float32)
    out, h = chunked_guided_ddim(
        model, enc, x0, ch, sched, bsz=4, ref=ref, Q_q=0.5, tau=1.0, nu=0.3, lam=0.3, c_max=1.0,
        ddim_steps=8, normalize_grad=True, scaler=scaler, device="cpu", guidance_space="x0",
        burnin_prefix=pre, ref_anom=ref[:3], lam_anom=1.0, anom_energy_kind="proto",
        anom_proto_seed=0, anom_proto_shift_max=3, anom_proto_shift_seed=1, lam_repel=0.2,
    )
    assert out.shape == x0.shape and np.isfinite(out).all() and h.shape == (6,)


def test_shift_time_embedding_matches_numpy_twin():
    z = torch.randn(5, 2 * 8)
    s = torch.tensor([0, 1, -2, 7, -9])
    assert np.allclose(shift_time_embedding(z, s, 8).numpy(), shift_time_embedding_np(z.numpy(), s.numpy(), 8))


def test_causal_and_bidirectional_backbones():
    for bidir in (False, True):
        model, *_ = _setup(bidirectional=bidir)
        model.eval()
        x = torch.randn(2, 1, W)
        t = torch.full((2,), 10, dtype=torch.long)
        c = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            base = model(x, t, c)
            x2 = x.clone()
            x2[..., W // 2] += 1.0
            d = (model(x2, t, c) - base).abs().squeeze(1)
        before = float(d[:, : W // 2].max())
        if bidir:
            assert before > 1e-6
        else:
            # float32 FFT: exact zero in exact arithmetic, ~1e-8 in practice
            assert before < 1e-5 * max(float(d.max()), 1.0)


def test_flip_ensemble_sees_both_sides_and_skips_bidirectional():
    model, *_ = _setup()
    wrapped = wrap_denoiser(model, "ramp")
    assert isinstance(wrapped, FlipEnsemble)
    x = torch.randn(1, 1, W)
    t = torch.full((1,), 10, dtype=torch.long)
    c = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        base = wrapped(x, t, c)
        x2 = x.clone()
        x2[..., W // 2] += 1.0
        d = (wrapped(x2, t, c) - base).abs().squeeze()
    assert float(d[: W // 2].max()) > 1e-6
    bidir, *_ = _setup(bidirectional=True)
    assert wrap_denoiser(bidir, "ramp") is bidir
    assert wrap_denoiser(model, None) is model
