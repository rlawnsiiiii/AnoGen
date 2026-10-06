"""train_denoiser must return the schedule *object* (s1 samples with it)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anogen.shell.diffusion import DiffusionSchedule, train_denoiser  # noqa: E402


@pytest.mark.parametrize("kind", ["linear", "geometric"])
def test_train_denoiser_returns_schedule_object(kind):
    rng = np.random.default_rng(0)
    x = (0.5 + 0.1 * rng.standard_normal((16, 32))).astype(np.float32)
    ch = np.zeros(16, dtype=np.int64)
    out = train_denoiser(x, ch, n_channels=1, hidden=8, n_times=20, steps=2, batch_size=8,
                         backbone="tsdiff", n_layers=1, d_state=4, schedule_kind=kind, device="cpu")
    assert isinstance(out["schedule"], DiffusionSchedule)
    assert out["schedule_kind"] == kind


def test_vtoeps_recovers_noise_from_exact_v():
    from anogen.shell.diffusion import VToEps, q_sample, v_target

    sched = DiffusionSchedule.geometric(20)
    x0 = torch.randn(3, 1, 16)
    t = torch.tensor([0, 7, 19])
    noise = torch.randn_like(x0)
    xt, _ = q_sample(x0, t, sched, noise=noise)
    v = v_target(x0, noise, sched.alpha_bar[t].view(-1, 1, 1))

    class Oracle(torch.nn.Module):
        def forward(self, x, tt, c):
            return v

    eps = VToEps(Oracle(), sched.alpha_bar)(xt, t, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(eps, noise, atol=1e-5)


def test_v_parameterization_and_ema_round_trip_through_checkpoint():
    from anogen.shell.diffusion import VToEps, denoiser_from_ckpt

    rng = np.random.default_rng(0)
    x = (0.5 + 0.1 * rng.standard_normal((16, 32))).astype(np.float32)
    ch = np.zeros(16, dtype=np.int64)
    out = train_denoiser(x, ch, n_channels=1, hidden=8, n_times=20, steps=3, batch_size=8, backbone="tsdiff",
                         n_layers=1, d_state=4, schedule_kind="geometric", device="cpu",
                         parameterization="v", ema_decay=0.9)
    assert out["parameterization"] == "v" and isinstance(out["eps_model"], VToEps)
    ckpt = {"state_dict": out["model"].state_dict(), "backbone": "tsdiff", "hidden": 8, "n_layers": 1, "d_state": 4,
            "n_channels": 1, "n_times": 20, "schedule": "geometric", "parameterization": "v"}
    loaded = denoiser_from_ckpt(ckpt)
    xt = torch.randn(2, 1, 32)
    t = torch.tensor([3, 15])
    c = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(loaded(xt, t, c), out["eps_model"](xt, t, c), atol=1e-6)
