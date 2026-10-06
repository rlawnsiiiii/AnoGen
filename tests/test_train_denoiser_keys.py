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
