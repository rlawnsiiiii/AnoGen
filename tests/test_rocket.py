import numpy as np

from anogen.shell.rocket import RocketSpace, RocketTransform, c2st_auc, rocket_c2st
from anogen.testbed.gauss import GPChannel


def test_transform_shape_determinism_and_conv():
    x = GPChannel().sample(20, 128, np.random.default_rng(0))
    a = RocketTransform(n_kernels=16, seed=3).fit(x).transform(x)
    b = RocketTransform(n_kernels=16, seed=3).fit(x).transform(x)
    assert a.shape == (20, 32) and np.array_equal(a, b)
    assert np.all((a[:, ::2] >= 0) & (a[:, ::2] <= 1))  # ppv is a proportion


def test_c2st_null_near_half_and_detects_amplitude_change():
    ch, r = GPChannel(), np.random.default_rng(0)
    sp = RocketSpace(n_kernels=100, n_components=16).fit(ch.sample(200, 256, r))
    a, b = ch.sample(240, 256, r), ch.sample(240, 256, r)
    g = np.arange(240) // 4
    null = c2st_auc(sp.features(a), sp.features(b), g, g)["auc"]
    alt = c2st_auc(sp.features(a), sp.features(0.5 + 1.5 * (b - 0.5)), g, g)["auc"]
    assert abs(null - 0.5) < 0.12 and alt > 0.9
    assert sp.embed(a).shape == (240, 16)


def test_rocket_c2st_balances_channels():
    ch, r = GPChannel(), np.random.default_rng(1)
    sp = RocketSpace(n_kernels=60, n_components=8).fit(ch.sample(100, 128, r))
    real, syn = ch.sample(80, 128, r), ch.sample(200, 128, r)
    rc = np.repeat([0, 1], 40)
    sc = np.r_[np.zeros(20, int), np.ones(180, int)]
    out = rocket_c2st(sp, real, syn, rc, sc, np.arange(80) // 2, np.arange(200), seeds=(0, 1))
    assert 0.3 < out["auc"] < 0.7 and out["n_seeds"] == 2
