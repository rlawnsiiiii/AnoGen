import numpy as np

from anogen.testbed.gauss import (
    BlockEncoder,
    GPChannel,
    GuideTerm,
    LinearDenoiser,
    Schedule,
    guided_ddim,
    proto_energy,
)
from anogen.testbed.metrics import gallery_report


def _brute_causal(sigma, ab):
    n = len(sigma)
    sy = ab * sigma + (1 - ab) * np.eye(n)
    a = np.zeros((n, n))
    for i in range(n):
        a[i, : i + 1] = np.linalg.solve(sy[: i + 1, : i + 1], np.sqrt(ab) * sigma[i, : i + 1])
    return a


def test_causal_denoiser_is_exact_and_causal():
    ch, sch = GPChannel(), Schedule()
    for t in (0, 40, 199):
        den = LinearDenoiser(ch, 20, sch, "causal")
        a = den.matrix(t)
        assert np.allclose(a, _brute_causal(ch.cov(20), sch.alpha_bar[t]), atol=1e-10)
        assert np.abs(np.triu(a, 1)).max() == 0.0


def test_bidirectional_denoiser_matches_closed_form():
    ch, sch = GPChannel(), Schedule()
    sigma = ch.cov(16)
    ab = sch.alpha_bar[40]
    want = np.sqrt(ab) * sigma @ np.linalg.inv(ab * sigma + (1 - ab) * np.eye(16))
    assert np.allclose(LinearDenoiser(ch, 16, sch, "bidir").matrix(40), want, atol=1e-10)


def test_anticausal_is_flip_of_causal():
    ch, sch = GPChannel(), Schedule()
    c = LinearDenoiser(ch, 12, sch, "causal").matrix(10)
    a = LinearDenoiser(ch, 12, sch, "anticausal").matrix(10)
    assert np.abs(np.tril(a, -1)).max() == 0.0
    assert np.allclose(a, c[::-1, ::-1])


def test_causal_vjp_smears_left():
    ch, sch = GPChannel(), Schedule()
    w = 128
    g = np.zeros((1, w))
    g[0, w // 2] = 1.0
    for kind, lo, hi in (("bidir", 0.45, 0.55), ("causal", 0.0, 0.45)):
        v = np.abs(LinearDenoiser(ch, w, sch, kind).vjp(g, 40)[0])
        centroid = (v * np.arange(w)).sum() / v.sum() / w
        assert lo <= centroid <= hi, (kind, centroid)


def test_unguided_causal_sampler_has_start_bias():
    ch, sch = GPChannel(), Schedule()
    rng = np.random.default_rng(0)
    w, n = 256, 256
    donor = ch.sample(n, w, rng)
    lo, hi = float(donor.min()) - 0.5, float(donor.max()) + 0.5
    out = {}
    for kind in ("bidir", "causal"):
        x = guided_ddim(LinearDenoiser(ch, w, sch, kind), donor, [], rng=np.random.default_rng(1), nu=1.0, ddim_steps=20)
        out[kind] = gallery_report(x, lo, hi)
    assert out["causal"]["peak_at_start"] > 2.5 * out["bidir"]["peak_at_start"]
    assert out["causal"]["peak_at_end"] < out["bidir"]["peak_at_end"]


def test_final_scale_only_changes_the_last_step():
    ch, sch = GPChannel(), Schedule()
    w, n = 64, 8
    rng = np.random.default_rng(0)
    donor = ch.sample(n, w, rng)
    enc = BlockEncoder(w, 8)
    proto = enc.encode(ch.sample(1, w, rng) + 0.3)
    term = [GuideTerm(1.0, lambda x: proto_energy(enc, x, np.repeat(proto, n, 0)))]
    den = LinearDenoiser(ch, w, sch, "causal")
    a = guided_ddim(den, donor, term, rng=np.random.default_rng(3), final_scale=1.0, ddim_steps=10)
    b = guided_ddim(den, donor, term, rng=np.random.default_rng(3), final_scale=0.0, ddim_steps=10)
    diff = np.linalg.norm(a - b, axis=1)
    # unit-normalized, clipped kick: per-sample L2 norm is exactly lam = 1
    assert np.allclose(diff, 1.0, atol=1e-6)
