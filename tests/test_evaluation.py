import numpy as np

from anogen.shell.coverage import arp, min_distances
from anogen.shell.diffusion import flip_blend_weights
from anogen.shell.evaluation import (
    arp_coverage_ci,
    calibration_galleries,
    distance_matrix,
    envelope_frequency_report,
    paired_arp_difference,
    peak_positions,
    position_report,
)
from anogen.shell.genias import genias_patch_alg2
from anogen.shell.steer import proto_shift_draws, shift_time_embedding_np


def test_distance_matrix_matches_bruteforce():
    rng = np.random.default_rng(0)
    q, g = rng.normal(size=(7, 12)), rng.normal(size=(1100, 12))
    d = distance_matrix(q, g, block=3)
    brute = np.sqrt(((q[:, None] - g[None]) ** 2).sum(-1))
    assert np.allclose(d, brute, atol=1e-9)
    assert np.allclose(d.min(axis=1), min_distances(q, g))


def test_arp_ci_point_matches_protocol_and_brackets():
    rng = np.random.default_rng(1)
    q, g = rng.normal(size=(60, 12)), rng.normal(size=(200, 12))
    ev = np.repeat(np.arange(12), 5)
    r = arp_coverage_ci(q, g, ev, tau=3.0, n_boot=300, seed=0, resample_gallery=True)
    assert abs(r["arp"] - arp(min_distances(q, g))) < 1e-12
    assert r["arp_ci"][0] <= r["arp"] <= r["arp_ci"][1]
    assert r["n_events"] == 12


def test_cluster_bootstrap_is_wider_than_iid_when_events_cluster():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(300, 4))
    centers = rng.normal(scale=3.0, size=(6, 4))
    q = np.repeat(centers, 20, axis=0) + 0.05 * rng.normal(size=(120, 4))
    clustered = arp_coverage_ci(q, g, np.repeat(np.arange(6), 20), tau=1.0, n_boot=400, seed=0)
    iid = arp_coverage_ci(q, g, np.arange(120), tau=1.0, n_boot=400, seed=0)
    w = lambda r: r["arp_ci"][1] - r["arp_ci"][0]
    assert w(clustered) > 2.0 * w(iid)


def test_paired_difference_of_identical_galleries_is_zero():
    rng = np.random.default_rng(3)
    q, g = rng.normal(size=(30, 5)), rng.normal(size=(80, 5))
    r = paired_arp_difference(q, g, g, np.repeat(np.arange(10), 3), tau=1.0, n_boot=50, resample_gallery=False)
    assert r["arp_diff"] == 0.0 and r["arp_diff_ci"] == [0.0, 0.0]


def test_peak_positions_and_report():
    x = np.zeros((4, 101))
    for i, p in enumerate((3, 50, 97, 60)):
        x[i, p] = 5.0
    assert np.allclose(peak_positions(x), [0.03, 0.5, 0.97, 0.6])
    rep = position_report(x)
    assert rep["peak_at_start"] == 0.25 and rep["peak_at_end"] == 0.25
    rng = np.random.default_rng(0)
    u = rng.normal(size=(2000, 128))
    rep = position_report(u)
    assert abs(rep["peak_at_start"] - 0.1) < 0.03 and rep["position_entropy"] > 0.98


def test_envelope_frequency_report_separates_frequency_from_magnitude():
    real = np.full((100, 10), 0.5)
    real[:4, 3] = 3.0  # 4 % of windows exit, by 2 spans
    gen = np.full((100, 10), 0.5)
    gen[:60, 3] = 3.0
    r = envelope_frequency_report(gen, real)
    assert r["exit_frac"] == 0.6 and abs(r["exit_frac_ratio"] - 15.0) < 1e-9
    assert abs(r["excess_p95_given_exit"] - 2.0) < 1e-9


def test_genias_alg2_replaces_whole_windows():
    parent = np.tile(np.sin(np.linspace(0, 6, 64)), (3, 1))
    gen = parent.copy()
    gen[1] += 0.5  # large deviation everywhere
    gen[2, 10] += 0.05  # tiny deviation
    out, rep = genias_patch_alg2(parent, gen, tau=0.1)
    assert rep.tolist() == [False, True, False]
    assert np.allclose(out[1], gen[1]) and np.allclose(out[2], parent[2])
    w_out, w_rep = genias_patch_alg2(parent * 100, gen * 100, tau=0.1, units="whitened")
    assert w_rep.tolist() == rep.tolist()  # whitened criterion is scale-free


def test_shift_time_embedding_np():
    z = np.arange(2 * 3 * 5, dtype=float).reshape(2, 15)  # C=3, T=5
    out = shift_time_embedding_np(z, np.array([0, 2]), 5).reshape(2, 3, 5)
    assert np.allclose(out[0], z[0].reshape(3, 5))
    assert np.allclose(out[1, 0], [15, 15, 15, 16, 17])  # edge-replicated, moved later


def test_proto_shift_draws_and_flip_weights():
    s = proto_shift_draws(1000, 4, seed=0)
    assert s.min() == -4 and s.max() == 4
    w = flip_blend_weights(5, "ramp")
    assert np.allclose(w, [0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(flip_blend_weights(3, "avg"), 0.5)


def test_calibration_galleries_are_out_of_fold():
    xa = np.arange(10)[:, None] * np.ones((1, 8))
    fold = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    g = calibration_galleries(xa, fold, np.ones((5, 8)), fold_id=0, rng=np.random.default_rng(0))
    assert set(g["oracle_train_fold_anomalies"][:, 0].astype(int)) == {2, 3, 4, 5, 7, 8}


def test_prdc_separates_fidelity_from_spread():
    from anogen.shell.evaluation import prdc

    rng = np.random.default_rng(0)
    real = rng.normal(size=(300, 4))
    same = prdc(real, rng.normal(size=(300, 4)))
    wide = prdc(real, rng.normal(scale=4.0, size=(300, 4)))
    collapsed = prdc(real, rng.normal(scale=0.05, size=(300, 4)))
    assert same["precision"] > 0.8 and same["coverage"] > 0.8
    assert wide["precision"] < 0.4 and wide["density"] < same["density"]
    assert collapsed["precision"] > 0.9 and collapsed["coverage"] < 0.3


def test_feature_attribution_sums_to_one():
    from anogen.shell.evaluation import PHI_NAMES, feature_attribution

    rng = np.random.default_rng(1)
    q, g = rng.normal(size=(20, 12)), rng.normal(size=(50, 12))
    g[:, 11] += 10.0
    a = feature_attribution(q, g, PHI_NAMES)
    assert abs(sum(a.values()) - 1.0) < 1e-9 and max(a, key=a.get) == "fft_band4"


def test_paired_event_bootstrap():
    from anogen.shell.detector import paired_event_bootstrap

    ev = [f"e{i}" for i in range(10)]
    a = {s: {e: (i < 7) for i, e in enumerate(ev)} for s in range(3)}
    b = {s: {e: (i < 5) for i, e in enumerate(ev)} for s in range(3)}
    r = paired_event_bootstrap(a, b, n_boot=500)
    assert abs(r["diff"] - 0.2) < 1e-12 and r["lo"] >= 0.0 and r["p_le_zero"] < 0.2
    same = paired_event_bootstrap(a, a, n_boot=100)
    assert same["diff"] == 0.0 and same["p_le_zero"] == 1.0


def test_eval_scores_and_rank_fuse():
    from anogen.shell.detector import eval_scores, rank_fuse

    rng = np.random.default_rng(0)
    s_n = rng.normal(size=1000)
    s_a = np.concatenate([rng.normal(5, 1, 10), rng.normal(0, 1, 10)])
    ev = np.repeat(["a", "b"], 10)
    r = eval_scores(s_a, s_n, np.zeros(0), event_id=ev, kind=np.array(["k"] * 20), far=0.01)
    assert r["event_hits"]["a"] and 0.005 <= r["nominal_far"] <= 0.02 and r["auroc"] > 0.6
    f = rank_fuse(np.array([0.0, 10.0]), np.array([100.0, -100.0]), reference=[s_n, s_n * 100])
    assert f.shape == (2,) and 0.0 <= f.min() <= f.max() <= 1.0
