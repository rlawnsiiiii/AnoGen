"""fixsweep._generate_slices wiring (no torch: the sampler is a recording stub)."""

import numpy as np

from anogen.phases.fixsweep import _generate_slices, fixsweep_table

KINDS = ("real level shift", "real ESA Point / Global", "other")


def _slice_lam(kind, proto_kinds, default):
    return True, default


def _run(no_steer=False, only=False, masks=True):
    calls = []

    def chunked(model, enc, x, ch, schedule, **kw):
        calls.append(kw)
        return np.asarray(x, dtype=np.float32), np.zeros(len(x))

    n, w = 12, 128
    x0 = np.random.default_rng(0).normal(0.5, 0.05, (n, w))
    alloc = np.array([KINDS[i % 3] for i in range(n)])
    plan = {k: {"kind": c, "amplitudes": np.array([0.1, -0.2]), "lam": 1.0, "only": only, "leaked": False}
            for k, c in ((KINDS[0], "step"), (KINDS[1], "spike"))}
    mask_out = np.zeros((n, w), dtype=bool) if masks else None
    _generate_slices(chunked, None, None, x0, np.zeros(n, int), None,
                     common={"lam": 0.3, "lam_anom": 1.0, "lam_repel": 0.2}, kind_z={k: object() for k in KINDS},
                     alloc=alloc, proto_kinds=None, seed=0, kind_order=KINDS, slice_lam_anom=_slice_lam,
                     prefix=None, proto_shift_max=0, contrast=plan, mask_out=mask_out, mask_dilate=8,
                     no_steer=no_steer)
    return calls, mask_out, alloc


def test_masks_written_for_contrast_kinds_only():
    calls, m, alloc = _run()
    assert m[alloc == KINDS[0]].any(axis=1).all() and m[alloc == KINDS[1]].any(axis=1).all()
    assert not m[alloc == KINDS[2]].any()
    assert all("anomaly_mask" in c for c in calls[:2]) and "anomaly_mask" not in calls[2]


def test_no_steer_turns_everything_off_but_keeps_the_same_masks():
    steered, m1, _ = _run()
    calls, m2, _ = _run(no_steer=True)
    assert np.array_equal(m1, m2)
    for c in calls:
        assert c["mode"] == "off" and c["lam"] == 0.0 and c["lam_anom"] == 0.0 and c["lam_repel"] == 0.0
        assert "contrast_weights" not in c and "ref_anom" not in c


def test_contrast_only_drops_the_prototype_on_contrast_kinds():
    calls, _, _ = _run(only=True)
    assert calls[0]["lam_anom"] == 0.0 and "ref_anom" not in calls[0] and "contrast_weights" in calls[0]
    assert calls[2]["anom_energy_kind"] == "proto"


def test_twin_block_in_table():
    r = {"arp_diff": 0.01, "arp_diff_ci": [-0.01, 0.03]}
    res = {"v": {"folds": [], "paired_vs_twin": {"twin": "t", "frozen_phi": [r], "standardized_phi": [r]}}}
    assert "| v | t | +0.010 | [-0.010, +0.030] |" in fixsweep_table(res)
