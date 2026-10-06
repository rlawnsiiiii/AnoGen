"""augdetect arms that read fixsweep galleries (numpy only)."""

import numpy as np

from anogen.phases.augdetect import _arm_leaks_fold0, _is_plus_real, _synth_for_arm


def test_fixsweep_arm_reads_variant_and_keeps_donor_disjointness(tmp_path):
    n, w = 30, 16
    cond_ch = np.arange(n) % 2
    x = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, w))  # row i = donor i
    np.savez_compressed(tmp_path / "flip_x0_contrast_masked_fold1.npz", x=x, channel_idx=cond_ch)
    rng = np.random.default_rng(0)
    xs, ch = _synth_for_arm("fixsweep:flip_x0_contrast_masked+real", fold_id=1, rng=rng, n_synth=100, galleries={},
                            hybrid=tmp_path, enc_score=tmp_path, cond_ch=cond_ch, exclude_donor_fold=1,
                            fixsweep_dir=tmp_path)
    donors = xs[:, 0].astype(int)
    assert len(xs) == 20 and np.all(donors % 3 != 1)  # test-fold donors never become positives
    assert np.array_equal(ch, cond_ch[donors])
    assert _is_plus_real("fixsweep:a+real") and not _is_plus_real("fixsweep:a")
    assert _arm_leaks_fold0("fixsweep:a")


def test_missing_fixsweep_gallery_is_a_file_not_found(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        _synth_for_arm("fixsweep:nope", fold_id=0, rng=np.random.default_rng(0), n_synth=5, galleries={},
                       hybrid=tmp_path, enc_score=tmp_path, cond_ch=np.zeros(3, int), fixsweep_dir=tmp_path)


def test_plus_real_suffix_works_for_any_gallery_arm(tmp_path):
    n, w = 12, 8
    cond_ch = np.zeros(n, int)
    gal = {"genias": (np.tile(np.arange(n, dtype=np.float32)[:, None], (1, w)), cond_ch)}
    xs, _ = _synth_for_arm("genias+real", fold_id=0, rng=np.random.default_rng(0), n_synth=50, galleries=gal,
                           hybrid=tmp_path, enc_score=tmp_path, cond_ch=cond_ch, exclude_donor_fold=0)
    assert len(xs) == 8 and np.all(xs[:, 0].astype(int) % 3 != 0)
    assert _is_plus_real("genias+real")


def test_twin_pairs_and_paired_contrasts_drop_leaked_folds():
    from anogen.phases.augdetect import _aggregate, _auto_twin_pairs, _compare_pairs

    variants = {"steer": {"twin": "plain"}, "plain": {}}
    arms = ("real_only", "fixsweep:steer+real", "fixsweep:plain+real", "fixsweep:steer")
    assert _auto_twin_pairs(arms, variants) == [("fixsweep:steer+real", "fixsweep:plain+real")]

    def row(arm, fold, seed, hits, leaked):
        return {"arm": arm, "fold": fold, "seed": seed, "ok": True, "leaked": leaked, "event_hits": hits,
                "window_recall": 0.0, "ap": 0.0, "auroc": 0.5, "rare_far": 0.0, "nominal_far": 0.0,
                "per_kind_recall": {}}

    rows = []
    for seed in (0, 1):
        # fold 0 (leaked for fixsweep arms): steered hits e0, twin misses it
        rows += [row("fixsweep:steer+real", 0, seed, {"e0": True}, True),
                 row("fixsweep:plain+real", 0, seed, {"e0": False}, True),
                 row("real_only", 0, seed, {"e0": False}, False)]
        # fold 1: both hit e1, neither hits e2
        rows += [row(a, 1, seed, {"e1": True, "e2": False}, False)
                 for a in ("fixsweep:steer+real", "fixsweep:plain+real", "real_only")]
    pairs = _compare_pairs(rows, [("fixsweep:steer+real", "fixsweep:plain+real")], n_boot=200)
    res = pairs["fixsweep:steer+real - fixsweep:plain+real"]
    assert res["all"]["n_events"] == 3 and abs(res["all"]["diff"] - 1 / 3) < 1e-9
    assert res["noleak"]["n_events"] == 2 and res["noleak"]["diff"] == 0.0  # the leaked win is gone
    summ = _aggregate(rows, arms=("real_only", "fixsweep:steer+real"), seeds=[0, 1], folds=[0, 1])
    assert abs(summ["fixsweep:steer+real"]["vs_real_only"]["diff"] - 1 / 3) < 1e-9
    assert summ["fixsweep:steer+real"]["vs_real_only_noleak"]["diff"] == 0.0


def test_synth_filter_drops_nominal_like_positives_and_caches_per_key():
    from anogen.phases.augdetect import _filter_synth
    from anogen.shell.novelty import KnnNovelty

    class Identity:
        def transform(self, x, ch):
            return np.asarray(x, dtype=np.float64)

    rng = np.random.default_rng(1)
    t = np.arange(96)

    def nominal(n):
        return 0.5 + 0.1 * np.sin(2 * np.pi * t / 30 + rng.uniform(0, 6.3, (n, 1))) + 0.01 * rng.standard_normal((n, 96))

    syn = nominal(40)
    syn[:20, 40:44] += 0.7  # half real anomalies, half regenerated nominals
    cache = {}
    kw = dict(synth_filter=KnnNovelty(n_kernels=80, n_components=12), scaler=Identity(), x_ref=nominal(250),
              ch_ref=np.zeros(250, int), x_cal=nominal(150), ch_cal=np.zeros(150, int), cache=cache, key=(0, 0))
    xs, ch, rate = _filter_synth(syn, np.arange(40), **kw)
    assert set(ch.tolist()) >= set(range(18)) and len(set(ch.tolist()) & set(range(20, 40))) <= 2
    assert abs(rate - len(xs) / 40) < 1e-12 and (0, 0) in cache
    fitted = cache[(0, 0)]
    _filter_synth(syn, np.arange(40), **kw)
    assert cache[(0, 0)] is fitted  # reused, not refitted


def test_return_donor_tracks_s3_rows_and_plus_real_keeps_the_leak_flag(tmp_path):
    n, w = 30, 16
    cond_ch = np.arange(n) % 2
    x = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, w))
    np.savez_compressed(tmp_path / "v_fold2.npz", x=x, channel_idx=cond_ch)
    xs, ch, donor = _synth_for_arm("fixsweep:v+real", fold_id=2, rng=np.random.default_rng(0), n_synth=7,
                                   galleries={}, hybrid=tmp_path, enc_score=tmp_path, cond_ch=cond_ch,
                                   exclude_donor_fold=2, fixsweep_dir=tmp_path, return_donor=True)
    assert len(xs) == 7 and np.array_equal(donor, xs[:, 0].astype(int)) and np.all(donor % 3 != 2)
    # the same rng use as without donors: identical rows
    xs2, _ = _synth_for_arm("fixsweep:v+real", fold_id=2, rng=np.random.default_rng(0), n_synth=7, galleries={},
                            hybrid=tmp_path, enc_score=tmp_path, cond_ch=cond_ch, exclude_donor_fold=2,
                            fixsweep_dir=tmp_path)
    assert np.array_equal(xs, xs2)
    assert _arm_leaks_fold0("genfsdiff_c1+real") and _arm_leaks_fold0("c1_plus_real")
    assert not _arm_leaks_fold0("genias+real")


def test_geniasfair_arm_reads_the_fair_gallery_donor_disjoint(tmp_path):
    import pytest

    n, w = 24, 8
    cond_ch = np.arange(n) % 3
    x = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, w))
    np.savez_compressed(tmp_path / "galleries.npz", channel_idx=cond_ch, cond=x, genias_fair=x + 0.5)
    xs, ch = _synth_for_arm("geniasfair:genias_fair+real", fold_id=0, rng=np.random.default_rng(0), n_synth=100,
                            galleries={}, hybrid=tmp_path, enc_score=tmp_path, cond_ch=cond_ch,
                            exclude_donor_fold=0, geniasfair_dir=tmp_path)
    donors = (xs[:, 0] - 0.5).astype(int)
    assert len(xs) == 16 and np.all(donors % 3 != 0) and np.array_equal(ch, cond_ch[donors])
    with pytest.raises(FileNotFoundError):
        _synth_for_arm("geniasfair:nope", fold_id=0, rng=np.random.default_rng(0), n_synth=5, galleries={},
                       hybrid=tmp_path, enc_score=tmp_path, cond_ch=cond_ch, geniasfair_dir=tmp_path)
