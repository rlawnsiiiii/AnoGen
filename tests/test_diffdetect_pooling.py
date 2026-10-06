import numpy as np

from anogen.phases.diffdetect import fuse_scores, pool_bins_np


def test_pooling_localizes_short_anomalies():
    rng = np.random.default_rng(0)
    nom = rng.chisquare(1, (200, 512))
    an = rng.chisquare(1, (200, 512))
    an[:, 100:103] += 30.0  # a 3-bin spike in the per-bin error
    for pool, floor in (("mean", 0.0), ("top16", 0.95), ("max16", 0.95)):
        a, n = pool_bins_np(an, pool), pool_bins_np(nom, pool)
        auc = (a[:, None] > n[None, :]).mean()
        if pool != "mean":
            assert auc > floor
    assert (pool_bins_np(an, "top16")[:, None] > pool_bins_np(nom, "top16")[None, :]).mean() > \
           (pool_bins_np(an, "mean")[:, None] > pool_bins_np(nom, "mean")[None, :]).mean()
    x = np.arange(20.0)[None, :]
    assert np.isclose(pool_bins_np(x, "max4")[0], np.mean([16, 17, 18, 19]))
    assert np.isclose(pool_bins_np(x, "top4")[0], np.mean([16, 17, 18, 19]))


def test_fuse_mean_is_original_and_zmax_standardizes_per_t():
    rng = np.random.default_rng(1)
    ref = np.c_[rng.normal(1, 0.1, 300), rng.normal(10, 2.0, 300)]
    ch = np.zeros(300, dtype=int)
    s = np.array([[1.0, 10.0], [1.5, 10.0], [1.0, 20.0]])
    assert np.allclose(fuse_scores(s, np.zeros(3, int), ref, ch, "mean"), s.mean(1))
    z = fuse_scores(s, np.zeros(3, int), ref, ch, "zmax")
    assert z[1] > 3 and z[2] > 3 and abs(z[0]) < 1
