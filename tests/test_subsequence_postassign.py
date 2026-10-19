"""Post-sampling names for the band-only subsequence pool."""

from __future__ import annotations

import numpy as np

from anogen.shell.morphology import (
    SUBSEQ_GLOBAL,
    SUBSEQ_LOCAL,
    SUBSEQ_PLAIN,
    assign_subsequence_locality,
)


def _nominal(n: int = 40, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    level = np.linspace(0.0, 1.0, n)
    x = np.repeat(level[:, None], width, axis=1)
    return x, np.zeros(n, dtype=np.int64)


def test_split_follows_range_then_context_and_leaves_other_kinds():
    width = 64
    nominal, nch = _nominal(width=width)
    flat = np.full(width, 0.5)
    bump = flat.copy()
    bump[:16] = 0.85
    bump[16:] = 0.15
    bump_b = flat.copy()
    bump_b[:16] = 0.15
    bump_b[16:] = 0.85
    rows = np.stack(
        [
            np.full(width, 3.0),
            np.full(width, -2.0),
            bump,
            bump_b,
            flat,
            flat.copy(),
            np.full(width, 4.0),
        ]
    )
    labels = np.array(
        [SUBSEQ_LOCAL, SUBSEQ_GLOBAL, SUBSEQ_LOCAL, SUBSEQ_GLOBAL, SUBSEQ_LOCAL, SUBSEQ_GLOBAL, "real ESA Point / Global"],
        dtype=object,
    )
    ch = np.zeros(len(rows), dtype=np.int64)
    out = assign_subsequence_locality(
        rows, ch, labels, nominal, nch, segment=16, stride=16
    )
    assert out[-1] == "real ESA Point / Global"
    assert list(labels)[0] == SUBSEQ_LOCAL  # input is not rewritten
    assert list(out[:2]) == [SUBSEQ_GLOBAL, SUBSEQ_GLOBAL]
    assert list(out[2:4]) == [SUBSEQ_LOCAL, SUBSEQ_LOCAL]
    assert list(out[4:6]) == [SUBSEQ_PLAIN, SUBSEQ_PLAIN]


def test_shares_are_equal_across_channels():
    rng = np.random.default_rng(0)
    width = 64
    nominal, _ = _nominal(width=width)
    n_ch, per = 6, 128
    x = rng.normal(0.5, 0.05, size=(n_ch * per, width))
    ch = np.repeat(np.arange(n_ch), per)
    labels = np.array([SUBSEQ_LOCAL, SUBSEQ_GLOBAL] * (n_ch * per // 2), dtype=object)
    nch = np.repeat(np.arange(n_ch), len(nominal))
    nominal_all = np.vstack([nominal + 0.01 * c for c in range(n_ch)])
    out = assign_subsequence_locality(x, ch, labels, nominal_all, nch, segment=16, stride=16)
    counts = {name: int((out == name).sum()) for name in (SUBSEQ_GLOBAL, SUBSEQ_LOCAL, SUBSEQ_PLAIN)}
    assert counts == {SUBSEQ_GLOBAL: 256, SUBSEQ_LOCAL: 256, SUBSEQ_PLAIN: 256}
    # channel 0 keeps the extra window on global and local; channel 1 rotates
    for c, expect in (
        (0, {SUBSEQ_GLOBAL: 43, SUBSEQ_LOCAL: 43, SUBSEQ_PLAIN: 42}),
        (1, {SUBSEQ_GLOBAL: 42, SUBSEQ_LOCAL: 43, SUBSEQ_PLAIN: 43}),
    ):
        got = {name: int(((out == name) & (ch == c)).sum()) for name in expect}
        assert got == expect


def test_empty_pool_is_unchanged():
    x = np.zeros((2, 64))
    labels = np.array(["real level shift", "real ESA Point / Global"], dtype=object)
    nominal, nch = _nominal()
    out = assign_subsequence_locality(x, np.zeros(2, dtype=np.int64), labels, nominal, nch)
    assert list(out) == list(labels)
