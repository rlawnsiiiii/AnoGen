"""h_anom energy strategies: nearest / knn / proto vs locked soft KDE."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.diffusion import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch extra not installed")


def test_nearest_is_min_distance():
    import torch

    from anogen.shell.steer import nearest_energy, soft_energy

    z = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    ref = torch.tensor([[1.0, 0.0], [0.0, 4.0], [100.0, 0.0]])
    h = nearest_energy(z, ref)
    assert torch.allclose(h, torch.tensor([1.0, 81.0]))
    tiny = soft_energy(z, ref, tau=1e-8)
    assert torch.allclose(tiny, h, atol=1e-3)


def test_knn_ignores_far_refs():
    import torch

    from anogen.shell.steer import knn_energy, soft_energy

    z = torch.tensor([[0.0, 0.0]])
    near = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mid = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
    ref = torch.cat([near, mid], dim=0)
    h_k = knn_energy(z, ref, tau=1.0, k=2)
    h_only_near = soft_energy(z, near, tau=1.0)
    h_all = soft_energy(z, ref, tau=1.0)
    assert torch.allclose(h_k, h_only_near, atol=1e-5)
    assert float(h_all) < float(h_k) - 1e-6


def test_proto_is_one_ref_per_row():
    import torch

    from anogen.shell.steer import proto_energy

    z = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    proto = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    h = proto_energy(z, proto)
    assert torch.allclose(h, torch.tensor([1.0, 16.0]))


def test_anom_energy_dispatch_and_grad():
    import torch

    from anogen.shell.steer import anom_energy

    z = torch.tensor([[0.2, -0.1]], requires_grad=True)
    ref = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    for kind, extra in (
        ("soft", {}),
        ("nearest", {}),
        ("knn", {"knn": 2}),
        ("proto", {"proto": ref[:1]}),
        ("kind_soft", {"kind_ids": torch.tensor([0, 0, 1])}),
    ):
        z.grad = None
        h = anom_energy(z, ref, tau=0.5, kind=kind, **extra)
        h.mean().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
    with pytest.raises(ValueError):
        anom_energy(z.detach(), ref, 0.5, kind="nope")


def test_kind_balanced_soft_equalizes_kinds():
    import torch

    from anogen.shell.steer import kind_balanced_soft_energy, soft_energy

    z = torch.tensor([[100.0, 0.0]])
    many = torch.zeros(20, 2)
    few = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
    ref = torch.cat([many, few], dim=0)
    ids = torch.tensor([0] * 20 + [1, 1])
    h_all = soft_energy(z, ref, tau=1.0)
    h_bal = kind_balanced_soft_energy(z, ref, tau=1.0, kind_ids=ids)
    # Unbalanced KDE is dominated by the 20 zeros; balanced must differ.
    assert float(h_bal) != float(h_all)
    h_only_few = soft_energy(z, few, tau=1.0)
    h_only_many = soft_energy(z, many, tau=1.0)
    assert float(h_only_many) < float(h_bal) < float(h_only_few) + 5.0


def test_kindmix_cli_and_splits():
    from anogen.cli import _PHASES
    from anogen.phases.kindmix import VARIANTS, _split_counts

    assert "kindmix" in _PHASES
    from anogen.phases.kindmix import ALL_VARIANTS

    assert VARIANTS == ("stratified", "kind_soft", "twomix")
    assert "proto" in ALL_VARIANTS and "band" in ALL_VARIANTS
    assert "hybrid" in ALL_VARIANTS and "hybrid_needles" in ALL_VARIANTS
    from anogen.phases.kindmix import HYBRID_NEEDLES_PROTO_KINDS, HYBRID_PROTO_KINDS

    assert "real level shift" in HYBRID_PROTO_KINDS
    assert "real ESA Point / Global" in HYBRID_NEEDLES_PROTO_KINDS
    assert _split_counts(256, 5) == [52, 51, 51, 51, 51]
    from anogen.phases.kindmix import _kind_alloc_labels

    labels = _kind_alloc_labels(8, ["a", "b"])
    assert list(labels) == ["a", "a", "a", "a", "b", "b", "b", "b"]
    labels4 = _kind_alloc_labels(1536, ["p", "l", "s", "g"])
    assert len(labels4) == 1536
    assert list(np.unique(labels4, return_counts=True)[1]) == [384, 384, 384, 384]


def test_kindproto_recipes_and_cli():
    from anogen.cli import _PHASES
    from anogen.phases.kindproto import RECIPES

    assert "kindproto" in _PHASES
    assert RECIPES["parent50"]["ddim_steps"] == 50
    assert RECIPES["parent50"]["start_from_noise"] is False
    assert RECIPES["noise200"]["ddim_steps"] == 200
    assert RECIPES["noise200"]["start_from_noise"] is True
    assert RECIPES["noise200"]["lam_rare"] == 0.0


def test_kindmixhtune_grid_and_recommend():
    from anogen.cli import _PHASES
    from anogen.phases.kindmixhtune import GRID, recommend_hparams, total_variation

    assert "kindmixhtune" in _PHASES
    assert GRID[0] == ("l03_a10", 0.3, 1.0)
    flat = np.ones((4, 8), dtype=np.float64)
    assert np.allclose(total_variation(flat), 0.0)
    zig = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    assert float(total_variation(zig)[0]) == 1.0
    local = "real ESA local subsequence"
    glob = "real ESA global subsequence"
    point = "real ESA Point / Global"
    shift = "real level shift"

    def _row(recipe, tag, frac_shift, frac_amp, tv_l, tv_g, occ=0.1, arp=0.4):
        return {
            "slices": {
                shift: {"frac_shift": frac_shift},
                point: {"frac_amp": frac_amp},
                local: {"med_tv": tv_l},
                glob: {"med_tv": tv_g},
            },
            "occupancy": occ,
            "arp_anomaly": arp,
            "tv_median": 0.5 * (tv_l + tv_g),
        }

    methods = {
        "time_both_stratified_l03_a10": _row("stratified", "l03_a10", 1.0, 0.9, 0.08, 0.08),
        "time_both_hybrid_needles_l10_a05": _row("hybrid_needles", "l10_a05", 0.95, 0.8, 0.02, 0.02),
        "time_both_hybrid_l03_a10": _row("hybrid", "l03_a10", 0.95, 0.1, 0.01, 0.01),
        "time_recon_stratified_l03_a10": _row("stratified", "l03_a10", 1.0, 0.9, 0.08, 0.08),
        "time_recon_hybrid_needles_l10_a05": _row("hybrid_needles", "l10_a05", 0.95, 0.8, 0.02, 0.02),
    }
    real_tv = {local: 0.02, glob: 0.02}
    rec = recommend_hparams(methods, real_tv)
    assert rec["time_both"]["recipe"] == "hybrid_needles"
    assert rec["time_both"]["tag"] == "l10_a05"
    assert rec["time_both"]["viable_kept_kinds"] is True
