"""Sampling-HP grid and winner rules (no GPU sweep)."""

from __future__ import annotations

from anogen.phases.tune import default_grids, pick_winners


def test_default_grids_have_s4_anchor_and_contrast():
    grid = default_grids()
    assert len(grid) >= 12
    families = {s["family"] for s in grid}
    assert families == {"zs", "contrast"}
    assert any(
        s["family"] == "zs"
        and s["ddim_steps"] == 20
        and s["lam"] == 0.3
        and s["nu"] == 0.2
        and s["n_correct"] == 1
        for s in grid
    )
    assert any(s["family"] == "contrast" and s["lam_anom"] > 0 and s["lam_rare"] > 0 for s in grid)
    assert all(s["ddim_steps"] >= 20 for s in grid)


def test_pick_winners_respects_declared_keys():
    rows = [
        {
            "family": "zs",
            "finite": True,
            "explode": False,
            "occupancy": 0.2,
            "arp_anomaly": 0.3,
            "gap": 0.01,
            "id": 0,
        },
        {
            "family": "zs",
            "finite": True,
            "explode": False,
            "occupancy": 0.8,
            "arp_anomaly": 0.2,
            "gap": -0.02,
            "id": 1,
        },
        {
            "family": "contrast",
            "finite": True,
            "explode": False,
            "occupancy": 0.05,
            "arp_anomaly": 0.55,
            "gap": 0.04,
            "id": 2,
        },
        {
            "family": "zs",
            "finite": False,
            "explode": True,
            "occupancy": 1.0,
            "arp_anomaly": 0.99,
            "gap": 1.0,
            "id": 3,
        },
    ]
    win = pick_winners(rows)
    assert win["best_occupancy"]["id"] == 1
    assert win["best_arp_anomaly"]["id"] == 2
    assert win["best_gap"]["id"] == 2
