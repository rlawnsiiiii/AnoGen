"""Write time-series PNGs of real events vs generated windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anogen.config import REPO_ROOT
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available, unguided_from_nominal
from anogen.shell.coverage import arp, min_distances
from anogen.shell.features import embed_windows
from anogen.shell.plots import (
    save_bar,
    save_compare_rows,
    save_embedding_scatter,
    save_family_grid,
    save_loss_curves,
    save_overlay,
    save_same_parent,
    save_strip,
)
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import ConvEncoder, embed_encoder, guided_ddim


def run_plots(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s2 = _abs(cfg.get("s2_dir", root / "results/shell_s2"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    out = _abs(cfg.get("plots_dir", root / "results/shell_plots"), root)
    out.mkdir(parents=True, exist_ok=True)
    bin_seconds = int(cfg.get("bin_seconds", 30))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    lab_path = s0 / "labeled_arrays.npz"
    gal_path = s3 / "galleries.npz"
    if not lab_path.is_file() or not gal_path.is_file():
        report = {"ok": False, "skipped": True, "reason": "Need S0 labeled_arrays and S3 galleries."}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    lab = np.load(lab_path)
    gal = np.load(gal_path)
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    ch_a = np.asarray(lab["anomaly_channel"])
    parent = np.asarray(gal["cond"])
    ch_p = np.asarray(gal["channel_idx"])
    genias = np.asarray(gal["genias"])
    post = np.asarray(gal["posthoc"]) if "posthoc" in gal.files else None
    gia1 = np.asarray(gal["genias_psi1.0"]) if "genias_psi1.0" in gal.files else None
    gia15 = np.asarray(gal["genias_psi1.5"]) if "genias_psi1.5" in gal.files else None

    unguided, shell = _load_or_sample_diffusion(cfg, s1, s2, s4, parent, ch_p)
    _copy_train_diagnostics(s1, s3, out)

    save_family_grid(
        out / "real_families.png",
        {"anomaly": x_a, "rare": x_r, "nominal": parent},
        bin_seconds=bin_seconds,
        rng=rng,
    )
    save_strip(
        out / "real_anomalies.png",
        x_a,
        title="Real anomaly windows (largest range first)",
        color="#b2182b",
        bin_seconds=bin_seconds,
        idx=_pick_mixed(x_a, 8, rng, extreme=True),
    )
    save_strip(
        out / "real_rares.png",
        x_r,
        title="Real rare-event windows",
        color="#2166ac",
        bin_seconds=bin_seconds,
        idx=_pick_mixed(x_r, 8, rng, extreme=False),
    )
    save_strip(
        out / "real_nominal.png",
        parent,
        title="Nominal conditioning windows",
        color="#4d4d4d",
        bin_seconds=bin_seconds,
        idx=_pick_mixed(parent, 8, rng, extreme=False),
    )

    if genias is not None and len(genias):
        save_strip(
            out / "gen_genias.png",
            genias,
            title="GenIAS ψ=2 (generated)",
            color="#e7298a",
            bin_seconds=bin_seconds,
            idx=_pick_mixed(genias, 8, rng, extreme=True),
        )
    if post is not None and len(post):
        save_strip(
            out / "gen_posthoc.png",
            post,
            title="Post-hoc injection (generated)",
            color="#7570b3",
            bin_seconds=bin_seconds,
            idx=_pick_mixed(post, 8, rng, extreme=True),
        )
    if unguided is not None and len(unguided):
        save_strip(
            out / "gen_unguided.png",
            unguided,
            title="Unguided diffusion (generated)",
            color="#66a61e",
            bin_seconds=bin_seconds,
            idx=_pick_mixed(unguided, 8, rng, extreme=True),
        )
    if shell is not None and len(shell):
        save_strip(
            out / "gen_shell.png",
            shell,
            title="Shell-steered diffusion (generated)",
            color="#e6ab02",
            bin_seconds=bin_seconds,
            idx=_pick_mixed(shell, 8, rng, extreme=True),
        )

    compare = {
        "real anomaly": _head(x_a, _pick_mixed(x_a, 5, rng, extreme=True)),
        "real rare": _head(x_r, _pick_mixed(x_r, 5, rng, extreme=False)),
        "nominal parent": _head(parent, _pick_mixed(parent, 5, rng, extreme=False)),
        "GenIAS ψ=2": _head(genias, _pick_mixed(genias, 5, rng, extreme=True)),
    }
    if post is not None:
        compare["post-hoc"] = _head(post, _pick_mixed(post, 5, rng, extreme=True))
    if unguided is not None:
        compare["unguided diffusion"] = _head(unguided, _pick_mixed(unguided, 5, rng, extreme=True))
    if shell is not None:
        compare["shell ZS"] = _head(shell, _pick_mixed(shell, 5, rng, extreme=True))
    save_compare_rows(
        out / "real_vs_generated.png",
        compare,
        bin_seconds=bin_seconds,
        n_cols=5,
        colors={
            "real anomaly": "#b2182b",
            "real rare": "#2166ac",
            "nominal parent": "#4d4d4d",
            "GenIAS ψ=2": "#e7298a",
            "post-hoc": "#7570b3",
            "unguided diffusion": "#66a61e",
            "shell ZS": "#e6ab02",
        },
    )

    cols = _one_per_channel(ch_p, n_max=6) or list(range(min(4, len(parent))))
    kids = {"genias ψ=2": genias[cols]}
    if post is not None:
        kids = {"post-hoc": post[cols], **kids}
    if gia1 is not None:
        kids = {"genias ψ=1 (recon)": gia1[cols], **kids}
    if gia15 is not None:
        kids["genias ψ=1.5"] = gia15[cols]
    if unguided is not None and len(unguided) > max(cols):
        kids["unguided diffusion"] = unguided[cols]
    if shell is not None and len(shell) > max(cols):
        kids["shell ZS"] = shell[cols]
    save_same_parent(out / "same_parent_grid.png", parent[cols], kids, bin_seconds=bin_seconds)

    ov = {k: v for k, v in kids.items() if k != "genias ψ=1.5"}
    save_overlay(out / "same_parent_overlay.png", parent[cols], ov, bin_seconds=bin_seconds)

    _write_arp_tsne(out, x_a, x_r, genias, post, unguided, shell, rng)

    ch_dir = out / "by_channel"
    ch_dir.mkdir(exist_ok=True)
    channels = list(cfg.get("channels") or [f"channel_{41 + i}" for i in range(6)])
    for cidx, name in enumerate(channels):
        a_idx = np.where(ch_a == cidx)[0]
        p_idx = np.where(ch_p == cidx)[0]
        if len(a_idx) == 0 or len(p_idx) == 0:
            continue
        a_pick = _pick_mixed(x_a[a_idx], min(4, len(a_idx)), rng, extreme=True)
        save_strip(
            ch_dir / f"{name}_anomalies.png",
            x_a[a_idx],
            title=f"{name} real anomalies",
            color="#b2182b",
            bin_seconds=bin_seconds,
            idx=a_pick,
        )
        save_strip(
            ch_dir / f"{name}_nominal.png",
            parent[p_idx],
            title=f"{name} nominal",
            color="#4d4d4d",
            bin_seconds=bin_seconds,
            idx=_pick_mixed(parent[p_idx], min(4, len(p_idx)), rng, extreme=False),
        )
        if len(p_idx):
            g_pick = _pick_mixed(genias[p_idx], min(4, len(p_idx)), rng, extreme=True)
            save_strip(
                ch_dir / f"{name}_genias.png",
                genias[p_idx],
                title=f"{name} GenIAS",
                color="#e7298a",
                bin_seconds=bin_seconds,
                idx=g_pick,
            )
            if shell is not None and len(shell) == len(parent):
                save_strip(
                    ch_dir / f"{name}_shell.png",
                    shell[p_idx],
                    title=f"{name} shell ZS",
                    color="#e6ab02",
                    bin_seconds=bin_seconds,
                    idx=_pick_mixed(shell[p_idx], min(4, len(p_idx)), rng, extreme=True),
                )

    written = sorted(p.name for p in out.glob("*.png")) + [
        f"by_channel/{p.name}" for p in sorted(ch_dir.glob("*.png"))
    ]
    (out / "INDEX.txt").write_text(
        "Time-series plots (x = hours at 30 s bins, W = 512 ≈ 4.3 h).\n\n"
        + "\n".join(f"  {n}" for n in written)
        + "\n\nreal_families.png / real_*.png: labeled ESA windows.\n"
        "gen_*.png: generated galleries (extreme range first).\n"
        "real_vs_generated.png: one row each, independent examples.\n"
        "same_parent_*: one quiet parent and every generator on that same window.\n"
        "train_s1_*.png / train_s3_*.png: loss curves and recon/sample checks.\n"
        "by_channel/: anomalies, nominal, GenIAS, shell per channel 41–46.\n"
        "arp_tsne.png: feature_pack_v1 t-SNE (real anomalies/rares vs galleries).\n"
        "arp_bars.png: GenIAS-style ARP (anomaly and rare) per method.\n"
    )
    from anogen.phases.plot_tune import run_plot_tune

    tune_report = run_plot_tune(cfg)
    written = written + [f"shell_tune/{n}" for n in tune_report.get("files") or []]
    report = {"ok": True, "skipped": False, "dir": str(out), "files": written, "shell_tune": tune_report}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _load_or_sample_diffusion(
    cfg: dict[str, Any],
    s1: Path,
    s2: Path,
    s4: Path,
    parent: np.ndarray,
    ch: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Unguided from S1 (fresh denoiser). Shell from S4/S2 eval galleries."""
    unguided = None
    shell = None
    s1u = s1 / "unguided_samples.npz"
    if s1u.is_file():
        unguided = np.asarray(np.load(s1u)["x"])
    gal4 = s4 / "shell_gallery.npz"
    if gal4.is_file():
        blob = np.load(gal4)
        if "x" in blob.files:
            shell = np.asarray(blob["x"])
        if unguided is None and "unguided" in blob.files:
            unguided = np.asarray(blob["unguided"])
    s2g = s2 / "guided_samples.npz"
    if shell is None and s2g.is_file():
        shell = np.asarray(np.load(s2g)["x"])
    if (unguided is None or shell is None) and torch_available() and (s1 / "denoiser.pt").is_file():
        live_u, live_s = _maybe_sample_diffusion(cfg, s1, s2, parent[: min(32, len(parent))], ch[: min(32, len(ch))])
        unguided = unguided if unguided is not None else live_u
        shell = shell if shell is not None else live_s
    return unguided, shell


def _maybe_sample_diffusion(
    cfg: dict[str, Any],
    s1: Path,
    s2: Path,
    parent: np.ndarray,
    ch: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not torch_available() or not (s1 / "denoiser.pt").is_file():
        return None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device)
    scaler = resolve_scaler(den)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(torch.device(device))
    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    unguided = unguided_from_nominal(
        model,
        schedule,
        parent,
        ch,
        nu=float(dcfg.get("unguided_nu", 1.0)),
        ddim_steps=int(dcfg.get("ddim_steps", 20)),
        device=device,
        scaler=scaler,
    )
    shell = None
    enc_path = s2 / "encoder.pt"
    if enc_path.is_file():
        enc_blob = torch.load(enc_path, map_location=device, weights_only=False)
        encoder = ConvEncoder(
            int(enc_blob["width"]), hidden=int(enc_blob["hidden"]), emb=int(enc_blob["emb"])
        )
        encoder.load_state_dict(enc_blob["state_dict"])
        encoder.to(device)
        ref = torch.from_numpy(embed_encoder(encoder, parent[: min(512, len(parent))], device))
        chunks = []
        bsz = 32
        for i in range(0, len(parent), bsz):
            s, _h = guided_ddim(
                model,
                encoder,
                parent[i : i + bsz],
                ch[i : i + bsz],
                schedule,
                ref=ref,
                Q_q=float(enc_blob["Q_q"]),
                tau=float(enc_blob["tau"]),
                nu=float(scfg.get("nu", 0.2)),
                lam=float(scfg.get("lambda", 0.3)),
                c_max=float(scfg.get("c_max", 1.0)),
                ddim_steps=int(dcfg.get("ddim_steps", 20)),
                device=device,
                mode="band",
                scaler=scaler,
            )
            chunks.append(s)
        shell = np.concatenate(chunks, axis=0)
    return unguided, shell


def _one_per_channel(ch: np.ndarray, n_max: int) -> list[int]:
    cols = []
    if len(ch) == 0:
        return cols
    for cidx in range(int(ch.max()) + 1):
        hits = np.where(ch == cidx)[0]
        if len(hits) == 0:
            continue
        cols.append(int(hits[0]))
        if len(cols) >= n_max:
            break
    return cols


def _head(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if x is None or len(x) == 0 or len(idx) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    return x[idx]


def _pick_mixed(x: np.ndarray, n: int, rng: np.random.Generator, *, extreme: bool) -> np.ndarray:
    n = min(n, len(x))
    if n == 0:
        return np.zeros(0, dtype=int)
    n_ext = n // 2 if extreme else 0
    if n_ext:
        score = x.max(axis=1) - x.min(axis=1)
        ext = np.argsort(score)[::-1][:n_ext]
        rest_pool = np.setdiff1d(np.arange(len(x)), ext)
        n_rand = n - len(ext)
        rand = (
            rng.choice(rest_pool, size=min(n_rand, len(rest_pool)), replace=False)
            if len(rest_pool)
            else np.zeros(0, dtype=int)
        )
        return np.concatenate([ext, rand])
    return rng.choice(len(x), size=n, replace=False)


def _copy_train_diagnostics(s1: Path, s3: Path, out: Path) -> None:
    """Re-render S1/S3 training curves into the plot folder."""
    s1h = s1 / "loss_history.npz"
    if s1h.is_file():
        h = np.load(s1h)
        series = {}
        if "loss_step" in h.files:
            series["train EMA ε-MSE"] = (h["loss_step"], h["loss_value"])
        if "val_step" in h.files:
            series["val ε-MSE"] = (h["val_step"], h["val_value"])
        if series:
            save_loss_curves(
                out / "train_s1_loss.png",
                series,
                title="S1 denoiser: noise-prediction loss",
                ylabel="ε-MSE",
            )
        if "loss_by_t" in h.files:
            labels = [f"t {int(a)}–{int(b) - 1}" for a, b in zip(h["loss_by_t_lo"], h["loss_by_t_hi"])]
            save_bar(
                out / "train_s1_loss_by_t.png",
                labels,
                h["loss_by_t"],
                title="S1 val ε-MSE by diffusion time",
                ylabel="MSE",
            )
    s3h = s3 / "loss_history.npz"
    if s3h.is_file():
        h = np.load(s3h)
        series = {}
        if "loss_step" in h.files:
            series["train EMA (recon+βKL)"] = (h["loss_step"], h["loss_value"])
            if "loss_recon" in h.files:
                series["train recon MSE"] = (h["loss_step"], h["loss_recon"])
        if "val_step" in h.files:
            series["val recon MSE"] = (h["val_step"], h["val_recon"])
        if series:
            save_loss_curves(
                out / "train_s3_loss.png",
                series,
                title="S3 GenIAS: per-window z-scored reconstruction + KL",
                ylabel="whitened MSE / loss",
            )
    for src, name in (
        (s1 / "unguided_vs_parent.png", "train_s1_unguided_vs_parent.png"),
        (s3 / "recon_vs_parent.png", "train_s3_recon_vs_parent.png"),
    ):
        if src.is_file():
            dest = out / name
            dest.write_bytes(src.read_bytes())


def _write_arp_tsne(
    out: Path,
    x_a: np.ndarray,
    x_r: np.ndarray,
    genias: np.ndarray,
    post: np.ndarray | None,
    unguided: np.ndarray | None,
    shell: np.ndarray | None,
    rng: np.random.Generator,
) -> None:
    del rng
    points = {
        "real anomaly": embed_windows(x_a),
        "real rare": embed_windows(x_r),
        "GenIAS": embed_windows(genias),
    }
    colors = {
        "real anomaly": "#b2182b",
        "real rare": "#2166ac",
        "GenIAS": "#e7298a",
        "post-hoc": "#7570b3",
        "unguided": "#66a61e",
        "shell ZS": "#e6ab02",
    }
    if post is not None and len(post):
        points["post-hoc"] = embed_windows(post)
    if unguided is not None and len(unguided):
        points["unguided"] = embed_windows(unguided)
    if shell is not None and len(shell):
        points["shell ZS"] = embed_windows(shell)
    save_embedding_scatter(
        out / "arp_tsne.png",
        points,
        colors=colors,
        title="feature_pack_v1 t-SNE (real queries vs generated galleries)",
    )
    qa, qr = points["real anomaly"], points["real rare"]
    labels, va, vr = [], [], []
    for name in ("shell ZS", "unguided", "GenIAS", "post-hoc"):
        if name not in points:
            continue
        labels.append(name)
        va.append(arp(min_distances(qa, points[name])))
        vr.append(arp(min_distances(qr, points[name])))
    if labels:
        fig_labels = [f"{n}\nanom" for n in labels] + [f"{n}\nrare" for n in labels]
        save_bar(
            out / "arp_bars.png",
            fig_labels,
            np.array(va + vr, dtype=np.float64),
            title="ARP in feature_pack_v1 (higher = closer to that query set)",
            ylabel="ARP",
        )


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
