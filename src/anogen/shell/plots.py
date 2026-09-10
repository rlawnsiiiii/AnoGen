"""Time-series figures for visual comparison of real vs generated windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _hours(width: int, bin_seconds: int = 30) -> np.ndarray:
    return np.arange(width) * float(bin_seconds) / 3600.0


def _require_mpl() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plots. uv sync --extra neural")


def _pick(x: np.ndarray, n: int, rng: np.random.Generator, *, extreme: bool) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    n = min(n, len(x))
    if extreme:
        score = x.max(axis=1) - x.min(axis=1)
        return np.argsort(score)[::-1][:n]
    return rng.choice(len(x), size=n, replace=False)


def _one_ax(ax: Any, y: np.ndarray, t: np.ndarray, color: str, title: str, ylim: tuple[float, float] | None) -> None:
    ax.plot(t, y, color=color, lw=0.9)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    if ylim is not None:
        ax.set_ylim(*ylim)


def save_family_grid(
    path: Path,
    series: dict[str, np.ndarray],
    *,
    bin_seconds: int,
    rng: np.random.Generator,
    n_cols: int = 6,
) -> None:
    """One row per family, one extreme example per column."""
    _require_mpl()
    families = list(series.keys())
    t = _hours(next(iter(series.values())).shape[1], bin_seconds)
    fig, axes = plt.subplots(len(families), n_cols, figsize=(2.4 * n_cols, 1.8 * len(families)), sharex=True)
    if len(families) == 1:
        axes = np.asarray([axes])
    colors = {"anomaly": "#b2182b", "rare": "#2166ac", "nominal": "#4d4d4d"}
    for r, name in enumerate(families):
        x = series[name]
        idx = _pick(x, n_cols, rng, extreme=name != "nominal")
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(idx):
                ax.axis("off")
                continue
            _one_ax(
                ax,
                x[idx[c]],
                t,
                colors.get(name, "#333333"),
                f"{name} #{int(idx[c])}",
                None,
            )
            if r == len(families) - 1:
                ax.set_xlabel("hours", fontsize=7)
    fig.suptitle("Real windows (anomalies: largest range; rares/nominal: random mix)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_strip(path: Path, x: np.ndarray, *, title: str, color: str, bin_seconds: int, idx: np.ndarray) -> None:
    _require_mpl()
    n = len(idx)
    if n == 0:
        return
    t = _hours(x.shape[1], bin_seconds)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 1.9 * rows), sharex=True)
    axes = np.atleast_2d(axes)
    for i, j in enumerate(idx):
        ax = axes[i // cols, i % cols]
        _one_ax(ax, x[int(j)], t, color, f"{title} #{int(j)}", None)
        if i // cols == rows - 1:
            ax.set_xlabel("hours", fontsize=7)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_same_parent(
    path: Path,
    parent: np.ndarray,
    kids: dict[str, np.ndarray],
    *,
    bin_seconds: int,
    titles: list[str] | None = None,
) -> None:
    """Each column is one parent window; rows are parent + generators."""
    _require_mpl()
    names = ["parent (nominal)"] + list(kids.keys())
    n_ex = parent.shape[0]
    t = _hours(parent.shape[1], bin_seconds)
    fig, axes = plt.subplots(len(names), n_ex, figsize=(3.2 * n_ex, 1.55 * len(names)), sharex=True)
    if n_ex == 1:
        axes = axes.reshape(-1, 1)
    for c in range(n_ex):
        stack = [parent[c]] + [kids[k][c] for k in kids]
        lo = min(float(s.min()) for s in stack)
        hi = max(float(s.max()) for s in stack)
        pad = 0.05 * (hi - lo + 1e-6)
        ylim = (lo - pad, hi + pad)
        palette = ["#4d4d4d", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
        for r, (name, y) in enumerate(zip(names, stack, strict=True)):
            ax = axes[r, c]
            _one_ax(ax, y, t, palette[r % len(palette)], f"{name}" if c == 0 else name, ylim)
            if r == len(names) - 1:
                ax.set_xlabel("hours", fontsize=7)
    fig.suptitle("Same nominal parent, each generator (shared y-scale per column)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    _ = titles


def save_overlay(
    path: Path,
    parent: np.ndarray,
    overlays: dict[str, np.ndarray],
    *,
    bin_seconds: int,
) -> None:
    _require_mpl()
    n = parent.shape[0]
    t = _hours(parent.shape[1], bin_seconds)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.0 * n), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {
        "parent": "#4d4d4d",
        "genias ψ=1 (recon)": "#1b9e77",
        "genias ψ=1.5": "#d95f02",
        "genias ψ=2": "#e7298a",
        "post-hoc": "#7570b3",
        "unguided diffusion": "#66a61e",
        "shell ZS": "#e6ab02",
    }
    for i in range(n):
        ax = axes[i]
        ax.plot(t, parent[i], color=colors["parent"], lw=1.1, label="parent" if i == 0 else None)
        for name, y in overlays.items():
            ax.plot(t, y[i], lw=0.9, alpha=0.85, color=colors.get(name, None), label=name if i == 0 else None)
        ax.set_ylabel("value", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("hours")
    if n:
        axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Overlay on the same nominal parent", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_compare_rows(
    path: Path,
    rows: dict[str, np.ndarray],
    *,
    bin_seconds: int,
    n_cols: int = 5,
    colors: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """One row per family, shared column count. Independent examples (not same parent)."""
    _require_mpl()
    names = [k for k, v in rows.items() if v is not None and len(v)]
    if not names:
        return
    t = _hours(next(iter(rows[n] for n in names)).shape[1], bin_seconds)
    fig, axes = plt.subplots(len(names), n_cols, figsize=(2.6 * n_cols, 1.7 * len(names)), sharex=True)
    if len(names) == 1:
        axes = np.asarray([axes])
    palette = colors or {}
    default = ["#b2182b", "#2166ac", "#4d4d4d", "#1b9e77", "#7570b3", "#66a61e", "#e6ab02", "#d95f02"]
    for r, name in enumerate(names):
        x = rows[name]
        n = min(n_cols, len(x))
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= n:
                ax.axis("off")
                continue
            color = palette.get(name, default[r % len(default)])
            _one_ax(ax, x[c], t, color, name if c == 0 else f"#{c}", None)
            if r == len(names) - 1:
                ax.set_xlabel("hours", fontsize=7)
    fig.suptitle(title or "Real vs generated windows (independent examples per row)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_named_grid(
    path: Path,
    cells: dict[tuple[str, str], np.ndarray],
    *,
    row_names: list[str],
    col_names: list[str],
    bin_seconds: int,
    colors: dict[str, str] | None = None,
    cell_titles: dict[tuple[str, str], str] | None = None,
    title: str = "",
    share_col_ylim: bool = False,
) -> None:
    """One named window per (row, column). Used for kind × method comparisons."""
    _require_mpl()
    if not row_names or not col_names:
        return
    sample = next(iter(cells.values()))
    t = _hours(len(sample), bin_seconds)
    fig, axes = plt.subplots(
        len(row_names),
        len(col_names),
        figsize=(2.7 * len(col_names), 1.75 * len(row_names)),
        sharex=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if len(row_names) == 1 else axes.reshape(-1, 1)
    palette = colors or {}
    default = ["#b2182b", "#2166ac", "#1b9e77", "#d95f02", "#7570b3"]
    titles = cell_titles or {}
    col_ylim: dict[str, tuple[float, float]] = {}
    if share_col_ylim:
        for col in col_names:
            ys = [np.asarray(cells[(row, col)]) for row in row_names if (row, col) in cells]
            if not ys:
                continue
            lo = min(float(y.min()) for y in ys)
            hi = max(float(y.max()) for y in ys)
            pad = 0.05 * (hi - lo + 1e-6)
            col_ylim[col] = (lo - pad, hi + pad)
    for r, row in enumerate(row_names):
        for c, col in enumerate(col_names):
            ax = axes[r, c]
            key = (row, col)
            if key not in cells:
                ax.axis("off")
                continue
            color = palette.get(row, default[r % len(default)])
            head = titles.get(key) or (col if r == 0 else f"{row}")
            if r == 0:
                head = col
            elif c == 0:
                head = row
            else:
                head = titles.get(key, "")
            _one_ax(ax, np.asarray(cells[key]), t, color, head, col_ylim.get(col))
            if r == len(row_names) - 1:
                ax.set_xlabel("hours", fontsize=7)
    fig.suptitle(title or "Real anomaly kinds vs generated", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_loss_curves(
    path: Path,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    ylabel: str = "loss",
    log_y: bool = True,
) -> None:
    """One PNG with several (step, value) traces."""
    _require_mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = ["#2166ac", "#b2182b", "#4d4d4d", "#1b9e77", "#d95f02", "#7570b3"]
    for i, (name, (step, value)) in enumerate(series.items()):
        if step is None or value is None or len(step) == 0:
            continue
        ax.plot(step, value, lw=1.2, color=colors[i % len(colors)], label=name)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_bar(
    path: Path,
    labels: list[str],
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> None:
    _require_mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(np.arange(len(labels)), values, color="#2166ac")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_embedding_scatter(
    path: Path,
    points: dict[str, np.ndarray],
    *,
    colors: dict[str, str],
    title: str,
    seed: int = 0,
    max_per: int = 256,
) -> None:
    """2-D t-SNE of locked embeddings (GenIAS Fig. 4 style)."""
    from sklearn.manifold import TSNE

    _require_mpl()
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    names_kept: list[str] = []
    for name, emb in points.items():
        emb = np.asarray(emb, dtype=np.float64)
        if len(emb) == 0:
            continue
        if len(emb) > max_per:
            emb = emb[rng.choice(len(emb), size=max_per, replace=False)]
        chunks.append(emb)
        names_kept.append(name)
    if not chunks:
        return
    stacked = np.concatenate(chunks, axis=0)
    perp = min(30, max(5, len(stacked) // 4))
    xy = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(stacked)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    start = 0
    for name, emb in zip(names_kept, chunks, strict=True):
        sl = slice(start, start + len(emb))
        start += len(emb)
        ax.scatter(
            xy[sl, 0],
            xy[sl, 1],
            s=12,
            c=colors.get(name, "#333333"),
            alpha=0.65,
            label=name,
            linewidths=0,
        )
    ax.legend(fontsize=8, markerscale=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
