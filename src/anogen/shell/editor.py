"""Post-hoc editor: keep the donor carrier, take only the slow hitch from DDIM.

Quiet subsequence and level-shift slices: residual copy
``x = slow(gen) + fast(parent)``. Point/Global needles are high-frequency, so
those slices use GenIAS deviation-patch instead (keep gen only where it leaves
the parent enough). Isolated from S3/S4 sampling.
"""

from __future__ import annotations

import numpy as np

from anogen.shell.genias import deviation_patch

NEEDLE_KINDS = frozenset({"real ESA Point / Global"})
DEFAULT_N_KEEP = 3
DEFAULT_PATCH_TAU = 0.2


def slow_component(x: np.ndarray, n_keep: int = DEFAULT_N_KEEP) -> np.ndarray:
    """Keep the first ``n_keep`` rFFT bins (DC + slow trend / shelf)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    n_keep = max(1, int(n_keep))
    spec = np.fft.rfft(x, axis=1)
    n_keep = min(n_keep, spec.shape[1])
    spec[:, n_keep:] = 0.0
    out = np.fft.irfft(spec, n=x.shape[1], axis=1)
    return out[0] if squeeze else out


def residual_copy(
    parent: np.ndarray,
    generated: np.ndarray,
    *,
    n_keep: int = DEFAULT_N_KEEP,
) -> np.ndarray:
    """Replace the generated carrier with the donor's: parent + slow(gen − parent)."""
    parent = np.asarray(parent, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if parent.shape != generated.shape:
        raise ValueError(f"parent {parent.shape} vs generated {generated.shape}")
    squeeze = parent.ndim == 1
    if squeeze:
        parent = parent[None, :]
        generated = generated[None, :]
    slow_g = slow_component(generated, n_keep)
    slow_p = slow_component(parent, n_keep)
    out = parent + (slow_g - slow_p)
    if squeeze:
        return out[0].astype(np.float32)
    return out.astype(np.float32)


def apply_editor(
    parent: np.ndarray,
    generated: np.ndarray,
    kind_alloc: np.ndarray | None = None,
    *,
    n_keep: int = DEFAULT_N_KEEP,
    patch_tau: float = DEFAULT_PATCH_TAU,
    needle_kinds: frozenset[str] = NEEDLE_KINDS,
) -> np.ndarray:
    """Edit a hybrid_needles gallery. Residual copy, except needles."""
    parent = np.asarray(parent)
    generated = np.asarray(generated)
    if parent.shape != generated.shape:
        raise ValueError(f"parent {parent.shape} vs generated {generated.shape}")
    squeeze = parent.ndim == 1
    if squeeze:
        parent = parent[None, :]
        generated = generated[None, :]
        kind_alloc = None if kind_alloc is None else np.asarray(kind_alloc).reshape(-1)
    out = residual_copy(parent, generated, n_keep=n_keep)
    if kind_alloc is not None:
        alloc = np.asarray(kind_alloc).astype(str).reshape(-1)
        if len(alloc) != len(out):
            raise ValueError(f"kind_alloc {len(alloc)} vs gallery {len(out)}")
        needles = np.array([k in needle_kinds for k in alloc], dtype=bool)
        if needles.any() and float(patch_tau) > 0:
            out[needles] = deviation_patch(
                parent[needles], generated[needles], float(patch_tau)
            )
    if squeeze:
        return np.asarray(out[0], dtype=np.float32)
    return np.asarray(out, dtype=np.float32)
