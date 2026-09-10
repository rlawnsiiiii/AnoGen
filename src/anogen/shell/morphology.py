"""Anomaly-window kinds: official ESA-ADB types plus one shape overlay.

ESA-ADB does not name events level shift / spike / campaign. Official type
columns in ``anomaly_types.csv`` are Length (Point / Subsequence) and
Locality (Local / Global), plus Dimensionality. See docs/ESA_LABELS.md.

Default kinds are those two ESA columns (Blázquez-García / Kotowski). The
only extra rule is Lai's trend / mean-shift test on global subsequences.
The old duration+amplitude five-name split is ``assign_morphology_kinds``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anogen.shell.features import embed_windows

KIND_ORDER = (
    "real ESA Point / Local",
    "real ESA Point / Global",
    "real ESA local subsequence",
    "real level shift",
    "real ESA global subsequence",
)

KIND_SLUG = {
    "real ESA Point / Local": "esa_point_local",
    "real ESA Point / Global": "esa_point_global",
    "real ESA local subsequence": "esa_local_subseq",
    "real level shift": "level_shift",
    "real ESA global subsequence": "esa_global_subseq",
}

MORPH_KIND_ORDER = (
    "real short spike",
    "real short subtle",
    "real level shift",
    "real medium event",
    "real long campaign",
)

MORPH_KIND_SLUG = {
    "real short spike": "short_spike",
    "real short subtle": "short_subtle",
    "real level shift": "level_shift",
    "real medium event": "medium_event",
    "real long campaign": "long_campaign",
}

_ESA_LEN = ("esa_length", "Length")
_ESA_LOC = ("esa_locality", "Locality")


def _cell(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _all_blank(series: pd.Series) -> bool:
    return bool(series.map(_cell).eq("").all())


def esa_length_locality(
    windows: pd.DataFrame,
    types: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Length / Locality per window from window columns or ``anomaly_types.csv``."""
    w = windows.reset_index(drop=True)
    for length_col, loc_col in ((_ESA_LEN[0], _ESA_LOC[0]), (_ESA_LEN[1], _ESA_LOC[1])):
        if length_col in w.columns and loc_col in w.columns and not _all_blank(w[length_col]):
            return (
                w[length_col].map(_cell).to_numpy(dtype=object),
                w[loc_col].map(_cell).to_numpy(dtype=object),
            )
    if types is None or "ID" not in types.columns:
        raise ValueError(
            "Need ESA Length/Locality on the window table or an anomaly_types.csv "
            "with an ID column"
        )
    idx = types.copy()
    idx["ID"] = idx["ID"].astype(str)
    idx = idx.drop_duplicates("ID").set_index("ID")
    ids = w["event_id"].astype(str)
    length = np.array(
        [_cell(idx.loc[i, "Length"]) if i in idx.index and "Length" in idx.columns else "" for i in ids],
        dtype=object,
    )
    locality = np.array(
        [_cell(idx.loc[i, "Locality"]) if i in idx.index and "Locality" in idx.columns else "" for i in ids],
        dtype=object,
    )
    return length, locality


def assign_anomaly_kinds(
    x: np.ndarray,
    length: np.ndarray,
    locality: np.ndarray,
    *,
    shift_cut: float = 0.01,
) -> np.ndarray:
    """Label windows from official ESA Length × Locality, plus a level-shift overlay.

    Point / Local / Global / Subsequence come from ``anomaly_types.csv``.
    ``real level shift`` is the only invented rule: a *global subsequence*
    whose crop has |μ_L − μ_R| ≥ ``shift_cut`` (Lai trend / mean shift).
    """
    x = np.asarray(x)
    n = len(x)
    length = np.asarray(length, dtype=object)
    locality = np.asarray(locality, dtype=object)
    if n == 0:
        return np.zeros(0, dtype=object)
    if len(length) != n or len(locality) != n:
        raise ValueError("length and locality must align with x")
    _, dmu = window_shape_stats(x)
    labels = np.empty(n, dtype=object)
    for i in range(n):
        leng = _cell(length[i]).lower()
        loc = _cell(locality[i]).lower()
        if leng == "point":
            labels[i] = (
                "real ESA Point / Local" if loc == "local" else "real ESA Point / Global"
            )
        elif leng == "subsequence":
            if loc == "local":
                labels[i] = "real ESA local subsequence"
            elif float(dmu[i]) >= shift_cut:
                labels[i] = "real level shift"
            else:
                labels[i] = "real ESA global subsequence"
        else:
            labels[i] = "real ESA unknown"
    return labels


def kinds_for_windows(
    x: np.ndarray,
    windows: pd.DataFrame,
    types: pd.DataFrame | None = None,
    *,
    shift_cut: float = 0.01,
) -> np.ndarray:
    """``assign_anomaly_kinds`` after joining ESA type columns onto ``windows``."""
    length, locality = esa_length_locality(windows, types)
    return assign_anomaly_kinds(x, length, locality, shift_cut=shift_cut)


def assign_morphology_kinds(
    x: np.ndarray,
    span: np.ndarray,
    *,
    amp_spike: float = 0.1,
    shift_cut: float = 0.01,
    short: int = 100,
    medium: int = 4000,
) -> np.ndarray:
    """Legacy duration+shape names. Not ESA types. Kept to read old galleries."""
    x = np.asarray(x)
    span = np.asarray(span, dtype=np.int64)
    if len(x) == 0:
        return np.zeros(0, dtype=object)
    amp = x.max(axis=1) - x.min(axis=1)
    half = x.shape[1] // 2
    shift = np.abs(x[:, :half].mean(axis=1) - x[:, half:].mean(axis=1))
    labels = np.empty(len(x), dtype=object)
    for i in range(len(x)):
        if int(span[i]) < short and float(amp[i]) >= amp_spike:
            labels[i] = "real short spike"
        elif int(span[i]) < short:
            labels[i] = "real short subtle"
        elif float(shift[i]) >= shift_cut:
            labels[i] = "real level shift"
        elif int(span[i]) < medium:
            labels[i] = "real medium event"
        else:
            labels[i] = "real long campaign"
    return labels


def window_shape_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window amplitude and |μ_L − μ_R| (level-shift overlay)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z
    half = x.shape[1] // 2
    amp = x.max(axis=1) - x.min(axis=1)
    dmu = np.abs(x[:, :half].mean(axis=1) - x[:, half:].mean(axis=1))
    return amp, dmu


def kind_stat(x: np.ndarray, kind: str) -> np.ndarray:
    """Sort key for 'best attempt' at a kind. Higher = more like that rule."""
    amp, dmu = window_shape_stats(x)
    name = str(kind)
    if "level shift" in name:
        return dmu
    if "subtle" in name or "local subsequence" in name:
        return -amp
    return amp


def rank_for_kind(x: np.ndarray, kind: str, n: int) -> np.ndarray:
    """Indices of the ``n`` windows with the largest ``kind_stat``."""
    x = np.asarray(x)
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    n = min(int(n), len(x))
    order = np.argsort(-kind_stat(x, kind))
    return order[:n].astype(int)


def kind_ref_indices(
    kinds: np.ndarray,
    fold: np.ndarray,
    kind: str,
    *,
    query_fold: int = 0,
    max_n: int | None = 256,
) -> tuple[np.ndarray, bool]:
    """Indices for kind-conditional proto refs.

    Prefer event-OOF (``fold != query_fold``). If that set is empty — as with
    all six Mission-1 41–46 level shifts, which sit in fold 0 — fall back to
    every window of that kind and set ``leaked=True``.
    """
    kinds = np.asarray(kinds)
    fold = np.asarray(fold)
    oof = np.flatnonzero((kinds == kind) & (fold != int(query_fold)))
    leaked = False
    idx = oof
    if len(idx) == 0:
        idx = np.flatnonzero(kinds == kind)
        leaked = len(idx) > 0
    if max_n is not None and len(idx) > int(max_n):
        idx = idx[: int(max_n)]
    return idx.astype(int), leaked


def pick_kind_examples(
    x: np.ndarray,
    kinds: np.ndarray,
    event_id: np.ndarray,
    *,
    n: int = 4,
    order: tuple[str, ...] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Typical windows per kind, preferring distinct event IDs over max range."""
    x = np.asarray(x)
    kinds = np.asarray(kinds)
    event_id = np.asarray(event_id)
    amp = x.max(axis=1) - x.min(axis=1) if len(x) else np.zeros(0)
    seen = [str(k) for k in dict.fromkeys(kinds.tolist())] if len(kinds) else []
    pref = list(order or (list(KIND_ORDER) + [k for k in MORPH_KIND_ORDER if k not in KIND_ORDER]))
    walk = [k for k in pref if k in seen] + [k for k in seen if k not in pref]
    out: dict[str, dict[str, np.ndarray]] = {}
    for kind in walk:
        idx = np.flatnonzero(kinds == kind)
        if len(idx) == 0:
            continue
        med = float(np.median(amp[idx]))
        order = idx[np.argsort(np.abs(amp[idx] - med))]
        chosen: list[int] = []
        used: set[str] = set()
        for j in order:
            ev = str(event_id[j])
            if ev in used:
                continue
            chosen.append(int(j))
            used.add(ev)
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            for j in order:
                if int(j) not in chosen:
                    chosen.append(int(j))
                if len(chosen) >= n:
                    break
        sel = np.asarray(chosen, dtype=int)
        out[kind] = {"idx": sel, "x": x[sel], "event_id": event_id[sel]}
    return out


def nearest_generated(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """One unused gallery window nearest each query in feature_pack_v1."""
    q = np.asarray(queries)
    g = np.asarray(gallery)
    if len(q) == 0 or len(g) == 0:
        return g[:0]
    phi_q = embed_windows(q)
    phi_g = embed_windows(g)
    mu = phi_g.mean(axis=0)
    sd = np.where(phi_g.std(axis=0) < 1e-8, 1.0, phi_g.std(axis=0))
    zq = (phi_q - mu) / sd
    zg = (phi_g - mu) / sd
    taken: set[int] = set()
    picked: list[int] = []
    for i in range(len(q)):
        d = np.linalg.norm(zg - zq[i], axis=1)
        for j in taken:
            d[j] = np.inf
        j = int(np.argmin(d))
        taken.add(j)
        picked.append(j)
    return g[np.asarray(picked, dtype=int)]


def diverse_idx(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point sample in feature_pack_v1 (not max amplitude)."""
    x = np.asarray(x)
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    n = min(int(n), len(x))
    phi = embed_windows(x)
    sd = np.where(phi.std(axis=0) < 1e-8, 1.0, phi.std(axis=0))
    z = (phi - phi.mean(axis=0)) / sd
    start = int(rng.integers(len(x)))
    chosen = [start]
    dmin = np.linalg.norm(z - z[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(z - z[nxt], axis=1))
    return np.asarray(chosen, dtype=int)
