"""Scout ESA channels for a transfer panel: continuous, dense, labeled.

Does not train. Does not read past official_train_end. Isolated print + JSON.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CD = Path("/mnt/extras/SSD/AI/CausalDiscovery")
W = 512
RNG = np.random.default_rng(0)


def naive_utc(values):
    s = pd.to_datetime(values)
    tz = getattr(s.dtype, "tz", None)
    if tz is not None:
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def load_events(root: Path, cut: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = pd.read_csv(root / "labels.csv")
    typ = pd.read_csv(root / "anomaly_types.csv")
    m = lab.merge(typ, on="ID", how="left")
    m["EndTime"] = naive_utc(m["EndTime"])
    m["StartTime"] = naive_utc(m["StartTime"])
    cut_t = pd.Timestamp(cut)
    pre = m[m["EndTime"] < cut_t].copy()
    return m, pre


def label_counts(pre: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ch, g in pre.groupby("Channel"):
        rows.append(
            {
                "channel": ch,
                "n_anom_events": int(g.loc[g["Category"] == "Anomaly", "ID"].nunique()),
                "n_rare_events": int(g.loc[g["Category"] == "Rare Event", "ID"].nunique()),
                "n_anom_spans": int((g["Category"] == "Anomaly").sum()),
            }
        )
    return pd.DataFrame(rows)


def window_stats(y: np.ndarray, counts: np.ndarray | None, n_probe: int = 48) -> dict:
    y = np.asarray(y, dtype=np.float32)
    T = len(y)
    if counts is None:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y) & (np.asarray(counts) > 0)
    finite_frac = float(ok.mean()) if T else 0.0
    n_win = 0
    if T >= W:
        # rolling all-finite via prefix
        c = np.cumsum(ok.astype(np.int32))
        c = np.concatenate([[0], c])
        # start s valid if c[s+W]-c[s] == W
        span = c[W:] - c[:-W]
        good = np.flatnonzero(span == W)
        n_win = int(len(good))
    uniques = []
    snap = []
    if n_win > 0:
        pick = good[RNG.choice(len(good), size=min(n_probe, len(good)), replace=False)]
        for s in pick:
            w = y[s : s + W]
            u = np.unique(np.round(w, 6))
            uniques.append(int(len(u)))
            if len(u) <= 32:
                # snap to nearest unique of this window
                d = np.min(np.abs(w[:, None] - u[None, :]), axis=1)
                gap = float(np.min(np.diff(np.sort(u)))) if len(u) > 1 else 1e-6
                snap.append(float(np.mean(d <= 0.25 * max(gap, 1e-9))))
            else:
                snap.append(0.0)
    finite = y[ok]
    n_unique_all = int(len(np.unique(np.round(finite, 6)))) if finite.size else 0
    return {
        "finite_frac": finite_frac,
        "n_finite_W512": n_win,
        "median_unique_per_win": float(np.median(uniques)) if uniques else 0.0,
        "mean_snap": float(np.mean(snap)) if snap else 0.0,
        "n_unique_all_r6": n_unique_all,
        "std": float(np.nanstd(y)) if y.size else 0.0,
        "ptp": float(np.nanmax(y) - np.nanmin(y)) if finite.size else 0.0,
    }


def scout_m2_panel() -> pd.DataFrame:
    meta = json.loads((CD / "results/stage_g1/panel_meta.json").read_text())
    blob = np.load(CD / "results/stage_g1/panel.npz", allow_pickle=False)
    Y = np.asarray(blob["Y"])
    counts = np.asarray(blob["counts"])
    chs = list(meta["channels"])
    ch_meta = pd.read_csv(CD / "data/ESA-Mission2/channels.csv")
    _, pre = load_events(CD / "data/ESA-Mission2", "2001-10-01")
    lab = label_counts(pre)
    rows = []
    for i, ch in enumerate(chs):
        print(f"scout M2 panel {ch}", flush=True)
        st = window_stats(Y[:, i], counts[:, i])
        meta_row = ch_meta[ch_meta["Channel"] == ch]
        lab_row = lab[lab["channel"] == ch]
        rows.append(
            {
                "mission": "M2",
                "channel": ch,
                "subsystem": meta_row["Subsystem"].iloc[0] if len(meta_row) else "",
                "unit": meta_row["Physical Unit"].iloc[0] if len(meta_row) else "",
                "group": int(meta_row["Group"].iloc[0]) if len(meta_row) else -1,
                "target": meta_row["Target"].iloc[0] if len(meta_row) else "",
                "categorical": meta_row["Categorical"].iloc[0] if len(meta_row) else "",
                "n_anom_events": int(lab_row["n_anom_events"].iloc[0]) if len(lab_row) else 0,
                "n_rare_events": int(lab_row["n_rare_events"].iloc[0]) if len(lab_row) else 0,
                **st,
            }
        )
    return pd.DataFrame(rows)


def _read_esa_channel(path: Path, name: str) -> pd.Series:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                obj = pd.read_pickle(fh)
    else:
        obj = pd.read_pickle(path)
    if isinstance(obj, pd.Series):
        s = obj
    else:
        col = name if name in obj.columns else obj.columns[0]
        s = obj[col]
    s.index = naive_utc(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype("float64")


def scout_m1_pickles(channels: list[str]) -> pd.DataFrame:
    ch_meta = pd.read_csv(CD / "data/ESA-Mission1/channels.csv")
    _, pre = load_events(CD / "data/ESA-Mission1", "2007-01-01")
    lab = label_counts(pre)
    cut = pd.Timestamp("2007-01-01")
    rows = []
    for ch in channels:
        print(f"scout M1 pickle {ch}", flush=True)
        path = CD / "data/ESA-Mission1/channels" / f"{ch}.zip"
        if not path.is_file():
            alt = path.with_suffix(".pkl")
            path = alt if alt.is_file() else path
        if not path.is_file():
            rows.append({"mission": "M1", "channel": ch, "missing": True})
            continue
        s = _read_esa_channel(path, ch)
        s = s.loc[s.index < cut]
        if s.empty:
            rows.append({"mission": "M1", "channel": ch, "empty": True})
            continue
        t0 = s.index.min().floor("30s")
        grid = pd.date_range(t0, cut, freq="30s", inclusive="left")
        binned = s.resample("30s", origin=t0).mean()
        cnt = s.resample("30s", origin=t0).count()
        y = binned.reindex(grid).to_numpy(dtype=np.float32)
        c = cnt.reindex(grid).fillna(0).to_numpy(dtype=np.int16)
        del s, binned, cnt
        st = window_stats(y, c)
        meta_row = ch_meta[ch_meta["Channel"] == ch]
        lab_row = lab[lab["channel"] == ch]
        rows.append(
            {
                "mission": "M1",
                "channel": ch,
                "subsystem": meta_row["Subsystem"].iloc[0] if len(meta_row) else "",
                "unit": meta_row["Physical Unit"].iloc[0] if len(meta_row) else "",
                "group": int(meta_row["Group"].iloc[0]) if len(meta_row) else -1,
                "target": meta_row["Target"].iloc[0] if len(meta_row) else "",
                "categorical": meta_row["Categorical"].iloc[0] if len(meta_row) else "",
                "n_anom_events": int(lab_row["n_anom_events"].iloc[0]) if len(lab_row) else 0,
                "n_rare_events": int(lab_row["n_rare_events"].iloc[0]) if len(lab_row) else 0,
                **st,
            }
        )
        del y, c
    return pd.DataFrame(rows)


def main() -> None:
    out = Path("/mnt/extras/SSD/AI/AnoGen/results/xfer/scout")
    out.mkdir(parents=True, exist_ok=True)

    m2 = scout_m2_panel()
    m2.to_csv(out / "mission2_g1.csv", index=False)

    m1_meta = pd.read_csv(CD / "data/ESA-Mission1/channels.csv")
    _, pre = load_events(CD / "data/ESA-Mission1", "2007-01-01")
    lab = label_counts(pre)
    used = {
        "channel_41",
        "channel_42",
        "channel_43",
        "channel_44",
        "channel_45",
        "channel_46",
        "channel_47",
        "channel_48",
        "channel_49",
        "channel_73",
        "channel_74",
        "channel_75",
    }
    analog = m1_meta[(m1_meta["Target"] == "YES") & (m1_meta["Categorical"] == "NO")]
    merged = analog.merge(lab, left_on="Channel", right_on="channel", how="left").fillna(0)
    # enough labels or same-family unused (61-63, 70-72, 76, 50-52, 57-66)
    cand = merged[
        (~merged["Channel"].isin(used))
        & ((merged["n_anom_events"] >= 3) | (merged["Group"].isin([8, 14, 18, 10, 13, 15, 16])))
    ]["Channel"].tolist()
    # always include locked 41-46 as a continuous reference (already binned? skip pickles)
    print(f"M1 candidates ({len(cand)}): {cand}", flush=True)
    m1 = scout_m1_pickles(cand)
    m1.to_csv(out / "mission1_unused.csv", index=False)

    both = pd.concat([m2, m1], ignore_index=True)
    both.to_csv(out / "all.csv", index=False)
    print(both.to_string(index=False))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
