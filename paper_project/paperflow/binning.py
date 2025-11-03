# paper_project/paperflow/binning.py
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple, Optional
import numpy as np, pandas as pd, h5py

VALID_AREAS = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def _time_edges(t0: float, t1: float, bin_s: float) -> np.ndarray:
    n = int(np.round((t1 - t0) / bin_s))
    edges = t0 + np.arange(n + 1) * bin_s
    if edges[-1] < t1 - 1e-12:
        edges = np.append(edges, t1)
    return edges

def _bin_one_unit(spike_sec: np.ndarray, event_sec: np.ndarray, edges: np.ndarray, t0: float, t1: float) -> np.ndarray:
    nT = event_sec.shape[0]; nB = edges.shape[0] - 1
    out = np.zeros((nT, nB), dtype=np.float32)
    for i, ev in enumerate(event_sec):
        if spike_sec.size == 0: continue
        mask = (spike_sec >= ev + t0) & (spike_sec < ev + t1)
        if not np.any(mask): continue
        rel = spike_sec[mask] - ev
        out[i, :] = np.histogram(rel, bins=edges)[0]
    return out

def _zscore_units(X: np.ndarray) -> np.ndarray:
    # X: (trials, bins, units)
    Xf = X.reshape(-1, X.shape[-1]).astype(np.float64)
    mu = np.nanmean(Xf, axis=0)
    sd = np.nanstd(Xf, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return Z.astype(np.float32)

def _area_dirs(root: str, sid: str, area: str) -> Tuple[str, str, str]:
    sess_dir = os.path.join(root, sid)
    area_dir = os.path.join(sess_dir, "areas", area)
    spikes_dir = os.path.join(area_dir, "spikes")
    return sess_dir, area_dir, spikes_dir

def _read_units_json(area_dir: str) -> List[Dict]:
    jpath = os.path.join(area_dir, "units.json")
    if not os.path.exists(jpath): return []
    return json.load(open(jpath, "r"))

def _read_spike_times(spikes_dir: str, rel_file: str) -> np.ndarray:
    fpath = os.path.join(spikes_dir, os.path.basename(rel_file))
    with h5py.File(fpath, "r") as h5:
        return h5["/t"][:].astype("float64")  # seconds

def _read_trials(root: str, sid: str) -> pd.DataFrame:
    p = os.path.join(root, sid, "trials.parquet")
    if os.path.exists(p): return pd.read_parquet(p)
    p = os.path.join(root, sid, "trials.csv")
    if os.path.exists(p): return pd.read_csv(p)
    raise FileNotFoundError(f"No trials.parquet/csv for sid {sid}")

def build_cache_for_session(
    root: str, sid: str, align: str,
    t0: float, t1: float, bin_s: float,
    out_root: str, correct_only: bool = True
) -> List[str]:
    """
    Build caches for all areas in <sid>, aligned to 'stim' or 'sacc'.
    Writes: <out_root>/<align>/<sid>/caches/area_<AREA>.npz
    Returns list of saved paths.
    """
    # trials & labels
    df = _read_trials(root, sid)
    # keep RCT trials with timestamps we need
    if align == "sacc":
        df = df[~df["Align_to_sacc_on"].isna()]
        event = df["Align_to_sacc_on"].to_numpy(float)
    elif align == "stim":
        df = df[~df["Align_to_cat_stim_on"].isna()]
        event = df["Align_to_cat_stim_on"].to_numpy(float)
    else:
        raise ValueError("align must be 'stim' or 'sacc'")

    if correct_only and "trial_error" in df.columns:
        df = df[df["trial_error"].fillna(0) == 0]
        if align == "sacc":
            event = df["Align_to_sacc_on"].to_numpy(float)
        else:
            event = df["Align_to_cat_stim_on"].to_numpy(float)

    nT = len(df)
    if nT == 0: return []

    # labels we store (present-or-NaN ok)
    def col(name, default=np.nan):
        return df[name].to_numpy(float) if name in df.columns else np.full(nT, default, float)
    C  = col("category")
    R  = col("direction")  # we can map to {1,2,3} later if needed
    S  = col("saccade_location_sign")
    OR = np.where(df.get("targets_vert", pd.Series([np.nan]*nT)).to_numpy(float) == 1, "vertical", "horizontal").astype(object)
    PT = col("PT_ms")
    IC = (df.get("trial_error", pd.Series([0]*nT)).fillna(0).to_numpy(int) == 0)

    edges = _time_edges(t0, t1, bin_s)
    out_paths = []

    # loop areas
    sess_dir = os.path.join(root, sid)
    areas_root = os.path.join(sess_dir, "areas")
    areas = [a for a in os.listdir(areas_root) if os.path.isdir(os.path.join(areas_root, a)) and a in VALID_AREAS] if os.path.isdir(areas_root) else []
    if not areas: return []

    for area in sorted(areas):
        _, area_dir, spikes_dir = _area_dirs(root, sid, area)
        units = _read_units_json(area_dir)
        if not units: continue

        # bin per unit
        mats = []
        for u in units:
            spk = _read_spike_times(spikes_dir, u["file"])
            mats.append(_bin_one_unit(spk, event, edges, t0, t1))  # (T,B)
        X = np.stack(mats, axis=-1).astype(np.float32)  # (T,B,U)
        Z = _zscore_units(X)

        meta = {
            "sid": sid, "area": area, "align_event": align,
            "window": [float(t0), float(t1)], "bin_s": float(bin_s),
            "n_trials": int(nT), "n_units": int(X.shape[-1])
        }

        out_dir = os.path.join(out_root, align, sid, "caches")
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, f"area_{area}.npz")
        np.savez_compressed(
            out_npz,
            X=X, Z=Z, time=((edges[:-1] + edges[1:]) / 2.0).astype(np.float32),
            lab_C=C, lab_R=R, lab_S=S, lab_orientation=OR,
            lab_PT_ms=PT, lab_is_correct=IC, lab_trial_index=df.get("trial_index", pd.Series(range(nT))).to_numpy(int),
            meta=json.dumps(meta)
        )
        out_paths.append(out_npz)
    return out_paths
