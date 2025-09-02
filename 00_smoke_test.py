#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_smoke_test.py
Robust loader for RCT_master_dataset_both_monkeys.mat (v7.3/HDF5).
- Normalizes data_master.Bhv into list of session dicts
- Builds per-session trial table
- Re-parameterizes direction D → (C,R)
- Optionally bins spikes for one area
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Any, Tuple

# ---------------------------------------------------------------------
# Loader for MATLAB v7.3 (HDF5)
# ---------------------------------------------------------------------
def _h5_to_py(obj, f):
    import h5py as _h5
    if isinstance(obj, _h5.Dataset):
        data = obj[()]
        if isinstance(data, bytes):
            return data.decode("utf8")
        if hasattr(data, 'dtype') and data.dtype == 'O':
            result = []
            for item in data.flat:
                if isinstance(item, _h5.h5r.Reference):
                    result.append(_h5_to_py(f[item], f))
                else:
                    result.append(item)
            return result
        return data
    elif isinstance(obj, _h5.Group):
        return {k: _h5_to_py(v, f) for k, v in obj.items()}
    elif isinstance(obj, _h5.h5r.Reference):
        return _h5_to_py(f[obj], f)
    else:
        return obj

def load_data_master(path: Path) -> Dict[str, Any]:
    with h5py.File(path.as_posix(), "r") as f:
        return _h5_to_py(f["data_master"], f)

# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------
def _scalarize(val):
    if isinstance(val, np.ndarray):
        flat = val.ravel()
        if flat.size >= 1:
            return flat[0].item()
    elif isinstance(val, list) and len(val) > 0:
        # Recursively scalarize the first element if it's a list
        return _scalarize(val[0])
    return val

def normalize_sessions(Bhv):
    if isinstance(Bhv, list):
        return Bhv
    if isinstance(Bhv, dict):
        if "session_id" in Bhv:
            return [Bhv]
        return [Bhv[k] for k in Bhv.keys()]
    if isinstance(Bhv, np.ndarray):
        return [normalize_sessions(x) if not isinstance(x, dict) else x for x in Bhv.tolist()]
    raise TypeError(f"Unexpected Bhv type: {type(Bhv)}")

def list_sessions(dm: Dict[str, Any]) -> List[int]:
    sessions = normalize_sessions(dm["Bhv"])
    out = []
    for s in sessions:
        val = _scalarize(s["session_id"])
        out.append(int(val))
    return out

def get_session(dm: Dict[str, Any], session_id: int) -> Dict[str, Any]:
    sessions = normalize_sessions(dm["Bhv"])
    for s in sessions:
        val = _scalarize(s["session_id"])
        if int(val) == int(session_id):
            return s
    raise ValueError(f"Session {session_id} not found")

# ---------------------------------------------------------------------
# Trial & neuron helpers
# ---------------------------------------------------------------------
def parse_date_from_neuron_id(neuron_id: str) -> int:
    return int(str(neuron_id).split("_")[0])

def iter_trials(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Trial_info into list of dicts."""
    T = session["Trial_info"]

    # Already list of dicts
    if isinstance(T, list) and all(isinstance(x, dict) for x in T):
        return T

    # Numpy array
    if isinstance(T, np.ndarray):
        return [x if isinstance(x, dict) else dict(x) if hasattr(x, "keys") else {} for x in T.tolist()]

    # List of lists
    if isinstance(T, list):
        out = []
        for x in T:
            if isinstance(x, dict):
                out.append(x)
            elif isinstance(x, list) and len(x) == 2:
                # sometimes [(field, value)] pairs
                try:
                    out.append(dict([x]))
                except Exception:
                    pass
        return out

    raise TypeError(f"Unexpected Trial_info type: {type(T)}")

def collect_session_areas(dm: Dict[str, Any], session_id: int) -> Dict[str, List[Dict[str, Any]]]:
    out = {"FEF": [], "LIP": [], "SC": []}
    for k in dm["Neuro"].keys():
        units = dm["Neuro"][k]
        if isinstance(units, dict):
            units = [units]
        mask = [parse_date_from_neuron_id(u["NeuronID"]) == session_id for u in units]
        sel = [u for u, m in zip(units, mask) if m]
        if not sel:
            continue
        if k.endswith("FEF"):
            out["FEF"].extend(sel)
        elif k.endswith("LIP"):
            out["LIP"].extend(sel)
        elif k.endswith("SC"):
            out["SC"].extend(sel)
    return out

# ---------------------------------------------------------------------
# Trial table with D → (C,R)
# ---------------------------------------------------------------------
def build_trial_table(session: Dict[str, Any], align_field: str = "Align_to_cat_stim_on") -> pd.DataFrame:
    rows = []
    for t in iter_trials(session):
        if not isinstance(t, dict):
            continue
        rows.append({
            "direction": float(t.get("direction", np.nan)),
            "category": float(t.get("category", np.nan)),
            "targets_vert": float(t.get("targets_vert", np.nan)),
            "trial_error": int(t.get("trial_error", 0)) if "trial_error" in t else 0,
            "align_ts": float(t.get(align_field, np.nan)),
            "sacc_on": float(t.get("Align_to_sacc_on", np.nan)),
        })
    df = pd.DataFrame(rows)
    df = df[~df["category"].isna()]
    df = df[df["trial_error"] == 0]
    df = df[~df["align_ts"].isna()].reset_index(drop=True)

    df["C"] = df["category"].astype(int)
    mapping = {}
    for Cval in [-1, 1]:
        dirs = sorted(np.unique(df.loc[df["C"] == Cval, "direction"]).tolist())
        for idx, d in enumerate(dirs, start=1):
            mapping[(Cval, d)] = idx
    df["R"] = [mapping.get((int(c), float(d)), np.nan) for c, d in zip(df["C"], df["direction"])]
    df["R"] = df["R"].astype("Int64")
    df["RT"] = df["sacc_on"] - df["align_ts"]
    return df

# ---------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------
def bin_area(units: List[Dict[str, Any]],
             align_ts: np.ndarray,
             window: Tuple[float, float] = (-0.3, 1.2),
             bin_size: float = 0.010,
             fs_spk: float = 40000.0) -> np.ndarray:
    nT = len(align_ts)
    nU = len(units)
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    nB = edges.size - 1
    X = np.zeros((nT, nB, nU), dtype=np.int16)
    spikes_s = [np.asarray(u.get("NeuronSpkT", []), dtype=float) / fs_spk for u in units]
    for u, st in enumerate(spikes_s):
        if st.size == 0:
            continue
        for i, ev in enumerate(align_ts):
            c, _ = np.histogram(st - ev, bins=edges)
            X[i, :, u] = c
    return X

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=Path, required=True)
    ap.add_argument("--session", type=int, default=None)
    ap.add_argument("--area", type=str, default="FEF", choices=["FEF", "LIP", "SC"])
    ap.add_argument("--do_bin", action="store_true")
    args = ap.parse_args()

    dm = load_data_master(args.mat)
    sessions = sorted(list_sessions(dm))
    sid = args.session or sessions[0]
    print(f"[info] sessions found: {len(sessions)}. Using session {sid}.")

    ses = get_session(dm, sid)
    df = build_trial_table(ses)

    print(f"[ok] RCT-correct trials with align event: {len(df)}")
    counts = df.groupby(["C", "R"]).size().unstack(fill_value=0)
    print("[counts] trials per (C,R):\n", counts)

    areas = collect_session_areas(dm, sid)
    for a in ["LIP", "FEF", "SC"]:
        print(f"[units] {a}: {len(areas[a])} neurons")

    if args.do_bin:
        area_units = areas[args.area]
        if len(area_units) == 0:
            print(f"[warn] No units for {args.area} in session {sid}")
            return
        X = bin_area(area_units, df["align_ts"].to_numpy(),
                     window=(-0.3, 1.2), bin_size=0.010, fs_spk=40000.0)
        print(f"[ok] Binned {args.area}: shape {X.shape} = (trials, bins, units)")

if __name__ == "__main__":
    main()
