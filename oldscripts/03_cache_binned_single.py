#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_cache_binned.py
- Reads exported RCT/ (manifest.json + trials.parquet)
- Filters: RCT-correct + chosen layout (default: vertical only)
- Re-parameterizes D -> (C,R)
- Bins spikes per session×area into (trials, bins, units)
- Saves: results/caches/<sid>_<AREA>.npz with:
    X (int16 counts), meta (json str), trials (pandas JSON)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import h5py

VALID_AREAS = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

# ---------- IO ----------
def read_manifest(root: Path) -> Dict[str, Any]:
    m = root / "manifest.json"
    if not m.exists():
        raise FileNotFoundError(f"manifest.json not found at {m}")
    with open(m, "r") as f:
        return json.load(f)

def trials_table(sdir: Path) -> pd.DataFrame:
    pq = sdir / "trials.parquet"
    if pq.exists(): return pd.read_parquet(pq)
    csv = sdir / "trials.csv"
    if csv.exists(): return pd.read_csv(csv)
    raise FileNotFoundError(f"No trials.parquet/csv in {sdir}")

def reparam_CR(df: pd.DataFrame, targets_vert_only: bool) -> pd.DataFrame:
    need = ["category","direction","trial_error","Align_to_cat_stim_on",
            "Align_to_sacc_on","targets_vert","block_number"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan

    x = df.copy()
    # RCT + correct + has alignment
    x = x[~x["category"].isna()]
    x = x[(x["trial_error"].fillna(0) == 0)]
    x = x[~x["Align_to_cat_stim_on"].isna()]

    # Layout filter
    if targets_vert_only:
        x = x[(x["targets_vert"].fillna(0) == 1)]

    x = x.reset_index(drop=True)
    x = x.rename(columns={"Align_to_cat_stim_on":"align_ts"})
    x["C"] = x["category"].astype(int)

    # Session-specific mapping of 3 dirs within each category → R={1,2,3}
    mapping = {}
    for Cval in (-1, 1):
        dirs = sorted(np.unique(x.loc[x["C"] == Cval, "direction"]).tolist())
        if len(dirs) not in (0, 3):
            raise AssertionError(f"Expected 3 directions for category {Cval}, got {dirs}")
        for idx, d in enumerate(dirs, start=1):
            mapping[(Cval, float(d))] = idx
    x["R"] = [mapping.get((int(c), float(d)), np.nan) for c, d in zip(x["C"], x["direction"])]
    x["R"] = x["R"].astype("Int64")

    x["RT"] = x["Align_to_sacc_on"] - x["align_ts"]
    return x

def read_units(area_dir: Path) -> pd.DataFrame:
    upath = area_dir / "units.json"
    if not upath.exists():
        raise FileNotFoundError(f"units.json not found at {upath}")
    with open(upath, "r") as f:
        return pd.DataFrame(json.load(f))

def load_spikes(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h:
        return np.asarray(h["/t"][()], dtype=np.float32)  # seconds

def bin_area(area_dir: Path, trials: pd.DataFrame,
             window: Tuple[float, float],
             bin_size: float) -> np.ndarray:
    spikes_dir = area_dir / "spikes"
    U = read_units(area_dir).reset_index(drop=True)
    nU = len(U)
    if nU == 0:
        raise RuntimeError(f"No units found under {area_dir}")
    align = trials["align_ts"].to_numpy(float)
    nT = align.size

    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    nB = edges.size - 1
    X = np.zeros((nT, nB, nU), dtype=np.int16)

    # Load spike vectors (already seconds)
    ST: List[np.ndarray] = []
    for _, r in U.iterrows():
        f = (area_dir / r["file"]) if not str(r["file"]).startswith("/") else Path(r["file"])
        if not f.exists():
            f = spikes_dir / Path(str(r["file"])).name
        ST.append(load_spikes(f))

    # Histogram per trial, per unit
    for u, st in enumerate(ST):
        if st.size == 0:
            continue
        for i, ev in enumerate(align):
            c, _ = np.histogram(st - ev, bins=edges)
            X[i, :, u] = c
    return X

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to RCT/ (has manifest.json)")
    ap.add_argument("--session", type=int, default=None, help="YYYYMMDD; default = first in manifest")
    ap.add_argument(
        "--areas", nargs="*", default=None,
        help="Actual area folders to cache (MFEF/MLIP/MSC/SFEF/SLIP/SSC). "
             "If omitted, caches whichever of these are present in the session."
    )
    ap.add_argument("--targets_vert_only", action="store_true", default=True,
                    help="Keep only vertical target layout trials (targets_vert==1). Default True.")
    ap.add_argument("--bin", type=float, default=0.010, help="Bin size (s)")
    ap.add_argument("--t0",  type=float, default=-0.25, help="Window start (s)")
    ap.add_argument("--t1",  type=float, default= 0.80, help="Window end (s)")
    args = ap.parse_args()

    root: Path = args.root
    man = read_manifest(root)
    sessions = sorted(int(s["session_id"]) for s in man["sessions"])
    if not sessions:
        raise SystemExit("No sessions in manifest.")
    sid = args.session or sessions[0]
    sdir = root / str(sid)

    areas_present_all = next(s for s in man["sessions"] if int(s["session_id"]) == sid).get("areas", [])
    present_actual = [a for a in areas_present_all if a in VALID_AREAS]

    # Choose targets:
    targets = present_actual if (not args.areas) else [a for a in args.areas if a in present_actual]
    if not targets:
        print(f"[skip] session {sid}: no requested valid areas present (had {areas_present_all})")
        return

    # Trials & labels (layout filtered)
    df = reparam_CR(trials_table(sdir), targets_vert_only=args.targets_vert_only)

    # Output dir
    out_dir = Path("results/caches"); out_dir.mkdir(parents=True, exist_ok=True)

    # Cache each requested actual area
    for a in targets:  # e.g., 'MFEF' or 'SFEF'
        adir = sdir / "areas" / a
        X = bin_area(adir, df, window=(args.t0, args.t1), bin_size=args.bin)
        meta = {
            "session": int(sid),
            "area_folder": a,
            "shape": {"trials": int(X.shape[0]), "bins": int(X.shape[1]), "units": int(X.shape[2])},
            "bin_size_s": float(args.bin),
            "window_s": [args.t0, args.t1],
            "targets_vert_only": bool(args.targets_vert_only),
        }
        out_path = out_dir / f"{sid}_{a}.npz"
        np.savez_compressed(out_path, X=X, meta=json.dumps(meta), trials=df.to_json(orient="records"))
        print(f"[ok] cached {out_path.name}  shape={X.shape}")
    print("[done]")

if __name__ == "__main__":
    main()