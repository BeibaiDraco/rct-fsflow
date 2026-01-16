#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_h5_smoke.py

Minimal dataset sanity check for the exported RCT layout:

- Reads RCT/manifest.json
- Loads one session's trials.parquet (fallback trials.csv)
- Re-parameterizes direction D into (C, R)
- Optionally bins one area's spikes into (trials, bins, units)
- Writes a tiny JSON summary under results/

Run (interactive):
  python 01_h5_smoke.py --root RCT --do-bin --area FEF

Tip: default root is BASE/RCT if BASE env is set.
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import h5py
from datetime import datetime


# ---------- IO helpers ----------
def read_manifest(root: Path) -> Dict[str, Any]:
    mpath = root / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"manifest.json not found at {mpath}")
    with open(mpath, "r") as f:
        return json.load(f)

def trials_table(session_dir: Path) -> pd.DataFrame:
    pq = session_dir / "trials.parquet"
    csv = session_dir / "trials.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No trials.parquet/csv in {session_dir}")

def read_units(area_dir: Path) -> pd.DataFrame:
    upath = area_dir / "units.json"
    if not upath.exists():
        raise FileNotFoundError(f"units.json not found at {upath}")
    with open(upath, "r") as f:
        U = json.load(f)
    return pd.DataFrame(U)

def load_unit_spikes(spike_file: Path) -> np.ndarray:
    with h5py.File(spike_file, "r") as h:
        t = h["/t"][()]  # seconds
    return np.asarray(t, dtype=np.float32)


# ---------- D -> (C, R) ----------
def reparameterize_CR(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = ["category", "direction", "trial_error", "Align_to_cat_stim_on", "Align_to_sacc_on"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    x = df.copy()
    # RCT + correct + has alignment
    x = x[~x["category"].isna()]
    x = x[(x["trial_error"].fillna(0) == 0)]
    x = x[~x["Align_to_cat_stim_on"].isna()].reset_index(drop=True)

    # Category ±1
    x["C"] = x["category"].astype(int)

    # Per-category sorted directions -> R ∈ {1,2,3}
    mapping = {}
    for Cval in (-1, 1):
        dirs = sorted(np.unique(x.loc[x["C"] == Cval, "direction"]).tolist())
        if len(dirs) not in (0, 3):
            raise AssertionError(f"Expected 3 directions for category {Cval}, got {dirs}")
        for idx, d in enumerate(dirs, start=1):
            mapping[(Cval, float(d))] = idx

    x["R"] = [mapping.get((int(c), float(d)), np.nan)
              for c, d in zip(x["C"], x["direction"])]
    x["R"] = x["R"].astype("Int64")

    x = x.rename(columns={"Align_to_cat_stim_on": "align_ts"})
    x["RT"] = x["Align_to_sacc_on"] - x["align_ts"]
    return x


# ---------- Binning ----------
def bin_area(area_dir: Path,
             trials: pd.DataFrame,
             window: Tuple[float, float] = (-0.3, 1.2),
             bin_size: float = 0.010) -> np.ndarray:
    """
    Returns X with shape (n_trials, n_bins, n_units).
    """
    spikes_dir = area_dir / "spikes"
    units = read_units(area_dir)
    nU = len(units)
    if nU == 0:
        raise RuntimeError(f"No units in {area_dir}")
    align = trials["align_ts"].to_numpy(dtype=np.float64)
    nT = align.size

    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    nB = edges.size - 1
    X = np.zeros((nT, nB, nU), dtype=np.int16)

    # Load spike vectors
    spk_all: List[np.ndarray] = []
    for _, r in units.reset_index(drop=True).iterrows():
        # r["file"] usually "spikes/unit_XXX.h5"
        f = (area_dir / r["file"]) if not str(r["file"]).startswith("/") else Path(r["file"])
        if not f.exists():
            # try relative to spikes dir
            f = spikes_dir / Path(str(r["file"])).name
        t = load_unit_spikes(f)
        spk_all.append(t)

    for u, st in enumerate(spk_all):
        if st.size == 0:
            continue
        for i, ev in enumerate(align):
            c, _ = np.histogram(st - ev, bins=edges)
            X[i, :, u] = c
    return X


# ---------- CLI ----------
def main():
    default_root = None
    base_env = os.environ.get("BASE") or os.environ.get("RCT_BASE")
    if base_env:
        default_root = Path(base_env) / "RCT"

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root,
                    help="Path to RCT/ (contains manifest.json). Default: $BASE/RCT if set.")
    ap.add_argument("--session", type=int, default=None,
                    help="Session id YYYYMMDD; default = first in manifest.")
    ap.add_argument("--area", type=str, default="FEF", choices=["FEF", "LIP", "SC"],
                    help="Area to bin (generic name; script will pick M*/S* match).")
    ap.add_argument("--do-bin", action="store_true", help="Also bin spikes for selected area.")
    args = ap.parse_args()

    if args.root is None:
        raise SystemExit("Please provide --root or set BASE and ensure $BASE/RCT exists.")

    root = args.root.resolve()
    man = read_manifest(root)
    sessions = sorted(int(s["session_id"]) for s in man["sessions"])
    if not sessions:
        raise SystemExit("No sessions found in manifest.")
    sid = args.session or sessions[0]

    # Areas present for this session
    sess_entry = next(s for s in man["sessions"] if int(s["session_id"]) == sid)
    areas_present = sess_entry.get("areas", [])
    print(f"[info] sessions total: {len(sessions)} → using session {sid}")
    print(f"[info] areas present in {sid}: {areas_present}")

    sdir = root / str(sid)
    df_raw = trials_table(sdir)
    df = reparameterize_CR(df_raw)
    print(f"[ok] RCT-correct trials with align event: {len(df)}")
    # Build counts table and cast to pure-Python ints for JSON
    counts_tbl = df.groupby(["C", "R"]).size().unstack(fill_value=0).astype(int)
    print("[counts] trials per (C,R):\n", counts_tbl)

    # Convert to nested dict {C: {R: count}} with plain int keys
    counts_json = {
        int(C): {int(R): int(v) for R, v in row.items()}
        for C, row in counts_tbl.to_dict(orient="index").items()
    }

    summary = {
        "session": int(sid),
        "n_trials": int(len(df)),
        "counts_per_C_R": counts_json,
        "areas_present": [str(a) for a in areas_present],
    }

    if args.do_bin:
        # pick area folder that endswith chosen generic area
        area_name = next((a for a in areas_present if a.endswith(args.area)), None)
        if area_name is None:
            raise SystemExit(f"No area ending with {args.area} found in session {sid}.")
        area_dir = sdir / "areas" / area_name
        X = bin_area(area_dir, df, window=(-0.3, 1.2), bin_size=0.010)
        print(f"[ok] Binned {area_name}  → shape {X.shape}  (trials, bins, units)")
        summary["binned"] = {
            "area": area_name,
            "shape": {"trials": int(X.shape[0]), "bins": int(X.shape[1]), "units": int(X.shape[2])},
            "bin_size_s": 0.010,
            "window_s": [-0.3, 1.2],
        }

    # write summary
    base = Path(os.environ.get("BASE", "."))  # default to CWD if BASE unset
    out_dir = (base / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"quickcheck_{sid}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] Wrote summary → {out_path}")

if __name__ == "__main__":
    main()