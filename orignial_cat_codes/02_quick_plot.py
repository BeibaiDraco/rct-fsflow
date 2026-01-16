#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_quick_plot.py
Plot raster + PSTH for a single unit from the exported RCT dataset.
Saves a PNG under results/plots/.

Usage:
  python 02_quick_plot.py --root RCT --area FEF --session 20200217 --unit-index 0
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve
try:
    from scipy.signal.windows import gaussian
except ImportError:
    def gaussian(M, std):
        n = np.arange(0, M) - (M - 1.0) / 2.0
        return np.exp(-0.5 * (n / std) ** 2)

def read_manifest(root: Path) -> Dict[str, Any]:
    with open(root / "manifest.json", "r") as f:
        return json.load(f)

def trials_table(session_dir: Path) -> pd.DataFrame:
    pq = session_dir / "trials.parquet"
    if pq.exists(): return pd.read_parquet(pq)
    return pd.read_csv(session_dir / "trials.csv")

def reparameterize_CR(df: pd.DataFrame) -> pd.DataFrame:
    need = ["category","direction","trial_error","Align_to_cat_stim_on","Align_to_sacc_on","targets_vert"]
    for c in need:
        if c not in df.columns: df[c] = np.nan
    x = df.copy()
    x = x[~x["category"].isna()]
    x = x[(x["trial_error"].fillna(0) == 0)]
    x = x[~x["Align_to_cat_stim_on"].isna()].reset_index(drop=True)
    x = x.rename(columns={"Align_to_cat_stim_on":"align_ts"})
    x["C"] = x["category"].astype(int)
    # build R mapping
    mapping = {}
    for Cval in (-1,1):
        dirs = sorted(np.unique(x.loc[x["C"]==Cval,"direction"]).tolist())
        for idx,d in enumerate(dirs, start=1):
            mapping[(Cval, float(d))] = idx
    x["R"] = [mapping[(int(c), float(d))] for c,d in zip(x["C"], x["direction"])]
    x["R"] = x["R"].astype("Int64")
    return x

def load_unit_spikes(spike_path: Path) -> np.ndarray:
    with h5py.File(spike_path, "r") as h:
        return np.asarray(h["/t"][()], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to RCT/")
    ap.add_argument("--session", type=int, default=None, help="YYYYMMDD; default = first in manifest")
    ap.add_argument("--area", type=str, default="FEF", choices=["FEF","LIP","SC"])
    ap.add_argument("--unit-index", type=int, default=0, help="0-based index within units.json to plot")
    ap.add_argument("--bin", type=float, default=0.005, help="Bin size (s) for PSTH (default 5 ms)")
    ap.add_argument("--smooth_ms", type=float, default=15.0, help="Gaussian smoothing (ms) for PSTH")
    ap.add_argument("--window", type=float, nargs=2, default=[-0.3, 1.2], help="Window (s) around align")
    args = ap.parse_args()

    man = read_manifest(args.root)
    sessions = sorted(int(s["session_id"]) for s in man["sessions"])
    sid = args.session or sessions[0]
    areas_present = next(s for s in man["sessions"] if int(s["session_id"])==sid).get("areas", [])
    area_name = next((a for a in areas_present if a.endswith(args.area)), None)
    if area_name is None:
        raise SystemExit(f"No area ending with {args.area} found in session {sid}")

    sdir = args.root / str(sid)
    adir = sdir / "areas" / area_name

    # trials (filters + labels)
    df_raw = trials_table(sdir)
    df = reparameterize_CR(df_raw)
    align = df["align_ts"].to_numpy(dtype=np.float64)
    groups = df["C"].to_numpy(int)  # color by category for simplicity

    # pick unit & load spikes
    import json as _json
    units = pd.DataFrame(_json.load(open(adir/"units.json","r")))
    if not (0 <= args.unit_index < len(units)):
        raise SystemExit(f"--unit-index out of range (0..{len(units)-1})")
    rel = units.iloc[args.unit_index]
    spike_path = (adir / rel["file"]) if not str(rel["file"]).startswith("/") else Path(rel["file"])
    if not spike_path.exists():
        spike_path = adir / "spikes" / Path(str(rel["file"])).name
    st = load_unit_spikes(spike_path)

    # bin to BA matrix: [trials, bins]
    edges = np.arange(args.window[0], args.window[1] + args.bin, args.bin)
    t = 0.5*(edges[:-1]+edges[1:])
    BA = np.zeros((align.size, t.size), dtype=np.int16)
    for i,ev in enumerate(align):
        c,_ = np.histogram(st - ev, bins=edges)
        BA[i] = c

    # PSTH (Hz) + smoothing
    psth = BA.mean(axis=0) / args.bin
    sig = args.smooth_ms/1000.0/args.bin
    if sig > 1:
        w = gaussian(int(round(sig*6))|1, std=sig); w /= w.sum()
        psth = convolve(psth, w, mode="same")

    # Plot
    fig = plt.figure(figsize=(9,7))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t, psth)
    ax1.axvline(0, ls="--", color="k", lw=1)
    ax1.set_ylabel("FR (Hz)")
    ax1.set_title(f"{area_name} {sid}  unit_index={args.unit_index}  cid={rel.get('cluster_id','?')}")

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    # raster: draw tick per spike bin
    tr, b = np.where(BA>0)
    if b.size > 0:
        x = t[b]
        y = tr + 1
        ax2.vlines(x, y-0.4, y+0.4, linewidth=0.5)
    ax2.axvline(0, ls="--", color="k", lw=1)
    ax2.set_ylabel("trial")

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    # category-wise PSTH
    for cval in [-1,1]:
        m = (groups==cval)
        if m.any():
            p = BA[m].mean(axis=0)/args.bin
            if sig > 1:
                p = convolve(p, w, mode="same")
            ax3.plot(t, p, label=f"C={cval}")
    ax3.axvline(0, ls="--", color="k", lw=1)
    ax3.set_xlabel("time (s) from cat_stim_on")
    ax3.set_ylabel("FR (Hz)")
    ax3.legend(frameon=False)

    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"psth_{area_name}_{sid}_u{args.unit_index:03d}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[done] Saved plot → {out_path}")

if __name__ == "__main__":
    main()