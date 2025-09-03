#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_axes_qc.py
QC for feature axes produced by 04_build_axes.py.

For one session+area:
- Compute AUC(time) of category from sC (want >> .5 in informative window)
- Compute AUC(time) of category from sR (want ~ .5)
- Compute macro AUC of within-category direction from sR (want > .5)

Saves a PNG under results/session/<sid>/qc_axes_<AREA>.png
Prints scalar summaries to stdout.

Usage (interactive):
  python 06_axes_qc.py --sid 20200217 --area MFEF

Notes:
- No SciPy dependencies; only numpy/pandas/sklearn/matplotlib.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def load_axes(sid: int, area: str, axes_dir: Path):
    z = np.load(axes_dir / f"{sid}/axes_{area}.npz", allow_pickle=True)
    sC = z["sC"]  # (trials, bins)
    sR = z["sR"]  # (trials, bins)
    meta = json.loads(str(z["meta"]))
    return sC, sR, meta

def load_trials(cache_dir: Path, sid: int, area_hint: str) -> pd.DataFrame:
    pref = cache_dir / f"{sid}_{area_hint}.npz"
    if pref.exists():
        z = np.load(pref, allow_pickle=True)
        return pd.read_json(z["trials"].item())
    anyc = sorted(cache_dir.glob(f"{sid}_*.npz"))
    if not anyc:
        raise FileNotFoundError(f"No caches for sid {sid} in {cache_dir}")
    z = np.load(anyc[0], allow_pickle=True)
    return pd.read_json(z["trials"].item())

def macro_auc_multiclass(scores: np.ndarray, y012: np.ndarray) -> float:
    """
    Macro one-vs-rest AUC given 1D scores; we sweep a simple linear classifier for stability.
    Here we just compute OVR AUCs using the scores directly.
    """
    aucs = []
    for k in (0,1,2):
        yk = (y012 == k).astype(int)
        try:
            aucs.append(roc_auc_score(yk, scores))
        except Exception:
            pass
    return float(np.nanmean(aucs)) if aucs else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True, help="Session id (YYYYMMDD)")
    ap.add_argument("--area", type=str, required=True, help="MFEF/MLIP/MSC or SFEF/SLIP/SSC")
    ap.add_argument("--axes_dir", type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    # Analysis window (bin indices) for scalar summaries; defaults assume 10 ms bins and [-0.3,1.2]s
    ap.add_argument("--win_start_bin", type=int, default=30, help="inclusive start bin (default ~0.0s)")
    ap.add_argument("--win_end_bin",   type=int, default=120, help="exclusive end bin (default ~0.9s)")
    args = ap.parse_args()

    if args.area not in VALID:
        raise SystemExit(f"--area must be one of {sorted(VALID)}")

    sC, sR, meta = load_axes(args.sid, args.area, args.axes_dir)
    T = load_trials(args.cache_dir, args.sid, args.area)
    C01 = (T["C"].to_numpy(int) > 0).astype(int)   # 0/1
    R123 = T["R"].to_numpy(int)                    # 1/2/3
    yR = (R123 - 1)                                # 0/1/2

    nT, nB = sC.shape
    # ---- AUC(time) for C from sC and sR ----
    aucC_from_sC = np.full(nB, np.nan)
    aucC_from_sR = np.full(nB, np.nan)
    for t in range(nB):
        try:
            aucC_from_sC[t] = roc_auc_score(C01, sC[:, t])
        except Exception:
            pass
        try:
            aucC_from_sR[t] = roc_auc_score(C01, sR[:, t])
        except Exception:
            pass

    # ---- Within-category direction decodability from sR ----
    # Simple summary: average macro AUC within each C using time-averaged sR
    sR_mean = sR.mean(axis=1)  # (trials,)
    macro_aucs = []
    for c in (0, 1):
        m = (C01 == c)
        if m.sum() < 30:
            continue
        macro_aucs.append(macro_auc_multiclass(sR_mean[m], yR[m]))
    macro_auc_R_withinC = float(np.nanmean(macro_aucs)) if macro_aucs else np.nan

    # ---- Scalar summaries over a window ----
    w0, w1 = int(args.win_start_bin), int(args.win_end_bin)
    mean_aucCsC = float(np.nanmean(aucC_from_sC[w0:w1]))
    mean_aucCsR = float(np.nanmean(aucC_from_sR[w0:w1]))

    print(f"[QC] {args.area}  session {args.sid}")
    print(f"  mean AUC(C | sC)[{w0}:{w1}] = {mean_aucCsC:.3f}   (want >> 0.5)")
    print(f"  mean AUC(C | sR)[{w0}:{w1}] = {mean_aucCsR:.3f}   (want ~ 0.5)")
    print(f"  macro AUC(R | sR) within C  = {macro_auc_R_withinC:.3f} (want > 0.5)")

    # ---- Plot time courses ----
    t = np.arange(nB)  # bin index (10 ms per bin if you used 0.010 s)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, aucC_from_sC, label="AUC(C | sC)")
    ax.plot(t, aucC_from_sR, label="AUC(C | sR)")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.axvspan(w0, w1, color="grey", alpha=0.1, label="summary window")
    ax.set_xlabel("time bin (10 ms each, aligned to cat_stim_on)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False)
    ax.set_title(f"Axes QC — {args.area}  sid={args.sid}")
    out_dir = args.axes_dir / f"{args.sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qc_axes_{args.area}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[done] Saved QC figure → {out_path}")

if __name__ == "__main__":
    main()