#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_axes_qc.py  (rev: adds time-resolved AUC(R|sR))

QC for axes: 
- AUC(C|sC) over time
- AUC(C|sR) over time (should ~0.5)
- macro AUC(R|sR) over time (within-category; average across categories)

Saves a PNG per area with time in seconds and prints window summaries.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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

def time_axis(meta):
    bin_size = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]
    return np.arange(t0 + bin_size/2, t1 + bin_size/2, bin_size)

def macro_auc_ovr(scores: np.ndarray, y012: np.ndarray) -> float:
    """Macro one-vs-rest AUC for 3-class labels given 1D scores."""
    aucs = []
    for k in (0,1,2):
        yk = (y012 == k).astype(int)
        try:
            aucs.append(roc_auc_score(yk, scores))
        except Exception:
            pass
    return float(np.nanmean(aucs)) if aucs else np.nan

def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return x
    w = min(w, len(x))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True, help="Session id (YYYYMMDD)")
    ap.add_argument("--area", type=str, required=True, help="MFEF/MLIP/MSC or SFEF/SLIP/SSC")
    ap.add_argument("--axes_dir", type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--sum_win_start", type=float, default=0.0, help="Summary window start (s)")
    ap.add_argument("--sum_win_end",   type=float, default=0.8, help="Summary window end (s)")
    ap.add_argument("--smooth_bins", type=int, default=1, help="Optional moving average width (bins)")
    args = ap.parse_args()

    if args.area not in VALID:
        raise SystemExit(f"--area must be one of {sorted(VALID)}")

    sC, sR, meta = load_axes(args.sid, args.area, args.axes_dir)
    T = load_trials(args.cache_dir, args.sid, args.area)
    C01 = (T["C"].to_numpy(int) > 0).astype(int)  # 0/1
    R123 = T["R"].to_numpy(int)                   # 1/2/3
    yR   = R123 - 1                                # 0/1/2

    nT, nB = sC.shape
    tsec = time_axis(meta)

    # --- AUC(C|sC) and AUC(C|sR) over time ---
    aucC_from_sC = np.full(nB, np.nan)
    aucC_from_sR = np.full(nB, np.nan)
    for i in range(nB):
        try:   aucC_from_sC[i] = roc_auc_score(C01, sC[:, i])
        except Exception: pass
        try:   aucC_from_sR[i] = roc_auc_score(C01, sR[:, i])
        except Exception: pass

    # --- AUC(R|sR) over time (macro, within each C then averaged) ---
    aucR_from_sR = np.full(nB, np.nan)
    for i in range(nB):
        vals = []
        for c in (0,1):
            m = (C01 == c)
            if m.sum() < 30: 
                continue
            vals.append(macro_auc_ovr(sR[m, i], yR[m]))
        aucR_from_sR[i] = float(np.nanmean(vals)) if vals else np.nan

    # Optional smoothing in bins
    sb = max(1, int(args.smooth_bins))
    aucC_from_sC = moving_avg(aucC_from_sC, sb)
    aucC_from_sR = moving_avg(aucC_from_sR, sb)
    aucR_from_sR = moving_avg(aucR_from_sR, sb)

    # --- Summaries in a time (seconds) window ---
    msum = (tsec >= args.sum_win_start) & (tsec < args.sum_win_end)
    mean_aucCsC = float(np.nanmean(aucC_from_sC[msum]))
    mean_aucCsR = float(np.nanmean(aucC_from_sR[msum]))
    mean_aucRsR = float(np.nanmean(aucR_from_sR[msum]))

    print(f"[QC] {args.area}  session {args.sid}")
    print(f"  mean AUC(C | sC)[{args.sum_win_start:.3f},{args.sum_win_end:.3f}] = {mean_aucCsC:.3f}   (want >> 0.5)")
    print(f"  mean AUC(C | sR)[{args.sum_win_start:.3f},{args.sum_win_end:.3f}] = {mean_aucCsR:.3f}   (want ~ 0.5)")
    print(f"  mean AUC(R | sR)[{args.sum_win_start:.3f},{args.sum_win_end:.3f}] = {mean_aucRsR:.3f}   (want > 0.5)")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(tsec, aucC_from_sC, label="AUC(C | sC)")
    ax.plot(tsec, aucC_from_sR, label="AUC(C | sR)")
    ax.plot(tsec, aucR_from_sR, label="macro AUC(R | sR)")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.axvspan(args.sum_win_start, args.sum_win_end, color="grey", alpha=0.12, label="summary window")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
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