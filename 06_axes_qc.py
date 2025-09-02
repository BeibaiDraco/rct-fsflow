#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_axes_qc.py — Subspace QC

Plots time-resolved:
  - AUC(C | ZC)(t)   (want >> 0.5)
  - AUC(C | ZR)(t)   (want ~ 0.5 if disentangled)
  - macro AUC(R | ZR)(t) within-category (want > 0.5 where direction exists)

Also prints pooled-window 5-fold CV metrics over the training windows stored in meta.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# add near the top
from sklearn.model_selection import StratifiedKFold
from io import StringIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def load_axes(sid: int, area: str, axes_dir: Path):
    z = np.load(axes_dir / f"{sid}/axes_{area}.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    # Prefer subspaces if present
    ZC = z["ZC"] if "ZC" in z.files else z["sC"][..., None]
    ZR = z["ZR"] if "ZR" in z.files else z["sR"][..., None]
    return ZC, ZR, meta

def load_trials(cache_dir: Path, sid: int, area: str) -> pd.DataFrame:
    z = np.load(cache_dir / f"{sid}_{area}.npz", allow_pickle=True)
    return pd.read_json(StringIO(z["trials"].item()))

def time_axis(meta):
    bs = float(meta.get("bin_size_s", 0.010)); t0, t1 = meta["window_s"]
    return np.arange(t0 + bs/2, t1 + bs/2, bs)

def mask_from_sec(meta, a: float, b: float):
    t = time_axis(meta)
    return (t >= a) & (t < b)

def macro_auc_R(Z: np.ndarray, y012: np.ndarray) -> float:
    """Macro one-vs-rest AUC for 3-class labels from multivariate scores (use prob of each class)."""
    # A simple linear softmax on Z; return avg one-vs-rest AUC
    if Z.shape[0] < 10: return np.nan
    # Fit multinomial, score on same set (QC curve; not CV)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(Z, y012)
    pro = clf.predict_proba(Z)  # (n,3)
    aucs=[]
    for k in (0,1,2):
        yk = (y012==k).astype(int)
        if len(np.unique(yk))<2: continue
        aucs.append(roc_auc_score(yk, pro[:,k]))
    return float(np.nanmean(aucs)) if aucs else np.nan

# replace your current auc_C_given_ZR_holdoutR with this version
def auc_C_given_ZR_holdoutR_across_categories(ZR_bin: np.ndarray, C01: np.ndarray, R123: np.ndarray) -> float:
    """
    Hold-one-R-out across categories:
      For each R in {1,2,3}, train on trials with R != hold (pooling both categories),
      test on trials with R == hold (pooling both categories), compute AUC on pooled test.
      Average AUCs over hold=1,2,3 (ignoring folds with too-few samples).
    """
    aucs = []
    for hold in (1, 2, 3):
        train = (R123 != hold)
        test  = (R123 == hold)
        # Need both C classes in train & test
        if train.sum() < 30 or test.sum() < 20:
            continue
        if np.unique(C01[train]).size < 2 or np.unique(C01[test]).size < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(ZR_bin[train], C01[train])
        p = clf.predict_proba(ZR_bin[test])[:, 1]
        aucs.append(roc_auc_score(C01[test], p))
    return float(np.nanmean(aucs)) if aucs else np.nan

def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w<=1: return x
    w = min(w, len(x))
    kern = np.ones(w)/w
    return np.convolve(x, kern, mode="same")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--area", type=str, required=True)
    ap.add_argument("--axes_dir", type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--smooth_bins", type=int, default=1)
    # summary window for printing
    ap.add_argument("--sum_win_start", type=float, default=0.0)
    ap.add_argument("--sum_win_end",   type=float, default=0.8)
    args = ap.parse_args()

    if args.area not in VALID:
        raise SystemExit(f"--area must be one of {sorted(VALID)}")

    ZC, ZR, meta = load_axes(args.sid, args.area, args.axes_dir)
    T = load_trials(args.cache_dir, args.sid, args.area)
    C01 = (T["C"].to_numpy(int) > 0).astype(int)
    yR  = T["R"].to_numpy(int) - 1

    tsec = time_axis(meta)
    nB = ZC.shape[1]
    dC = ZC.shape[2]; dR = ZR.shape[2]

    # ---- time-resolved AUC(C | ZC) and AUC(C | ZR) ----
    aucC_from_ZC = np.full(nB, np.nan)
    aucC_from_ZR = np.full(nB, np.nan)
    for i in range(nB):
        # simple linear logistic on current bin (no CV; QC curve)
        if dC>0:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(ZC[:,i,:], C01)
            pro = clf.predict_proba(ZC[:,i,:])[:,1]
            aucC_from_ZC[i] = roc_auc_score(C01, pro)
        if dR>0:
            aucC_from_ZR[i] = auc_C_given_ZR_holdoutR_across_categories(ZR[:, i, :], C01, T["R"].to_numpy(int))
    # ---- time-resolved macro AUC(R | ZR) within-category ----
    aucR_from_ZR = np.full(nB, np.nan)
    for i in range(nB):
        vals=[]
        for c in (0,1):
            m = (C01==c)
            if dR==0 or m.sum()<30: 
                continue
            vals.append(macro_auc_R(ZR[m,i,:], yR[m]))
        aucR_from_ZR[i] = float(np.nanmean(vals)) if vals else np.nan

    # smooth
    sb = max(1, int(args.smooth_bins))
    aucC_from_ZC = moving_avg(aucC_from_ZC, sb)
    aucC_from_ZR = moving_avg(aucC_from_ZR, sb)
    aucR_from_ZR = moving_avg(aucR_from_ZR, sb)

    # ---- pooled-window CV summaries using meta training windows ----
    Cwin = meta.get("trainC_window_s", [0.10, 0.30])
    Rwin = meta.get("trainR_window_s", [0.05, 0.20])
    mC = mask_from_sec(meta, Cwin[0], Cwin[1])
    mR = mask_from_sec(meta, Rwin[0], Rwin[1])

    # pooled features (mean over time in window)
    FC = ZC[:, mC, :].mean(axis=1) if dC>0 else np.zeros((ZC.shape[0],0))
    FR = ZR[:, mR, :].mean(axis=1) if dR>0 else np.zeros((ZR.shape[0],0))
    FC_from_ZR = ZR[:, mC, :].mean(axis=1) if dR>0 else np.zeros((ZR.shape[0],0))

    # 5-fold CV for C on FC
    def cv_auc(X, y):
        if X.shape[1]==0: return np.nan
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        vals=[]
        for tr,te in skf.split(X,y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[tr], y[tr])
            pro = clf.predict_proba(X[te])[:,1]
            vals.append(roc_auc_score(y[te], pro))
        return float(np.mean(vals))
    # 5-fold CV macro-OVR AUC for R|C on FR
    def cv_macro_auc_R(X, C01, yR):
        if X.shape[1]==0: return np.nan
        vals=[]
        for c in (0,1):
            m = (C01==c); 
            if m.sum()<60: continue
            yc = yR[m]
            if len(np.unique(yc))<3: continue
            Xc = X[m]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
            fold=[]
            for tr,te in skf.split(Xc, yc):
                clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
                clf.fit(Xc[tr], yc[tr])
                pro = clf.predict_proba(Xc[te])
                aucs=[]
                for k in (0,1,2):
                    yk=(yc[te]==k).astype(int)
                    if len(np.unique(yk))<2: continue
                    aucs.append(roc_auc_score(yk, pro[:,k]))
                fold.append(np.nanmean(aucs))
            if fold: vals.append(np.nanmean(fold))
        return float(np.nanmean(vals)) if vals else np.nan

    cv_aucC = cv_auc(FC, C01)
    cv_aucC_from_ZR = cv_auc(FC_from_ZR, C01)
    cv_aucR = cv_macro_auc_R(FR, C01, yR)

    print(f"[QC] {args.area}  sid={args.sid}")
    print(f"  CV AUC(C | ZC) pooled[{Cwin[0]:.2f},{Cwin[1]:.2f}] = {cv_aucC:.3f}")
    print(f"  CV AUC(C | ZR) pooled[{Cwin[0]:.2f},{Cwin[1]:.2f}] = {cv_aucC_from_ZR:.3f}")
    
    print(f"  CV macro AUC(R | ZR) pooled[{Rwin[0]:.2f},{Rwin[1]:.2f}] = {cv_aucR:.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10,4.8))
    ax.plot(tsec, aucC_from_ZC, label="AUC(C | ZC)")
    ax.plot(tsec, aucC_from_ZR, label="AUC(C | ZR)")
    ax.plot(tsec, aucR_from_ZR, label="macro AUC(R | ZR)")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False)
    ax.set_title(f"Axes QC — {args.area}  sid={args.sid}")
    out_dir = args.axes_dir / f"{args.sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qc_axes_{args.area}.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"[done] Saved QC figure → {out_path}")

if __name__ == "__main__":
    main()