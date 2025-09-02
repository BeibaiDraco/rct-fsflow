#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_decode_raw.py
Raw population decoding (no axis reduction) from cached vertical-only data.

- Loads: results/caches/<sid>_<AREA>.npz (from 03_cache_binned.py)
- Pooled-window decoding (default 80–200 ms):
    * Category (binary): 5-fold stratified CV on raw population → AUC, accuracy
    * Within-category direction (3-class): run inside each category separately (5-fold) → macro-avg accuracy & macro-OVR AUC, then average across categories
- Optional time-resolved curve with sliding window.

Outputs:
- Prints a small table
- Saves CSV to results/session/<sid>/decode_raw_<AREA>.csv
- Optional PNG if --timeseries is used
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def load_cache(cache_dir: Path, sid: int, area: str) -> Tuple[np.ndarray, Dict[str,Any], pd.DataFrame]:
    path = cache_dir / f"{sid}_{area}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    z = np.load(path, allow_pickle=True)
    X = z["X"]  # (trials, bins, units) int16
    meta = json.loads(str(z["meta"]))
    trials = pd.read_json(z["trials"].item())
    return X.astype(float), meta, trials

def time_grid(meta: Dict[str,Any]) -> np.ndarray:
    bs = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]
    return np.arange(t0 + bs/2, t1 + bs/2, bs)

def mask_from_window(meta: Dict[str,Any], start_s: float, end_s: float) -> np.ndarray:
    t = time_grid(meta)
    return (t >= start_s) & (t < end_s)

def pooled_features(X: np.ndarray, mwin: np.ndarray) -> np.ndarray:
    """Average across bins in window -> (trials, units)."""
    return X[:, mwin, :].mean(axis=1)

def decode_C_cv(X: np.ndarray, C01: np.ndarray, nfold: int = 5) -> Dict[str,float]:
    """Binary category decoding with 5-fold stratified CV."""
    # stratify by C only
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=123)
    aucs, accs = [], []
    for tr, te in skf.split(X, C01):
        sc = StandardScaler(with_mean=True, with_std=True)
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ytr = C01[tr]; yte = C01[te]
        clf = LogisticRegression(max_iter=2000, penalty="l2", solver="lbfgs")
        clf.fit(Xtr, ytr)
        pro = clf.predict_proba(Xte)[:,1]
        aucs.append(roc_auc_score(yte, pro))
        accs.append(accuracy_score(yte, (pro>=0.5).astype(int)))
    return {"auc": float(np.mean(aucs)), "acc": float(np.mean(accs))}

def decode_R_withinC_cv(X: np.ndarray, C01: np.ndarray, R123: np.ndarray, nfold: int = 5) -> Dict[str,float]:
    """
    Within-category 3-way decoding.
    For each category separately:
      - stratify by R (1/2/3)
      - multinomial logistic on raw population
      - report macro-accuracy (balanced) and macro-OVR AUC
    Then average across the two categories (only where enough trials per class).
    """
    out_acc, out_auc = [], []
    for c in (0,1):
        m = (C01 == c)
        if m.sum() < 60:  # need enough trials
            continue
        y = (R123[m] - 1).astype(int)  # 0/1/2
        # ensure all classes present
        if len(np.unique(y)) < 3:
            continue
        Xc = X[m]
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=123)
        accs, aucs = [], []
        for tr, te in skf.split(Xc, y):
            sc = StandardScaler(with_mean=True, with_std=True)
            Xtr = sc.fit_transform(Xc[tr]); Xte = sc.transform(Xc[te])
            ytr = y[tr]; yte = y[te]
            clf = LogisticRegression(max_iter=2000, penalty="l2", solver="lbfgs", multi_class="multinomial")
            clf.fit(Xtr, ytr)
            pro = clf.predict_proba(Xte)  # (n,3)
            # macro accuracy = average per-class accuracy
            pred = pro.argmax(1)
            accs.append(np.mean([(pred[yte==k]==k).mean() if np.any(yte==k) else np.nan for k in (0,1,2)]))
            # macro OVR AUC:
            ovr_aucs=[]
            for k in (0,1,2):
                yk = (yte==k).astype(int)
                # handle degenerate folds
                if np.unique(yk).size < 2: 
                    continue
                ovr_aucs.append(roc_auc_score(yk, pro[:,k]))
            aucs.append(np.nanmean(ovr_aucs))
        if accs:
            out_acc.append(np.nanmean(accs))
        if aucs:
            out_auc.append(np.nanmean(aucs))
    return {
        "macro_acc": float(np.nanmean(out_acc)) if out_acc else np.nan,
        "macro_auc": float(np.nanmean(out_auc)) if out_auc else np.nan
    }

def timeseries_decoding(X: np.ndarray, meta: Dict[str,Any], C01: np.ndarray, R123: np.ndarray,
                        win_w: float, win_step: float, nfold:int=5) -> pd.DataFrame:
    """
    Time-resolved decoding with sliding window:
      - For each window center, pool bins in [t, t+win_w), decode C and R|C
    Returns a DataFrame with time (s), aucC, accC, macroAccR, macroAucR
    """
    t = time_grid(meta)
    rows=[]
    start = t[0]
    end   = t[-1]
    # window anchors
    anchors = np.arange(start, end - win_w + 1e-9, win_step)
    for a in anchors:
        mwin = (t >= a) & (t < a + win_w)
        if mwin.sum() < 2: 
            continue
        F = pooled_features(X, mwin)
        resC = decode_C_cv(F, C01, nfold=nfold)
        resR = decode_R_withinC_cv(F, C01, R123, nfold=nfold)
        rows.append({
            "t_start": float(a),
            "t_end": float(a+win_w),
            "aucC": resC["auc"], "accC": resC["acc"],
            "macroAccR": resR["macro_acc"], "macroAucR": resR["macro_auc"],
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--area", type=str, required=True, help="MFEF/MLIP/MSC or SFEF/SLIP/SSC")
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--out_dir",   type=Path, default=Path("results/session"))
    # pooled window (s)
    ap.add_argument("--pool_start", type=float, default=0.08)
    ap.add_argument("--pool_end",   type=float, default=0.20)
    # optional time series
    ap.add_argument("--timeseries", action="store_true")
    ap.add_argument("--win_w",    type=float, default=0.12, help="sliding window width (s)")
    ap.add_argument("--win_step", type=float, default=0.02, help="sliding window step (s)")
    args = ap.parse_args()

    if args.area not in VALID:
        raise SystemExit(f"--area must be one of {sorted(VALID)}")

    X, meta, trials = load_cache(args.cache_dir, args.sid, args.area)
    C01 = (trials["C"].to_numpy(int) > 0).astype(int)
    R123 = trials["R"].to_numpy(int)

    # pooled window features
    mwin = mask_from_window(meta, args.pool_start, args.pool_end)
    if not mwin.any():
        raise SystemExit("Empty pooled window; adjust --pool_start/--pool_end")
    F = pooled_features(X, mwin)

    resC = decode_C_cv(F, C01, nfold=5)
    resR = decode_R_withinC_cv(F, C01, R123, nfold=5)

    print(f"[raw-decode] sid={args.sid} area={args.area} pooled [{args.pool_start:.3f},{args.pool_end:.3f}] s")
    print(f"  Category:  AUC={resC['auc']:.3f}  ACC={resC['acc']:.3f}")
    print(f"  R|C:       macro-ACC={resR['macro_acc']:.3f}  macro-AUC={resR['macro_auc']:.3f}")

    # Save CSV
    out_dir = args.out_dir / f"{args.sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "sid": args.sid, "area": args.area,
        "pool_start": args.pool_start, "pool_end": args.pool_end,
        "aucC": resC["auc"], "accC": resC["acc"],
        "macroAccR": resR["macro_acc"], "macroAucR": resR["macro_auc"],
    }]).to_csv(out_dir / f"decode_raw_{args.area}.csv", index=False)

    # Optional time-resolved
    if args.timeseries:
        df = timeseries_decoding(X, meta, C01, R123, win_w=args.win_w, win_step=args.win_step, nfold=5)
        df.to_csv(out_dir / f"decode_raw_timeseries_{args.area}.csv", index=False)
        # plot
        fig, ax = plt.subplots(figsize=(9,4.5))
        tmid = 0.5*(df["t_start"].to_numpy() + df["t_end"].to_numpy())
        ax.plot(tmid, df["aucC"], label="AUC(C) raw")
        ax.plot(tmid, df["macroAucR"], label="macro AUC(R|C) raw")
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_xlabel("time (s) from cat_stim_on")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.3, 1.0)
        ax.legend(frameon=False)
        ax.set_title(f"Raw decoding (sliding) — {args.area} sid={args.sid}")
        fig.tight_layout()
        fig.savefig(out_dir / f"decode_raw_timeseries_{args.area}.png", dpi=150)
        print(f"[done] wrote {out_dir}/decode_raw_timeseries_{args.area}.csv/.png")

if __name__ == "__main__":
    main()