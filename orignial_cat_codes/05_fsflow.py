#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_fsflow.py
Feature-specific flow between area pairs, one session.

Inputs produced by previous steps:
  - results/session/<sid>/axes_<AREA>.npz  (contains sC, sR)
  - results/caches/<sid>_<AREA>.npz        (contains trials JSON for labels)

Computes:
  - fsGC_C, fsGC_R: feature-specific linear GC (bits/bin)
  - dI_C, dI_R: label-centric ΔNLL ("ΔI") (bits/trial)

Writes:
  - results/session/<sid>/flow_<A>to<B>.npz
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import log_loss

VALID_AREAS = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

# --------- IO helpers ---------
def load_axes(sid: int, area: str, base: Path):
    """Load axes_<AREA>.npz => sC, sR (trial x time), meta"""
    z = np.load(base / f"{sid}/axes_{area}.npz", allow_pickle=True)
    sC = z["sC"]  # (trials, bins)
    sR = z["sR"]  # (trials, bins)
    meta = json.loads(str(z["meta"]))
    return sC, sR, meta

def load_trials_from_cache(cache_dir: Path, sid: int, area: str) -> pd.DataFrame:
    """Load the trials dataframe (for labels C, R). Area is only used to pick any existing cache file."""
    # prefer the requested area cache if present; else try any cache for this session
    prefer = cache_dir / f"{sid}_{area}.npz"
    if prefer.exists():
        z = np.load(prefer, allow_pickle=True)
        return pd.read_json(z["trials"].item())
    # fallback: any cache with this sid
    anyc = sorted(cache_dir.glob(f"{sid}_*.npz"))
    if not anyc:
        raise FileNotFoundError(f"No caches found for sid {sid} in {cache_dir}")
    z = np.load(anyc[0], allow_pickle=True)
    return pd.read_json(z["trials"].item())

# --------- time-series helpers ---------
def build_hist(x: np.ndarray, k: int) -> np.ndarray:
    """Create history design (past 1..k) for 1D series x (T,) -> (T, k)."""
    T = x.shape[0]
    H = np.zeros((T, k), float)
    for i in range(1, k+1):
        H[i:, i-1] = x[:-i]
    return H

def concat_trials(signal: np.ndarray) -> np.ndarray:
    """Stack trials across time with a 1-bin NaN gap; drop NaNs to prevent leakage."""
    T, B = signal.shape
    gap = np.full((T, 1), np.nan)
    flat = np.hstack([signal, gap]).ravel()
    return flat[~np.isnan(flat)]

# --------- fsGC (linear; Gaussian) ---------
def fs_gc_bits(x: np.ndarray, y: np.ndarray, Z: Optional[np.ndarray], k: int = 4, ridge: float = 1e-2) -> float:
    """
    Linear GC: y_t ~ [y_{t-1..k}, x_{t-0..k}, Z] vs. reduced without x-terms.
    Returns: 0.5 * log(var_red/var_full)/log(2)  (bits/bin)
    """
    # design matrices
    Hy = build_hist(y, k)                # past of y
    Hx = build_hist(x, k+1)              # include contemporaneous x_t
    X_full = np.hstack([Hy, Hx])
    X_red  = Hy.copy()
    if Z is not None:
        X_full = np.hstack([X_full, Z])
        X_red  = np.hstack([X_red,  Z])

    off = k + 1  # drop first rows with zeros
    yy = y[off:]
    F  = X_full[off:]
    R  = X_red[off:]

    reg = Ridge(alpha=ridge, fit_intercept=True)
    reg.fit(F, yy)
    resid_full = yy - reg.predict(F)

    reg2 = Ridge(alpha=ridge, fit_intercept=True)
    reg2.fit(R, yy)
    resid_red  = yy - reg2.predict(R)

    var_full = float(np.var(resid_full))
    var_red  = float(np.var(resid_red))
    val = 0.5 * np.log((var_red + 1e-12) / (var_full + 1e-12)) / np.log(2.0)
    return val

# --------- ΔNLL ("ΔI") label-centric ---------
def delta_nll_bits_binary(A: np.ndarray, Bsig: np.ndarray, labels01: np.ndarray, k: int = 4) -> float:
    """
    ΔI for binary labels (category). Average over time bins:
      reduced: labels ~ past(B)
      full:    labels ~ past(B) + past(A)
    Return mean bits/trial across bins.
    """
    Bbins = Bsig.shape[1]
    imps = []
    for t in range(Bbins):
        Hb = build_hist(Bsig[:, t], k)
        Ha = build_hist(A[:, t],    k+1)  # include x_t

        off = k + 1
        m = np.arange(Hb.shape[0]) >= off
        Xr = Hb[m]
        Xf = np.hstack([Hb, Ha])[m]
        yy = labels01[m].astype(int)

        clf_r = LogisticRegression(max_iter=1000)
        clf_r.fit(Xr, yy)
        nll_r = log_loss(yy, clf_r.predict_proba(Xr), labels=[0, 1])

        clf_f = LogisticRegression(max_iter=1000)
        clf_f.fit(Xf, yy)
        nll_f = log_loss(yy, clf_f.predict_proba(Xf), labels=[0, 1])

        imps.append((nll_r - nll_f) / np.log(2))  # bits/trial
    return float(np.nanmean(imps))

def delta_nll_bits_multiclass(A: np.ndarray, Bsig: np.ndarray, labels012: np.ndarray, k: int = 4) -> float:
    """
    ΔI for 3-class labels (within-category R). We compute per time bin and average.
    """
    Bbins = Bsig.shape[1]
    imps = []
    for t in range(Bbins):
        Hb = build_hist(Bsig[:, t], k)
        Ha = build_hist(A[:, t],    k+1)

        off = k + 1
        m = np.arange(Hb.shape[0]) >= off
        Xr = Hb[m]
        Xf = np.hstack([Hb, Ha])[m]
        yy = labels012[m].astype(int)

        clf_r = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf_r.fit(Xr, yy)
        nll_r = log_loss(yy, clf_r.predict_proba(Xr), labels=[0,1,2])

        clf_f = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf_f.fit(Xf, yy)
        nll_f = log_loss(yy, clf_f.predict_proba(Xf), labels=[0,1,2])

        imps.append((nll_r - nll_f) / np.log(2))
    return float(np.nanmean(imps))

# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True, help="Session id, e.g., 20200217")
    ap.add_argument("--A", type=str, required=True, help="Source area (MFEF/MLIP/MSC/SFEF/SLIP/SSC)")
    ap.add_argument("--B", type=str, required=True, help="Target area (MFEF/MLIP/MSC/SFEF/SLIP/SSC)")
    ap.add_argument("--axes_dir",  type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--k", type=int, default=4, help="History lags")
    ap.add_argument("--ridge", type=float, default=1e-2, help="Ridge for GC regressions")
    ap.add_argument("--condition_other", action="store_true",
                    help="Condition fsGC of C on R (and R on C) using both areas' other-feature histories")
    args = ap.parse_args()

    if args.A not in VALID_AREAS or args.B not in VALID_AREAS or args.A == args.B:
        raise SystemExit("Invalid --A/--B areas or A==B.")

    # Load feature signals
    sC_A, sR_A, metaA = load_axes(args.sid, args.A, args.axes_dir)
    sC_B, sR_B, metaB = load_axes(args.sid, args.B, args.axes_dir)

    # Labels from any cache of this session (prefer B-area cache)
    trials = load_trials_from_cache(args.cache_dir, args.sid, args.B)
    labC = (trials["C"].to_numpy(int) > 0).astype(int)  # {0,1}
    labR = trials["R"].to_numpy(int) - 1                # {0,1,2}

    # Flatten trials to single long series with gaps for fsGC
    xC = concat_trials(sC_A)
    yC = concat_trials(sC_B)
    xR = concat_trials(sR_A)
    yR = concat_trials(sR_B)

    # Optional conditioning: use the "other" feature from both areas as covariates with history
    Z_C = None
    Z_R = None
    if args.condition_other:
        # For C-flow, condition on R from both A and B (include contemporaneous A and past of B)
        zC1 = build_hist(concat_trials(sR_A), args.k+1)
        zC2 = build_hist(concat_trials(sR_B), args.k)
        Z_C = np.c_[zC1, zC2]

        # For R-flow, condition on C from both A and B
        zR1 = build_hist(concat_trials(sC_A), args.k+1)
        zR2 = build_hist(concat_trials(sC_B), args.k)
        Z_R = np.c_[zR1, zR2]

    # fsGC (bits/bin)
    fsGC_C = fs_gc_bits(xC, yC, Z_C, k=args.k, ridge=args.ridge)
    fsGC_R = fs_gc_bits(xR, yR, Z_R, k=args.k, ridge=args.ridge)

    # ΔNLL (bits/trial)
    dI_C = delta_nll_bits_binary(sC_A, sC_B, labC, k=args.k)
    # For R, compute within each category and average to respect conditioning on C
    dI_R_list = []
    for cval in (0, 1):
        mask = (labC == cval)
        if mask.sum() < 30:
            continue
        # remap R to {0,1,2} and compute ΔI with series restricted to this category
        dI_R_list.append(
            delta_nll_bits_multiclass(sR_A[mask, :], sR_B[mask, :], labR[mask], k=args.k)
        )
    dI_R = float(np.nanmean(dI_R_list)) if dI_R_list else float("nan")

    # Save
    out_dir = Path("results/session") / str(args.sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = {
        "sid": int(args.sid),
        "A": str(args.A),
        "B": str(args.B),
        "k": int(args.k),
        "ridge": float(args.ridge),
        "condition_other": bool(args.condition_other),
    }
    np.savez_compressed(
        out_dir / f"flow_{args.A}to{args.B}.npz",
        fsGC_C=float(fsGC_C),
        fsGC_R=float(fsGC_R),
        dI_C=float(dI_C),
        dI_R=float(dI_R),
        meta=json.dumps(meta_out),
    )
    print(f"[ok] {args.A}->{args.B}:  fsGC_C={fsGC_C:.4f} bits/bin,  fsGC_R={fsGC_R:.4f} bits/bin,  "
          f"ΔI_C={dI_C:.4f} bits/trial,  ΔI_R={dI_R:.4f} bits/trial")

if __name__ == "__main__":
    main()