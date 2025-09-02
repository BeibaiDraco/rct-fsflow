#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_axes.py
- Loads cached binned arrays (03_cache_binned outputs) for one session
- For each area and time bin:
    * train category axis beta_C with hold-one-R-out generalization
    * train within-category direction axis beta_R after demeaning per C
    * Gram–Schmidt orthogonalize to disentangle
- Saves weights and single-trial projections s_C(t), s_R(t) per area
Outputs: results/session/<sid>/axes_<AREA>.npz
Works with actual area names (MFEF/MLIP/MSC or SFEF/SLIP/SSC).
If --areas is omitted, auto-detects from caches in results/caches.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VALID_AREAS = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def load_cache(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["X"]
    meta = json.loads(str(z["meta"]))
    trials = pd.read_json(z["trials"].item())
    return X, meta, trials

def gram_schmidt(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Remove u-component from v, then renorm
    proj = (v @ u) / (u @ u + 1e-12) * u
    w = v - proj
    n = np.linalg.norm(w) + 1e-12
    return w / n

def train_cat_axis(Xt: np.ndarray, C: np.ndarray, R: np.ndarray, C_reg=1.0) -> np.ndarray:
    """
    Xt: (trials, units) at one time bin
    C:  {-1,+1}
    R:  {1,2,3}
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(Xt)

    betas = []
    for hold in (1,2,3):
        mask_train = (R != hold)
        mask_test  = (R == hold)
        if mask_train.sum() < 10 or mask_test.sum() < 5:
            continue
        lr = LogisticRegression(penalty="l2", C=C_reg, solver="liblinear", max_iter=1000)
        lr.fit(Xs[mask_train], (C[mask_train] > 0).astype(int))
        beta = lr.coef_.ravel() / (scaler.scale_ + 1e-12)  # undo scaling
        beta /= (np.linalg.norm(beta) + 1e-12)
        betas.append(beta)
    if not betas:
        return np.zeros(Xt.shape[1], dtype=float)
    b = np.mean(betas, axis=0)
    return b / (np.linalg.norm(b) + 1e-12)

def train_withinR_axis(Xt: np.ndarray, C: np.ndarray, R: np.ndarray, C_reg=1.0) -> np.ndarray:
    """
    Demean within each category; train 3-way softmax per category; take first PC of class boundaries.
    Average the two category-specific axes.
    """
    betas = []
    for cval in (-1, 1):
        m = (C == cval)
        if m.sum() < 15:
            continue
        Xc = Xt[m]
        Xc = Xc - Xc.mean(axis=0, keepdims=True)  # remove category mean
        scaler = StandardScaler(with_mean=False, with_std=True)
        Xs = scaler.fit_transform(Xc)
        y = R[m].astype(int) - 1  # {0,1,2}
        lr = LogisticRegression(penalty="l2", C=C_reg, solver="lbfgs",
                                multi_class="multinomial", max_iter=1000)
        lr.fit(Xs, y)
        W = lr.coef_  # (3, units)
        # first PC of rows of W
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        beta_dir = Vt[0]
        beta = beta_dir / (scaler.scale_ + 1e-12)  # undo scaling
        beta /= (np.linalg.norm(beta) + 1e-12)
        betas.append(beta)
    if not betas:
        return np.zeros(Xt.shape[1], dtype=float)
    b = np.mean(betas, axis=0)
    return b / (np.linalg.norm(b) + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True, help="Session id, e.g., 20200217")
    ap.add_argument(
        "--areas", nargs="*", default=None,
        help="Actual areas (MFEF/MLIP/MSC/SFEF/SLIP/SSC). "
             "If omitted, auto-detect from caches for this session."
    )
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--out_dir",   type=Path, default=Path("results/session"))
    ap.add_argument("--C_reg", type=float, default=1.0)
    args = ap.parse_args()

    # Auto-detect areas from caches if not provided
    if not args.areas:
        caches = sorted(args.cache_dir.glob(f"{args.sid}_*.npz"))
        areas = [p.stem.split("_", 1)[1] for p in caches]  # part after '<sid>_'
        args.areas = sorted({a for a in areas if a in VALID_AREAS})
        if not args.areas:
            print(f"[skip] no caches found for session {args.sid} in {args.cache_dir}")
            return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for a in args.areas:  # actual area name
        cache_path = args.cache_dir / f"{args.sid}_{a}.npz"
        if not cache_path.exists():
            print(f"[skip] cache missing: {cache_path}")
            continue

        X, meta, trials = load_cache(cache_path)
        nT, nB, nU = X.shape
        C = trials["C"].to_numpy(int)
        R = trials["R"].to_numpy(int)

        betaC = np.zeros((nB, nU), float)
        betaR = np.zeros((nB, nU), float)
        sC = np.zeros((nT, nB), float)
        sR = np.zeros((nT, nB), float)

        for t in range(nB):
            Xt = X[:, t, :].astype(float)
            bC = train_cat_axis(Xt, C, R, C_reg=args.C_reg)
            bR0 = train_withinR_axis(Xt, C, R, C_reg=args.C_reg)

            if np.linalg.norm(bC) > 0 and np.linalg.norm(bR0) > 0:
                # Make R-axis orthogonal to C-axis (and renormalize)
                bR = gram_schmidt(bC, bR0)
                # Optional: make C-axis orthogonal to R-axis (one pass) for symmetry
                bC = gram_schmidt(bR, bC)
            else:
                bR = bR0

            betaC[t] = bC
            betaR[t] = bR
            sC[:, t] = Xt @ bC
            sR[:, t] = Xt @ bR

        outp = args.out_dir / f"{args.sid}/axes_{a}.npz"
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outp, betaC=betaC, betaR=betaR, sC=sC, sR=sR, meta=json.dumps(meta))
        print(f"[ok] axes saved → {outp}  shapes: betaC {betaC.shape}, sC {sC.shape}")

if __name__ == "__main__":
    main()