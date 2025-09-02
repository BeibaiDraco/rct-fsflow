#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_axes.py
- Loads cached binned arrays (03_cache_binned outputs) for one session
- Learns feature axes per area:
    * train_mode=pooled (default): learn β_C and β_R on pooled training windows
       - Category window: [--trainC_start, --trainC_end] (s)
       - Direction window: [--trainR_start, --trainR_end] (s)
    * train_mode=perbin: old behavior (refit at each time bin)
    * Orthogonalize β_R to β_C (and re-orthogonalize β_C to β_R once)
- Saves β_C(t), β_R(t) (time-tiled in pooled mode) and projections s_C(t), s_R(t)
Outputs: results/session/<sid>/axes_<AREA>.npz
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
    """ Xt: (trials, units) pooled window; C in {-1,+1}; R in {1,2,3} """
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
    """ Xt: (trials, units) pooled window; returns 1-D direction axis """
    betas = []
    for cval in (-1, 1):
        m = (C == cval)
        if m.sum() < 15:
            continue
        Xc = Xt[m] - Xt[m].mean(axis=0, keepdims=True)  # remove category mean
        scaler = StandardScaler(with_mean=False, with_std=True)
        Xs = scaler.fit_transform(Xc)
        y = R[m].astype(int) - 1  # {0,1,2}
        lr = LogisticRegression(penalty="l2", C=C_reg, solver="lbfgs",
                                multi_class="multinomial", max_iter=1000)
        lr.fit(Xs, y)
        W = lr.coef_  # (3, units)
        # 1-D axis = first PC of class boundaries
        U,S,Vt = np.linalg.svd(W, full_matrices=False)
        beta_dir = Vt[0]
        beta = beta_dir / (scaler.scale_ + 1e-12)  # undo scaling
        beta /= (np.linalg.norm(beta) + 1e-12)
        betas.append(beta)
    if not betas:
        return np.zeros(Xt.shape[1], dtype=float)
    b = np.mean(betas, axis=0)
    return b / (np.linalg.norm(b) + 1e-12)

def pick_mask_from_time(meta, start_s: float, end_s: float) -> np.ndarray:
    bin_size = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]      # e.g., [-0.25, 0.80]
    # bin centers (T0 + half bin, T1 exclusive by earlier binning)
    grid = np.arange(t0 + bin_size/2, t1 + bin_size/2, bin_size)
    return (grid >= start_s) & (grid < end_s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True, help="Session id, e.g., 20200217")
    ap.add_argument(
        "--areas", nargs="*", default=None,
        help="Actual areas (MFEF/MLIP/MSC/SFEF/SLIP/SSC). If omitted, auto-detect from caches for this session."
    )
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--out_dir",   type=Path, default=Path("results/session"))
    ap.add_argument("--C_reg", type=float, default=1.0)

    # Training mode & windows
    ap.add_argument("--train_mode", choices=["pooled","perbin"], default="pooled",
                    help="pooled: learn 1 axis per feature on a time window, then project all bins; perbin: refit each bin.")
    ap.add_argument("--trainC_start", type=float, default=0.10, help="Category training window start (s)")
    ap.add_argument("--trainC_end",   type=float, default=0.30, help="Category training window end (s)")
    ap.add_argument("--trainR_start", type=float, default=0.08, help="Direction training window start (s)")
    ap.add_argument("--trainR_end",   type=float, default=0.20, help="Direction training window end (s)")
    args = ap.parse_args()

    # Auto-detect areas if not provided
    if not args.areas:
        caches = sorted(args.cache_dir.glob(f"{args.sid}_*.npz"))
        areas = [p.stem.split("_", 1)[1] for p in caches]
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

        if args.train_mode == "pooled":
            # Build pooled training sets
            mC = pick_mask_from_time(meta, args.trainC_start, args.trainC_end)
            mR = pick_mask_from_time(meta, args.trainR_start, args.trainR_end)
            if not mC.any():
                raise SystemExit("Category training mask empty; adjust --trainC_start/--trainC_end")
            if not mR.any():
                raise SystemExit("Direction training mask empty; adjust --trainR_start/--trainR_end")

            XmeanC = X[:, mC, :].mean(axis=1)  # (trials, units)
            XmeanR = X[:, mR, :].mean(axis=1)

            bC = train_cat_axis(XmeanC, C, R, C_reg=args.C_reg)
            bR0 = train_withinR_axis(XmeanR, C, R, C_reg=args.C_reg)

            if np.linalg.norm(bC) > 0 and np.linalg.norm(bR0) > 0:
                bR = gram_schmidt(bC, bR0)  # make R orthogonal to C
                bC = gram_schmidt(bR, bC)   # optional symmetric pass
            else:
                bR = bR0

            # Project the entire time course with fixed axes
            for t in range(nB):
                Xt = X[:, t, :].astype(float)
                sC[:, t] = Xt @ bC
                sR[:, t] = Xt @ bR

            # Tile betas for convenience
            betaC[:] = bC
            betaR[:] = bR

        else:  # perbin mode (legacy)
            for t in range(nB):
                Xt = X[:, t, :].astype(float)
                bC = train_cat_axis(Xt, C, R, C_reg=args.C_reg)
                bR0 = train_withinR_axis(Xt, C, R, C_reg=args.C_reg)
                if np.linalg.norm(bC) > 0 and np.linalg.norm(bR0) > 0:
                    bR = gram_schmidt(bC, bR0)
                    bC = gram_schmidt(bR, bC)
                else:
                    bR = bR0
                betaC[t] = bC; betaR[t] = bR
                sC[:, t] = Xt @ bC
                sR[:, t] = Xt @ bR

        outp = args.out_dir / f"{args.sid}/axes_{a}.npz"
        outp.parent.mkdir(parents=True, exist_ok=True)
        meta_out = dict(meta)
        meta_out.update({
            "train_mode": args.train_mode,
            "trainC_window_s": [args.trainC_start, args.trainC_end],
            "trainR_window_s": [args.trainR_start, args.trainR_end],
        })
        np.savez_compressed(outp, betaC=betaC, betaR=betaR, sC=sC, sR=sR,
                            meta=json.dumps(meta_out))
        print(f"[ok] axes saved → {outp}  shapes: betaC {betaC.shape}, sC {sC.shape}")

if __name__ == "__main__":
    main()