#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_axes.py  — Subspace version (Category & Direction) with per-category centering/whitening for R

- Loads cached binned arrays (03_cache_binned outputs) for one session (vertical-only).
- Learns pooled training-window subspaces per area:
    * Category subspace S_C (dim = --C_dim)
    * Direction subspace S_R (dim = --R_dim), learned on per-category centered/whitened data
      and made orthogonal to S_C; projections of ZR also use per-category center/whiten.
- Projects the entire analysis window to produce:
    * ZC: (trials, bins, C_dim)
    * ZR: (trials, bins, R_dim)
    * sC/sR: first component only (for backward compatibility)
- Saves: results/session/<sid>/axes_<AREA>.npz

Recommended defaults:
  --trainC_start 0.10 --trainC_end 0.30
  --trainR_start 0.05 --trainR_end 0.20
  --C_dim 1 --R_dim 2
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VALID_AREAS = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

# ---------- IO ----------
def load_cache(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["X"]                        # (trials, bins, units)
    meta = json.loads(str(z["meta"])) # bin_size_s, window_s, vertical-only, etc.
    trials = pd.read_json(StringIO(z["trials"].item()))
    return X.astype(float), meta, trials

# ---------- Helpers ----------
def pick_mask_from_time(meta: dict, start_s: float, end_s: float) -> np.ndarray:
    bs = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]  # e.g., [-0.25, 0.80]
    grid = np.arange(t0 + bs/2, t1 + bs/2, bs)  # bin centers
    return (grid >= start_s) & (grid < end_s)

def orthonormalize(M: np.ndarray) -> np.ndarray:
    """QR-based column orthonormalization (units x k) -> (units x k) with orthonormal columns."""
    if M.size == 0:
        return M
    Q, _ = np.linalg.qr(M)
    return Q

def project_out(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Make V orthogonal to subspace spanned by U's columns.
    U: (units x d), V: (units x r)  ->  V_orth = (I - UU^T) V; then re-orthonormalize.
    """
    if U.size == 0 or V.size == 0:
        return V
    Uo = orthonormalize(U)
    V_orth = V - Uo @ (Uo.T @ V)
    return orthonormalize(V_orth)

def residualize_against(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project X into the orthogonal complement of span(W). X:(trials,units), W:(units,k)."""
    if W.size == 0:
        return X
    Q, _ = np.linalg.qr(W)  # orthonormal basis
    return X - X @ Q @ Q.T

# ---------- Subspace learners ----------
def learn_category_subspace(Xmean: np.ndarray, C: np.ndarray, R: np.ndarray, C_dim: int, C_reg: float) -> np.ndarray:
    """
    Pooled training for Category.
    - Hold-one-R-out: get multiple weight vectors, stack -> SVD -> top C_dim right-singular vectors.
    Returns W_C (units x C_dim).
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(Xmean)
    Ws = []
    for hold in (1,2,3):
        mask_tr = (R != hold)
        mask_te = (R == hold)
        if mask_tr.sum() < 10 or mask_te.sum() < 5:
            continue
        lr = LogisticRegression(penalty="l2", C=C_reg, solver="liblinear", max_iter=2000)
        lr.fit(Xs[mask_tr], (C[mask_tr] > 0).astype(int))
        w = lr.coef_.ravel() / (scaler.scale_ + 1e-12)  # undo scaling
        Ws.append(w)
    if not Ws:
        return np.zeros((Xmean.shape[1], 0))
    W = np.vstack(Ws)  # (folds, units)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_C = Vt[:max(1, C_dim)].T  # (units x C_dim)
    return orthonormalize(W_C[:, :C_dim])

def learn_direction_subspace(Xmean_Rcw: np.ndarray, C: np.ndarray, R: np.ndarray, R_dim: int, C_reg: float) -> np.ndarray:
    """
    Pooled training for within-category direction, on per-category centered/whitened + residualized data.
    - For each category: multinomial logistic -> weight matrix (3 x units)
    - Stack across categories -> SVD -> top R_dim right-singular vectors
    Returns W_R (units x R_dim).
    """
    if R_dim <= 0 or Xmean_Rcw.shape[1] == 0:
        return np.zeros((Xmean_Rcw.shape[1], 0))
    blocks = []
    for cval in (-1, 1):
        m = (C == cval)
        if m.sum() < 30:
            continue
        Xc = Xmean_Rcw[m]  # already centered/whitened per category
        y = R[m].astype(int) - 1  # {0,1,2}
        if len(np.unique(y)) < 3:
            continue
        lr = LogisticRegression(penalty="l2", C=C_reg, solver="lbfgs", max_iter=2000)
        lr.fit(Xc, y)
        Wc = lr.coef_  # (3 x units)
        blocks.append(Wc)
    if not blocks:
        return np.zeros((Xmean_Rcw.shape[1], 0))
    W = np.vstack(blocks)  # (6 x units) ideally
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_R = Vt[:R_dim].T  # (units x R_dim)
    return orthonormalize(W_R)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True, help="Session id, e.g., 20200217")
    ap.add_argument("--areas", nargs="*", default=None,
                    help="Actual areas (MFEF/MLIP/MSC/SFEF/SLIP/SSC). If omitted, auto-detect from caches.")
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--out_dir",   type=Path, default=Path("results/session"))
    # training windows & dims
    ap.add_argument("--trainC_start", type=float, default=0.10, help="Category training window start (s)")
    ap.add_argument("--trainC_end",   type=float, default=0.30, help="Category training window end (s)")
    ap.add_argument("--trainR_start", type=float, default=0.05, help="Direction training window start (s)")
    ap.add_argument("--trainR_end",   type=float, default=0.20, help="Direction training window end (s)")
    ap.add_argument("--C_dim", type=int, default=1, help="Category subspace dimension")
    ap.add_argument("--R_dim", type=int, default=2, help="Direction subspace dimension")
    ap.add_argument("--C_reg", type=float, default=1.0)
    ap.add_argument("--out_tag", type=str, default="", help="Optional tag subfolder under results/session/<sid>/<tag>/")
    ap.add_argument("--skip_if_exists", action="store_true", default=False, help="Skip if output file exists.")

    args = ap.parse_args()

    # Auto-detect areas if none provided
    if not args.areas:
        caches = sorted(args.cache_dir.glob(f"{args.sid}_*.npz"))
        areas = [p.stem.split("_", 1)[1] for p in caches]
        args.areas = sorted({a for a in areas if a in VALID_AREAS})
        if not args.areas:
            print(f"[skip] no caches found for session {args.sid} in {args.cache_dir}")
            return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for area in args.areas:
        cache_path = args.cache_dir / f"{args.sid}_{area}.npz"
        if not cache_path.exists():
            print(f"[skip] cache missing: {cache_path}")
            continue

        X, meta, trials = load_cache(cache_path)
        nT, nB, nU = X.shape
        C = trials["C"].to_numpy(int)   # ±1
        R = trials["R"].to_numpy(int)   # 1/2/3

        # pooled training features
        mC = pick_mask_from_time(meta, args.trainC_start, args.trainC_end)
        mR = pick_mask_from_time(meta, args.trainR_start, args.trainR_end)
        if not mC.any(): raise SystemExit("Empty category training window; adjust --trainC_*")
        if not mR.any(): raise SystemExit("Empty direction training window; adjust --trainR_*")
        XmeanC = X[:, mC, :].mean(axis=1)   # (trials, units)
        XmeanR = X[:, mR, :].mean(axis=1)

        # ---- 1) Learn category subspace on pooled data (no per-cat transforms) ----
        W_C = learn_category_subspace(XmeanC, C, R, max(1, args.C_dim), args.C_reg)   # (units x d)
        d = W_C.shape[1]

        # ---- 2) Per-category centering/whitening for R (computed on R training window) ----
        cats = [-1, 1]
        mu_R = np.zeros((2, nU), float)   # index 0->-1, 1->+1
        std_R = np.ones((2, nU), float)
        for idx, cval in enumerate(cats):
            m = (C == cval)
            if m.sum() > 0:
                mu = XmeanR[m].mean(axis=0)
                sd = XmeanR[m].std(axis=0, ddof=1)
                sd[sd < 1e-8] = 1.0  # avoid div-by-zero; keep scale 1
                mu_R[idx] = mu
                std_R[idx] = sd

        # Build centered/whitened matrix for R-training
        XmeanR_cw = np.empty_like(XmeanR)
        for idx, cval in enumerate(cats):
            m = (C == cval)
            if m.sum() > 0:
                XmeanR_cw[m] = (XmeanR[m] - mu_R[idx]) / std_R[idx]

        # Residualize against W_C then learn W_R
        XmeanR_cw_res = residualize_against(XmeanR_cw, W_C)
        W_R = learn_direction_subspace(XmeanR_cw_res, C, R, max(0, args.R_dim), args.C_reg)
        r = W_R.shape[1]

        # Disentangle bases (belt & suspenders)
        W_R = project_out(W_C, W_R)
        W_C = project_out(W_R, W_C)

        # ---- 3) Project entire time course with per-category center/whiten for ZR ----
        ZC = np.zeros((nT, nB, d), float) if d>0 else np.zeros((nT, nB, 0), float)
        ZR = np.zeros((nT, nB, r), float) if r>0 else np.zeros((nT, nB, 0), float)

        # Precompute category index per trial: 0 for -1, 1 for +1
        cat_idx = ((C + 1) // 2).astype(int)  # maps -1->0, +1->1

        for t in range(nB):
            Xt = X[:, t, :]  # (trials, units)
            if d > 0:
                ZC[:, t, :] = Xt @ W_C
            if r > 0:
                # per-category center/whiten using training-window stats
                Xt_cw = np.empty_like(Xt)
                for idx in (0, 1):
                    m = (cat_idx == idx)
                    if m.any():
                        Xt_cw[m] = (Xt[m] - mu_R[idx]) / std_R[idx]
                # residualize against category subspace, then project
                Xt_cw_res = residualize_against(Xt_cw, W_C)
                ZR[:, t, :] = Xt_cw_res @ W_R

        # first component (for backward-compatible scripts)
        sC = ZC[..., 0] if d>0 else np.zeros((nT, nB), float)
        sR = ZR[..., 0] if r>0 else np.zeros((nT, nB), float)

        # Tagged output path
        base_dir = args.out_dir / f"{args.sid}"
        if args.out_tag:
            base_dir = base_dir / args.out_tag
        base_dir.mkdir(parents=True, exist_ok=True)
        outp = base_dir / f"axes_{area}.npz"
        if args.skip_if_exists and outp.exists():
            print(f"[skip] axes already exist → {outp}")
            continue

        meta_out = dict(meta)
        meta_out.update({
            "trainC_window_s": [args.trainC_start, args.trainC_end],
            "trainR_window_s": [args.trainR_start, args.trainR_end],
            "C_dim": int(d), "R_dim": int(r),
            "mu_R": mu_R.tolist(),
            "std_R": std_R.tolist(),
        })
        np.savez_compressed(outp,
                            W_C=W_C, W_R=W_R, ZC=ZC, ZR=ZR, sC=sC, sR=sR,
                            meta=json.dumps(meta_out))
        print(f"[ok] axes(subspaces) saved → {outp}  ZC {ZC.shape}  ZR {ZR.shape}")

if __name__ == "__main__":
    main()
