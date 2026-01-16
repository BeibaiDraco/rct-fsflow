#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_fsflow_timesliding.py
Time-sliding, multivariate, feature-specific GC between area pairs (one session).

Inputs:
  - results/session/<sid>/axes_<AREA>.npz  (must contain ZC, ZR, meta)
  - results/caches/<sid>_<AREA>.npz        (for trials: C,R)

For each time window:
  - C-flow: GC bits/bin for ZC_A -> ZC_B, conditioning on ZR_A,ZR_B histories
  - R-flow: GC bits/bin for ZR_A -> ZR_B, conditioning on ZC_A,ZC_B histories
  - Reverse-direction GC
  - Permutation nulls by shuffling A trials within (C,R) cells

Outputs:
  - results/session/<sid>/flow_timeseries_<A>to<B>.npz
    (curves, reverse, null percentiles, meta)
  - results/session/<sid>/flow_timeseries_<A>to<B>.png
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- IO ----------
def load_axes(sid: int, area: str, axes_dir: Path):
    z = np.load(axes_dir / f"{sid}/axes_{area}.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    ZC = z["ZC"] if "ZC" in z.files else z["sC"][..., None]  # (trials, bins, dC)
    ZR = z["ZR"] if "ZR" in z.files else z["sR"][..., None]  # (trials, bins, dR)
    return ZC.astype(float), ZR.astype(float), meta

def load_trials(cache_dir: Path, sid: int, prefer_area: str) -> pd.DataFrame:
    p = cache_dir / f"{sid}_{prefer_area}.npz"
    if not p.exists():
        anyp = sorted(cache_dir.glob(f"{sid}_*.npz"))
        if not anyp:
            raise FileNotFoundError(f"No caches for sid {sid} in {cache_dir}")
        p = anyp[0]
    z = np.load(p, allow_pickle=True)
    return pd.read_json(StringIO(z["trials"].item()))

def time_axis(meta: dict):
    bs = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]
    # bin centers; length should match ZC.shape[1]
    return np.arange(t0 + bs/2, t1 + bs/2, bs)

# ---------- GC machinery ----------
def make_VAR_design(ZA: np.ndarray, ZB: np.ndarray,
                    ZA_other: np.ndarray | None,
                    ZB_other: np.ndarray | None,
                    k: int, include_x0: bool=True):
    """
    Build design matrices within each trial then stack.
    ZA: (nT, T, dAx)  predictor feature
    ZB: (nT, T, dBy)  target feature
    *_other: (nT, T, dO) optional conditioning features
    Returns Y, X_full, X_red
    """
    nT, T, dAx = ZA.shape
    dBy = ZB.shape[2]
    rowsY = []
    rowsF = []
    rowsR = []
    for tr in range(nT):
        X_a = ZA[tr]    # (T, dAx)
        Y_b = ZB[tr]    # (T, dBy)
        Oa = ZA_other[tr] if ZA_other is not None and ZA_other.shape[2]>0 else None
        Ob = ZB_other[tr] if ZB_other is not None and ZB_other.shape[2]>0 else None
        # for t=k..T-1: build y_t and predictors
        for t in range(k, T):
            # past of y (1..k)
            past_y = []
            for i in range(1, k+1):
                past_y.append(Y_b[t-i])
            past_y = np.concatenate(past_y, axis=0) if past_y else np.zeros((0,))
            # x history including contemporaneous x_t
            hx = []
            if include_x0:
                hx.append(X_a[t])
            for i in range(1, k+1):
                hx.append(X_a[t-i])
            hx = np.concatenate(hx, axis=0) if hx else np.zeros((0,))
            # other covariates histories
            hz = []
            if Oa is not None:
                # include Oa_t and Oa past
                hz.append(Oa[t])
                for i in range(1, k+1):
                    hz.append(Oa[t-i])
            if Ob is not None:
                # include Ob past only (to avoid peeking at y_t)
                for i in range(1, k+1):
                    hz.append(Ob[t-i])
            hz = np.concatenate(hz, axis=0) if hz else np.zeros((0,))
            # stack
            X_full = np.concatenate([past_y, hx, hz], axis=0)
            X_red  = np.concatenate([past_y,           hz], axis=0)  # omit hx (i.e., omit A terms)
            rowsY.append(Y_b[t])
            rowsF.append(X_full)
            rowsR.append(X_red)
    Y = np.vstack(rowsY)            # (N, dBy)
    Xf = np.vstack(rowsF)           # (N, p_full)
    Xr = np.vstack(rowsR)           # (N, p_red)
    return Y, Xf, Xr

def ridge_resid_cov(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Multivariate ridge: solve B = (X^T X + alpha I)^-1 X^T Y
    Return residual covariance Σ = (E^T E)/N
    """
    # add small Tikhonov
    XtX = X.T @ X
    p = XtX.shape[0]
    A = XtX + alpha * np.eye(p)
    B = np.linalg.solve(A, X.T @ Y)   # (p, d)
    E = Y - X @ B
    # covariance with tiny jitter for stability
    Sigma = (E.T @ E) / max(1, Y.shape[0])
    # jitter if needed
    eps = 1e-12
    return Sigma + eps * np.eye(Sigma.shape[0])

def gc_bits(Y: np.ndarray, X_full: np.ndarray, X_red: np.ndarray, alpha: float) -> float:
    """GC in bits/bin via log-det of residual covariances (multivariate)."""
    S_full = ridge_resid_cov(X_full, Y, alpha)
    S_red  = ridge_resid_cov(X_red,  Y, alpha)
    # guard for det
    det_full = max(np.linalg.det(S_full), 1e-300)
    det_red  = max(np.linalg.det(S_red),  1e-300)
    return 0.5 * np.log(det_red / det_full) / np.log(2.0)

# ---------- Permutation within (C,R) ----------
def permute_A_within_CR(ZA: np.ndarray, trials: pd.DataFrame) -> np.ndarray:
    """
    Permute trial axis of ZA within each (C,R) cell. Returns a permuted copy.
    ZA: (nT, T, d)
    """
    ZA_perm = ZA.copy()
    idx = np.arange(len(trials))
    C = trials["C"].to_numpy(int)
    R = trials["R"].to_numpy(int)
    for c in (-1, 1):
        for r in (1, 2, 3):
            m = (C == c) & (R == r)
            ids = np.where(m)[0]
            if ids.size > 1:
                ZA_perm[m] = ZA[np.random.permutation(ids)]
    return ZA_perm

# ---------- Windows ----------
def build_windows(meta: dict, win_w: float, win_step: float):
    t = time_axis(meta)
    anchors = np.arange(t[0], t[-1] - win_w + 1e-9, win_step)
    # window masks and time centers
    wins = []
    centers = []
    for a in anchors:
        m = (t >= a) & (t < a + win_w)
        if m.sum() >= 2:
            wins.append(m)
            centers.append(a + win_w/2)
    return wins, np.array(centers, float)

# ---------- Main flow per window ----------
def flow_feature_timeseries(ZA: np.ndarray, ZB: np.ndarray,
                            ZA_other: np.ndarray, ZB_other: np.ndarray,
                            trials: pd.DataFrame, meta: dict,
                            win_w: float, win_step: float,
                            k: int, ridge: float, perms: int):
    """
    Compute GC time-series (forward, reverse) + permutation null bands.
    ZA, ZB: (nT, bins, d) for the *feature of interest* (C or R).
    ZA_other, ZB_other: conditioning features (may have dim 0).
    """
    wins, centers = build_windows(meta, win_w, win_step)
    nW = len(wins)
    fwd = np.full(nW, np.nan)
    rev = np.full(nW, np.nan)
    # nulls
    fnull = np.full((perms, nW), np.nan) if perms > 0 else None
    rnull = np.full((perms, nW), np.nan) if perms > 0 else None

    for w, m in enumerate(wins):
        # slice window
        ZA_w = ZA[:, m, :]
        ZB_w = ZB[:, m, :]
        ZAo_w = ZA_other[:, m, :] if ZA_other is not None and ZA_other.shape[2] > 0 else None
        ZBo_w = ZB_other[:, m, :] if ZB_other is not None and ZB_other.shape[2] > 0 else None

        # build designs for A->B
        Y, Xf, Xr = make_VAR_design(ZA_w, ZB_w, ZAo_w, ZBo_w, k=k, include_x0=True)
        if Y.shape[0] > (k+2) and Xf.shape[0] == Xr.shape[0]:
            fwd[w] = gc_bits(Y, Xf, Xr, alpha=ridge)

        # reverse: B->A
        Y2, Xf2, Xr2 = make_VAR_design(ZB_w, ZA_w, ZBo_w, ZAo_w, k=k, include_x0=True)
        if Y2.shape[0] > (k+2) and Xf2.shape[0] == Xr2.shape[0]:
            rev[w] = gc_bits(Y2, Xf2, Xr2, alpha=ridge)

        # permutations: shuffle A trials within (C,R) and recompute
        if perms > 0 and fnull is not None:
            for p in range(perms):
                ZA_perm = permute_A_within_CR(ZA_w, trials)
                # A->B under null
                Yp, Xfp, Xrp = make_VAR_design(ZA_perm, ZB_w, ZAo_w, ZBo_w, k=k, include_x0=True)
                fnull[p, w] = gc_bits(Yp, Xfp, Xrp, alpha=ridge) if Yp.shape[0] > (k+2) else np.nan
                # reverse null (permute B trials instead to match direction)
                ZB_perm = permute_A_within_CR(ZB_w, trials)
                Yq, Xfq, Xrq = make_VAR_design(ZB_perm, ZA_w, ZBo_w, ZAo_w, k=k, include_x0=True)
                rnull[p, w] = gc_bits(Yq, Xfq, Xrq, alpha=ridge) if Yq.shape[0] > (k+2) else np.nan

    # summarize nulls
    if perms > 0 and fnull is not None:
        f_lo = np.nanpercentile(fnull, 2.5, axis=0)
        f_hi = np.nanpercentile(fnull, 97.5, axis=0)
        r_lo = np.nanpercentile(rnull, 2.5, axis=0)
        r_hi = np.nanpercentile(rnull, 97.5, axis=0)
    else:
        f_lo = f_hi = r_lo = r_hi = None

    return centers, fwd, rev, (f_lo, f_hi, r_lo, r_hi)

# ---------- Plot ----------
def plot_pair(sid, A, B, tC, C_fwd, C_rev, C_band, tR, R_fwd, R_rev, R_band, out_png: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    # Category
    ax = axes[0]
    if C_band[0] is not None:
        ax.fill_between(tC, C_band[0], C_band[1], color="grey", alpha=0.15, label="perm 95%")
    ax.plot(tC, C_fwd, label=f"{A}→{B} (C)", lw=2)
    ax.plot(tC, C_rev, label=f"{B}→{A} (C, rev)", lw=1, ls="--")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_ylabel("GC bits/bin (C)")
    ax.legend(frameon=False)

    # Direction
    ax = axes[1]
    if R_band[0] is not None:
        ax.fill_between(tR, R_band[0], R_band[1], color="grey", alpha=0.15, label="perm 95%")
    ax.plot(tR, R_fwd, label=f"{A}→{B} (R)", lw=2)
    ax.plot(tR, R_rev, label=f"{B}→{A} (R, rev)", lw=1, ls="--")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel("GC bits/bin (R)")
    ax.legend(frameon=False)

    fig.suptitle(f"Feature-specific flow — {sid}: {A}→{B}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--A", type=str, required=True, help="Source area (MFEF/MLIP/MSC/SFEF/SLIP/SSC)")
    ap.add_argument("--B", type=str, required=True, help="Target area (MFEF/MLIP/MSC/SFEF/SLIP/SSC)")
    ap.add_argument("--axes_dir",  type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    # time-sliding params
    ap.add_argument("--win",   type=float, default=0.12, help="Window width (s)")
    ap.add_argument("--step",  type=float, default=0.02, help="Window step (s)")
    ap.add_argument("--k",     type=int,   default=4,     help="History lags (bins)")
    ap.add_argument("--ridge", type=float, default=1e-2,  help="Ridge regularization")
    ap.add_argument("--perms", type=int,   default=200,   help="Permutation count (0 to disable)")
    args = ap.parse_args()

    # Load features
    ZC_A, ZR_A, metaA = load_axes(args.sid, args.A, args.axes_dir)
    ZC_B, ZR_B, metaB = load_axes(args.sid, args.B, args.axes_dir)
    # sanity: bins must match
    if ZC_A.shape[1] != ZC_B.shape[1]:
        raise SystemExit("Mismatched bin counts between A and B axes.")
    # Trials dataframe for permutation grouping
    trials = load_trials(args.cache_dir, args.sid, args.B)

    # CATEGORY FLOW (condition on R)
    tC, C_fwd, C_rev, (C_lo, C_hi, C_rlo, C_rhi) = flow_feature_timeseries(
        ZA=ZC_A, ZB=ZC_B,
        ZA_other=ZR_A, ZB_other=ZR_B,
        trials=trials, meta=metaB,
        win_w=args.win, win_step=args.step,
        k=args.k, ridge=args.ridge, perms=args.perms
    )
    # DIRECTION FLOW (condition on C)
    # Skip gracefully if R-dim is zero in either area
    if ZR_A.shape[2] == 0 or ZR_B.shape[2] == 0:
        tR = tC.copy()
        R_fwd = np.full_like(tR, np.nan, dtype=float)
        R_rev = np.full_like(tR, np.nan, dtype=float)
        R_lo = R_hi = R_rlo = R_rhi = None
    else:
        tR, R_fwd, R_rev, (R_lo, R_hi, R_rlo, R_rhi) = flow_feature_timeseries(
            ZA=ZR_A, ZB=ZR_B,
            ZA_other=ZC_A, ZB_other=ZC_B,
            trials=trials, meta=metaB,
            win_w=args.win, win_step=args.step,
            k=args.k, ridge=args.ridge, perms=args.perms
        )

    # Save
    out_dir = args.axes_dir / f"{args.sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"flow_timeseries_{args.A}to{args.B}.npz"
    np.savez_compressed(
        out_npz,
        tC=tC, C_fwd=C_fwd, C_rev=C_rev, C_lo=C_lo, C_hi=C_hi, C_rlo=C_rlo, C_rhi=C_rhi,
        tR=tR, R_fwd=R_fwd, R_rev=R_rev, R_lo=R_lo, R_hi=R_hi, R_rlo=R_rlo, R_rhi=R_rhi,
        meta=json.dumps({
            "sid": int(args.sid), "A": args.A, "B": args.B,
            "win": float(args.win), "step": float(args.step),
            "k": int(args.k), "ridge": float(args.ridge),
            "perms": int(args.perms)
        })
    )
    print(f"[ok] saved {out_npz}")

    # Plot
    out_png = out_dir / f"flow_timeseries_{args.A}to{args.B}.png"
    plot_pair(args.sid, args.A, args.B,
              tC, C_fwd, C_rev, (C_lo, C_hi),
              tR, R_fwd, R_rev, (R_lo, R_hi),
              out_png)
    print(f"[done] wrote {out_png}")

if __name__ == "__main__":
    main()