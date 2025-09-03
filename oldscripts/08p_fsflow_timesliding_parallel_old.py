#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08p_fsflow_timesliding_parallel.py
Parallel time-sliding, multivariate, feature-specific GC between area pairs (one session).

Features vs 08:
- --n_jobs parallelizes permutation nulls with joblib
- --all_pairs runs every ordered pair among detected areas (MFEF/MLIP/MSC/SFEF/SLIP/SSC)

Outputs per pair:
  results/session/<sid>/flow_timeseries_<A>to<B>.npz/.png
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

VALID = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

# Avoid BLAS oversubscription when using joblib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- IO ----------
def load_axes(sid: int, area: str, axes_dir: Path):
    z = np.load(axes_dir / f"{sid}/axes_{area}.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    ZC = z["ZC"] if "ZC" in z.files else z["sC"][..., None]  # (nT, T, dC)
    ZR = z["ZR"] if "ZR" in z.files else z["sR"][..., None]  # (nT, T, dR)
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

def detect_areas(cache_dir: Path, sid: int):
    hits = sorted(cache_dir.glob(f"{sid}_*.npz"))
    areas = []
    for p in hits:
        a = p.stem.split("_", 1)[1]
        if a in VALID:
            areas.append(a)
    return sorted(set(areas))

def time_axis(meta: dict):
    bs = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]
    return np.arange(t0 + bs/2, t1 + bs/2, bs)

# ---------- GC machinery ----------
def make_VAR_design(ZA: np.ndarray, ZB: np.ndarray,
                    ZA_other: np.ndarray | None,
                    ZB_other: np.ndarray | None,
                    k: int, include_x0: bool=True):
    """
    ZA: (nT, T, dAx), ZB: (nT, T, dBy)
    *_other: (nT, T, dO) (may be None)
    Returns Y, X_full, X_reduced for multivariate ridge GC.
    """
    nT, T, dAx = ZA.shape
    dBy = ZB.shape[2]
    rowsY, rowsF, rowsR = [], [], []
    for tr in range(nT):
        Xa, Yb = ZA[tr], ZB[tr]
        Oa = ZA_other[tr] if ZA_other is not None and ZA_other.shape[2]>0 else None
        Ob = ZB_other[tr] if ZB_other is not None and ZB_other.shape[2]>0 else None
        for t in range(k, T):
            # y past
            past_y = np.concatenate([Yb[t-i] for i in range(1, k+1)], axis=0) if k>0 else np.zeros((0,))
            # x current + past
            hx = []
            if include_x0: hx.append(Xa[t])
            if k>0: hx.extend([Xa[t-i] for i in range(1, k+1)])
            hx = np.concatenate(hx, axis=0) if hx else np.zeros((0,))
            # other covariates (A: current+past; B: past only)
            hz = []
            if Oa is not None:
                hz.append(Oa[t])
                if k>0: hz.extend([Oa[t-i] for i in range(1, k+1)])
            if Ob is not None and k>0:
                hz.extend([Ob[t-i] for i in range(1, k+1)])
            hz = np.concatenate(hz, axis=0) if hz else np.zeros((0,))
            X_full = np.concatenate([past_y, hx, hz], axis=0)
            X_red  = np.concatenate([past_y,      hz], axis=0)  # without hx
            rowsY.append(Yb[t]); rowsF.append(X_full); rowsR.append(X_red)
    Y = np.vstack(rowsY)
    Xf = np.vstack(rowsF)
    Xr = np.vstack(rowsR)
    return Y, Xf, Xr

def ridge_resid_cov(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    XtX = X.T @ X
    A = XtX + alpha * np.eye(XtX.shape[0])
    B = np.linalg.solve(A, X.T @ Y)
    E = Y - X @ B
    S = (E.T @ E) / max(1, Y.shape[0])
    return S + 1e-12*np.eye(S.shape[0])

def gc_bits(Y: np.ndarray, X_full: np.ndarray, X_red: np.ndarray, alpha: float) -> float:
    S_full = ridge_resid_cov(X_full, Y, alpha)
    S_red  = ridge_resid_cov(X_red,  Y, alpha)
    det_full = max(np.linalg.det(S_full), 1e-300)
    det_red  = max(np.linalg.det(S_red),  1e-300)
    return 0.5*np.log(det_red/det_full)/np.log(2.0)

# ---------- Windows ----------
def build_windows(meta: dict, win_w: float, win_step: float):
    t = time_axis(meta)
    anchors = np.arange(t[0], t[-1] - win_w + 1e-9, win_step)
    wins, centers = [], []
    for a in anchors:
        m = (t >= a) & (t < a + win_w)
        if m.sum() >= 2:
            wins.append(m); centers.append(a + win_w/2)
    return wins, np.array(centers, float)

# ---------- Permutation helpers ----------
def permute_within_CR(trials: pd.DataFrame, nT: int, rng: np.random.Generator) -> np.ndarray:
    """Return permuted trial indices within each (C,R) cell."""
    idx = np.arange(nT)
    C = trials["C"].to_numpy(int); R = trials["R"].to_numpy(int)
    perm = idx.copy()
    for c in (-1,1):
        for r in (1,2,3):
            m = (C==c) & (R==r)
            ids = np.where(m)[0]
            if ids.size > 1:
                perm[m] = rng.permutation(ids)
    return perm

def flow_curves_for_perm(ZA: np.ndarray, ZB: np.ndarray,
                         ZA_other: np.ndarray, ZB_other: np.ndarray,
                         trials: pd.DataFrame, wins: list[np.ndarray],
                         k: int, ridge: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute forward and reverse GC curves across all windows for one permutation.
    For forward null: permute A trials; for reverse null: permute B trials.
    """
    rng = np.random.default_rng(seed)
    nT = ZA.shape[0]
    permA = permute_within_CR(trials, nT, rng)
    permB = permute_within_CR(trials, nT, rng)

    ZA_p = ZA[permA]
    ZB_p = ZB
    ZAo_p = ZA_other[permA] if ZA_other is not None and ZA_other.shape[2]>0 else None
    ZBo_p = ZB_other

    ZA_q = ZA
    ZB_q = ZB[permB]
    ZAo_q = ZA_other
    ZBo_q = ZB_other[permB] if ZB_other is not None and ZB_other.shape[2]>0 else None

    fwd = np.full(len(wins), np.nan); rev = np.full(len(wins), np.nan)
    for w, m in enumerate(wins):
        # forward
        Y, Xf, Xr = make_VAR_design(ZA_p[:,m,:], ZB_p[:,m,:],
                                    ZAo_p[:,m,:] if ZAo_p is not None else None,
                                    ZBo_p[:,m,:] if ZBo_p is not None else None,
                                    k=k, include_x0=True)
        if Y.shape[0] > (k+2): fwd[w] = gc_bits(Y, Xf, Xr, ridge)
        # reverse
        Y2, Xf2, Xr2 = make_VAR_design(ZB_q[:,m,:], ZA_q[:,m,:],
                                       ZBo_q[:,m,:] if ZBo_q is not None else None,
                                       ZAo_q[:,m,:] if ZAo_q is not None else None,
                                       k=k, include_x0=True)
        if Y2.shape[0] > (k+2): rev[w] = gc_bits(Y2, Xf2, Xr2, ridge)
    return fwd, rev

# ---------- Core: flow time-series with parallel nulls ----------
def flow_feature_timeseries_parallel(ZA: np.ndarray, ZB: np.ndarray,
                                     ZA_other: np.ndarray, ZB_other: np.ndarray,
                                     trials: pd.DataFrame, meta: dict,
                                     win_w: float, win_step: float,
                                     k: int, ridge: float,
                                     perms: int, n_jobs: int, seed: int):
    wins, centers = build_windows(meta, win_w, win_step)

    # true curves
    fwd = np.full(len(wins), np.nan); rev = np.full(len(wins), np.nan)
    for w, m in enumerate(wins):
        Y, Xf, Xr = make_VAR_design(ZA[:,m,:], ZB[:,m,:],
                                    ZA_other[:,m,:] if ZA_other is not None and ZA_other.shape[2]>0 else None,
                                    ZB_other[:,m,:] if ZB_other is not None and ZB_other.shape[2]>0 else None,
                                    k=k, include_x0=True)
        if Y.shape[0] > (k+2): fwd[w] = gc_bits(Y, Xf, Xr, ridge)

        Y2, Xf2, Xr2 = make_VAR_design(ZB[:,m,:], ZA[:,m,:],
                                       ZB_other[:,m,:] if ZB_other is not None and ZB_other.shape[2]>0 else None,
                                       ZA_other[:,m,:] if ZA_other is not None and ZA_other.shape[2]>0 else None,
                                       k=k, include_x0=True)
        if Y2.shape[0] > (k+2): rev[w] = gc_bits(Y2, Xf2, Xr2, ridge)

    # permutation curves in parallel (each task computes full curves)
    if perms > 0:
        seeds = np.random.SeedSequence(seed).spawn(perms)
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(flow_curves_for_perm)(
                ZA, ZB, ZA_other, ZB_other, trials, wins, k, ridge, int(s.generate_state(1)[0])
            ) for s in seeds
        )
        fnull = np.vstack([r[0] for r in results])  # (perms, nW)
        rnull = np.vstack([r[1] for r in results])
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
    ax = axes[0]
    if C_band[0] is not None:
        ax.fill_between(tC, C_band[0], C_band[1], color="grey", alpha=0.15, label="perm 95%")
    ax.plot(tC, C_fwd, label=f"{A}→{B} (C)", lw=2)
    ax.plot(tC, C_rev, label=f"{B}→{A} (C, rev)", lw=1, ls="--")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_ylabel("GC bits/bin (C)")
    ax.legend(frameon=False)

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

# ---------- One pair ----------
def run_one_pair(sid: int, A: str, B: str,
                 axes_dir: Path, cache_dir: Path,
                 win: float, step: float, k: int, ridge: float,
                 perms: int, n_jobs: int, seed: int):
    # Load features
    ZC_A, ZR_A, metaA = load_axes(sid, A, axes_dir)
    ZC_B, ZR_B, metaB = load_axes(sid, B, axes_dir)
    if ZC_A.shape[1] != ZC_B.shape[1]:
        raise SystemExit("Mismatched bin counts between A and B.")
    trials = load_trials(cache_dir, sid, B)

    # CATEGORY (condition on R)
    tC, C_fwd, C_rev, (C_lo, C_hi, C_rlo, C_rhi) = flow_feature_timeseries_parallel(
        ZA=ZC_A, ZB=ZC_B, ZA_other=ZR_A, ZB_other=ZR_B, trials=trials, meta=metaB,
        win_w=win, win_step=step, k=k, ridge=ridge, perms=perms, n_jobs=n_jobs, seed=seed+1
    )
    # DIRECTION (condition on C)
    if ZR_A.shape[2] == 0 or ZR_B.shape[2] == 0:
        tR = tC.copy()
        R_fwd = np.full_like(tR, np.nan, float); R_rev = np.full_like(tR, np.nan, float)
        R_lo = R_hi = R_rlo = R_rhi = None
    else:
        tR, R_fwd, R_rev, (R_lo, R_hi, R_rlo, R_rhi) = flow_feature_timeseries_parallel(
            ZA=ZR_A, ZB=ZR_B, ZA_other=ZC_A, ZB_other=ZC_B, trials=trials, meta=metaB,
            win_w=win, win_step=step, k=k, ridge=ridge, perms=perms, n_jobs=n_jobs, seed=seed+2
        )

    # Save NPZ + PNG
    out_dir = axes_dir / f"{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"flow_timeseries_{A}to{B}.npz"
    np.savez_compressed(
        out_npz,
        tC=tC, C_fwd=C_fwd, C_rev=C_rev, C_lo=C_lo, C_hi=C_hi, C_rlo=C_rlo, C_rhi=C_rhi,
        tR=tR, R_fwd=R_fwd, R_rev=R_rev, R_lo=R_lo, R_hi=R_hi, R_rlo=R_rlo, R_rhi=R_rhi,
        meta=json.dumps({"sid": sid, "A": A, "B": B, "win": win, "step": step, "k": k,
                         "ridge": ridge, "perms": perms, "n_jobs": n_jobs})
    )
    out_png = out_dir / f"flow_timeseries_{A}to{B}.png"
    plot_pair(sid, A, B, tC, C_fwd, C_rev, (C_lo, C_hi), tR, R_fwd, R_rev, (R_lo, R_hi), out_png)
    print(f"[done] {A}->{B}  saved {out_npz} and {out_png}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--A", type=str, help="Source area (MFEF/MLIP/MSC/SFEF/SLIP/SSC)")
    ap.add_argument("--B", type=str, help="Target area (MFEF/MLIP/MSC/SFEF/SLIP/SSC)")
    ap.add_argument("--all_pairs", action="store_true",
                    help="Run all ordered pairs among detected areas for this session.")
    ap.add_argument("--axes_dir",  type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--win",   type=float, default=0.12)
    ap.add_argument("--step",  type=float, default=0.02)
    ap.add_argument("--k",     type=int,   default=4)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int,   default=200)
    ap.add_argument("--n_jobs", type=int,  default=8, help="Parallel workers for permutations")
    ap.add_argument("--seed",  type=int,   default=123)
    args = ap.parse_args()

    if args.all_pairs:
        areas = detect_areas(args.cache_dir, args.sid)
        areas = [a for a in areas if a in VALID]
        if len(areas) < 2:
            raise SystemExit(f"Need >=2 areas; found {areas}")
        pairs = [(a,b) for a in areas for b in areas if a!=b]
        print(f"[info] sid={args.sid} running {len(pairs)} ordered pairs: {pairs}")
        for (A,B) in pairs:
            run_one_pair(args.sid, A, B, args.axes_dir, args.cache_dir,
                         args.win, args.step, args.k, args.ridge,
                         args.perms, args.n_jobs, args.seed)
    else:
        if not args.A or not args.B:
            raise SystemExit("Provide --A and --B, or use --all_pairs.")
        run_one_pair(args.sid, args.A, args.B, args.axes_dir, args.cache_dir,
                     args.win, args.step, args.k, args.ridge,
                     args.perms, args.n_jobs, args.seed)

if __name__ == "__main__":
    main()