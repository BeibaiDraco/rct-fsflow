#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08i_fsflow_timesliding_integrated.py
Time-sliding, multivariate, feature-specific GC + integrated-band test (parallel permutations).

NEW in this version:
  • Sliding integrated GC (rectangular window swept across time)
  • Per-time p-values from integrated-null curves
  • Save full permutation GC matrices (forward & reverse)
  • Two plots per pair: RAW (non-integrated) and INT (sliding-integrated)
  • Plot integrated-null mean ± 95% band in INT plot
  • Keep single pre-registered band integrated stats/p-values (no band span drawn by default)
  • Tagging & skip-if-exists preserved

Example:
  python 08i_fsflow_timesliding_integrated.py --sid 20200926 --all_pairs \
    --win 0.16 --step 0.02 --k 5 --perms 500 --n_jobs 16 \
    --band_start 0.12 --band_end 0.28 --int_win 0.16 \
    --out_tag win160_k5_perm500 --skip_if_exists \
    --annotate_p_raw --annotate_p_int --p_text_stride 5
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

# Avoid BLAS oversubscription when parallelizing
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

# ---------- IO ----------
def load_axes(sid: int, area: str, axes_dir: Path, out_tag: str = ""):
    base = axes_dir / str(sid)
    tagged = base / out_tag if out_tag else base
    f = tagged / f"axes_{area}.npz"
    if not f.exists() and out_tag:
        alt = base / f"axes_{area}.npz"
        if alt.exists():
            print(f"[warn] axes not found in tag '{out_tag}', falling back to {alt}")
            f = alt
        else:
            raise FileNotFoundError(f"axes_{area}.npz not found in {tagged} nor {base}")
    elif not f.exists():
        raise FileNotFoundError(f"axes_{area}.npz not found in {tagged}")
    z = np.load(f, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    ZC = z["ZC"] if "ZC" in z.files else z["sC"][..., None]
    ZR = z["ZR"] if "ZR" in z.files else z["sR"][..., None]
    return ZC.astype(float), ZR.astype(float), meta

def load_trials(cache_dir: Path, sid: int, prefer_area: str) -> pd.DataFrame:
    p = cache_dir / f"{sid}_{prefer_area}.npz"
    if not p.exists():
        anyp = sorted(cache_dir.glob(f"{sid}_*.npz"))
        if not anyp: raise FileNotFoundError(f"No caches for sid {sid}")
        p = anyp[0]
    z = np.load(p, allow_pickle=True)
    return pd.read_json(StringIO(z["trials"].item()))

def detect_areas(cache_dir: Path, sid: int):
    hits = sorted(cache_dir.glob(f"{sid}_*.npz"))
    areas=[]
    for p in hits:
        a = p.stem.split("_",1)[1]
        if a in VALID: areas.append(a)
    return sorted(set(areas))

def time_axis(meta: dict):
    bs = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]
    return np.arange(t0 + bs/2, t1 + bs/2, bs)

# ---------- GC machinery ----------
def make_VAR_design(ZA: np.ndarray, ZB: np.ndarray,
                    ZA_other: np.ndarray|None,
                    ZB_other: np.ndarray|None,
                    k: int, include_x0: bool=True):
    nT, T, _ = ZA.shape
    rowsY, rowsF, rowsR = [], [], []
    for tr in range(nT):
        Xa, Yb = ZA[tr], ZB[tr]
        Oa = ZA_other[tr] if ZA_other is not None and ZA_other.shape[2]>0 else None
        Ob = ZB_other[tr] if ZB_other is not None and ZB_other.shape[2]>0 else None
        for t in range(k, T):
            past_y = np.concatenate([Yb[t-i] for i in range(1,k+1)], axis=0) if k>0 else np.zeros((0,))
            hx = []
            if include_x0: hx.append(Xa[t])
            if k>0: hx.extend([Xa[t-i] for i in range(1,k+1)])
            hx = np.concatenate(hx, axis=0) if hx else np.zeros((0,))
            hz = []
            if Oa is not None:
                hz.append(Oa[t])
                if k>0: hz.extend([Oa[t-i] for i in range(1,k+1)])
            if Ob is not None and k>0:
                hz.extend([Ob[t-i] for i in range(1,k+1)])
            hz = np.concatenate(hz, axis=0) if hz else np.zeros((0,))
            X_full = np.concatenate([past_y, hx, hz], axis=0)
            X_red  = np.concatenate([past_y,      hz], axis=0)
            rowsY.append(Yb[t]); rowsF.append(X_full); rowsR.append(X_red)
    Y = np.vstack(rowsY); Xf = np.vstack(rowsF); Xr = np.vstack(rowsR)
    return Y, Xf, Xr

def ridge_resid_cov(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    XtX = X.T @ X
    A = XtX + alpha*np.eye(XtX.shape[0])
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
    return wins, np.array(centers,float)

def band_mask(tcenters: np.ndarray, start: float, end: float) -> np.ndarray:
    return (tcenters >= start) & (tcenters < end)

# ---------- Permutation helpers ----------
def permute_within_CR(trials: pd.DataFrame, nT: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(nT)
    C = trials["C"].to_numpy(int); R = trials["R"].to_numpy(int)
    perm = idx.copy()
    for c in (-1,1):
        for r in (1,2,3):
            m = (C==c) & (R==r)
            ids = np.where(m)[0]
            if ids.size>1: perm[m] = rng.permutation(ids)
    return perm

def flow_curves_for_perm(ZA: np.ndarray, ZB: np.ndarray,
                         ZA_other: np.ndarray, ZB_other: np.ndarray,
                         trials: pd.DataFrame, wins: list[np.ndarray],
                         k: int, ridge: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nT = ZA.shape[0]
    permA = permute_within_CR(trials, nT, rng)
    permB = permute_within_CR(trials, nT, rng)
    ZA_p = ZA[permA]; ZAo_p = ZA_other[permA] if ZA_other is not None and ZA_other.shape[2]>0 else None
    ZB_p = ZB;        ZBo_p = ZB_other
    ZA_q = ZA;        ZAo_q = ZA_other
    ZB_q = ZB[permB]; ZBo_q = ZB_other[permB] if ZB_other is not None and ZB_other.shape[2]>0 else None
    fwd = np.full(len(wins), np.nan); rev = np.full(len(wins), np.nan)
    for w,m in enumerate(wins):
        Y, Xf, Xr = make_VAR_design(ZA_p[:,m,:], ZB_p[:,m,:],
                                    ZAo_p[:,m,:] if ZAo_p is not None else None,
                                    ZBo_p[:,m,:] if ZBo_p is not None else None,
                                    k=k, include_x0=True)
        if Y.shape[0]>(k+2): fwd[w]=gc_bits(Y,Xf,Xr,ridge)
        Y2, Xf2, Xr2 = make_VAR_design(ZB_q[:,m,:], ZA_q[:,m,:],
                                       ZBo_q[:,m,:] if ZBo_q is not None else None,
                                       ZAo_q[:,m,:] if ZAo_q is not None else None,
                                       k=k, include_x0=True)
        if Y2.shape[0]>(k+2): rev[w]=gc_bits(Y2,Xf2,Xr2,ridge)
    return fwd, rev

# ---------- Convolution / sliding integration ----------
def sliding_integrate(series: np.ndarray, centers: np.ndarray, int_win: float) -> np.ndarray:
    """Rectangular integration over time: convolve with ones kernel of length L=int_win/step."""
    if series.size == 0: return series
    if len(centers) >= 2:
        step = float(np.median(np.diff(centers)))
    else:
        step = int_win
    L = max(1, int(round(int_win / max(step, 1e-9))))
    kernel = np.ones(L, dtype=float)
    return np.convolve(series, kernel, mode="same")

def sliding_integrate_null(null_mat: np.ndarray, centers: np.ndarray, int_win: float) -> np.ndarray:
    """Apply sliding_integrate to each permutation row (perms x nW)."""
    if null_mat is None or null_mat.size == 0:
        return np.empty((0,0), dtype=float)
    out = np.empty_like(null_mat, dtype=float)
    for i in range(null_mat.shape[0]):
        out[i] = sliding_integrate(null_mat[i], centers, int_win)
    return out

def pvals_from_null(obs: np.ndarray, null_mat: np.ndarray) -> np.ndarray:
    """One-sided p-value per time: P(null >= obs)."""
    if null_mat is None or null_mat.size == 0:
        return np.full_like(obs, np.nan, dtype=float)
    ge = (null_mat >= obs[None, :]).sum(axis=0)
    return (1 + ge) / (1 + null_mat.shape[0])

# ---------- Core: full time-series + permutations ----------
def flow_timeseries_full(ZA: np.ndarray, ZB: np.ndarray,
                         ZA_other: np.ndarray, ZB_other: np.ndarray,
                         trials: pd.DataFrame, meta: dict,
                         win_w: float, win_step: float,
                         k: int, ridge: float,
                         perms: int, n_jobs: int, seed: int):
    wins, centers = build_windows(meta, win_w, win_step)
    fwd = np.full(len(wins), np.nan); rev = np.full(len(wins), np.nan)
    for w,m in enumerate(wins):
        Y, Xf, Xr = make_VAR_design(ZA[:,m,:], ZB[:,m,:],
                                    ZA_other[:,m,:] if ZA_other is not None and ZA_other.shape[2]>0 else None,
                                    ZB_other[:,m,:] if ZB_other is not None and ZB_other.shape[2]>0 else None,
                                    k=k, include_x0=True)
        if Y.shape[0]>(k+2): fwd[w]=gc_bits(Y,Xf,Xr,ridge)
        Y2, Xf2, Xr2 = make_VAR_design(ZB[:,m,:], ZA[:,m,:],
                                       ZB_other[:,m,:] if ZB_other is not None and ZB_other.shape[2]>0 else None,
                                       ZA_other[:,m,:] if ZA_other is not None and ZA_other.shape[2]>0 else None,
                                       k=k, include_x0=True)
        if Y2.shape[0]>(k+2): rev[w]=gc_bits(Y2,Xf2,Xr2,ridge)

    if perms>0:
        seeds = np.random.SeedSequence(seed).spawn(perms)
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(flow_curves_for_perm)(
                ZA, ZB, ZA_other, ZB_other, trials, wins, k, ridge, int(s.generate_state(1)[0])
            ) for s in seeds
        )
        fnull = np.vstack([r[0] for r in results])  # (perms, nW)
        rnull = np.vstack([r[1] for r in results])
        f_lo = np.nanpercentile(fnull, 2.5, axis=0); f_hi = np.nanpercentile(fnull, 97.5, axis=0)
        r_lo = np.nanpercentile(rnull, 2.5, axis=0); r_hi = np.nanpercentile(rnull, 97.5, axis=0)
    else:
        fnull=rnull=None; f_lo=f_hi=r_lo=r_hi=None

    return centers, fwd, rev, fnull, rnull, (f_lo,f_hi,r_lo,r_hi)

def integrate_in_band(centers: np.ndarray, series: np.ndarray, start: float, end: float) -> float:
    mask = band_mask(centers, start, end)
    return float(np.nansum(series[mask]))

# ---------- Plotters ----------
def plot_raw_pair(sid, A, B,
                  tC, C_fwd, C_rev, C_bands,
                  tR, R_fwd, R_rev, R_bands,
                  out_png: Path,
                  pC: np.ndarray|None=None, pR: np.ndarray|None=None,
                  annotate: bool=False, stride: int=4):
    C_flo, C_fhi, C_rlo, C_rhi = C_bands
    R_flo, R_fhi, R_rlo, R_rhi = R_bands
    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)

    ax = axes[0]
    if C_flo is not None: ax.fill_between(tC, C_flo, C_fhi, color="grey", alpha=0.15, label="perm 95% (forward)")
    if C_rlo is not None: ax.fill_between(tC, C_rlo, C_rhi, facecolor="none", edgecolor="grey", hatch="///", alpha=0.18, label="perm 95% (reverse)")
    ax.plot(tC, C_fwd, lw=2, label=f"{A}→{B} (C)")
    ax.plot(tC, C_rev, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (C, rev)")
    if annotate and pC is not None and len(pC)==len(tC):
        for i,(x,y) in enumerate(zip(tC, C_fwd)):
            if i % max(1,stride)==0 and np.isfinite(y) and np.isfinite(pC[i]):
                ax.text(x, y, f"{pC[i]:.2f}", fontsize=7, va="bottom", ha="center")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_ylabel("GC bits/bin (C)")
    ax.legend(frameon=False)

    ax = axes[1]
    if R_flo is not None: ax.fill_between(tR, R_flo, R_fhi, color="grey", alpha=0.15, label="perm 95% (forward)")
    if R_rlo is not None: ax.fill_between(tR, R_rlo, R_rhi, facecolor="none", edgecolor="grey", hatch="///", alpha=0.18, label="perm 95% (reverse)")
    ax.plot(tR, R_fwd, lw=2, label=f"{A}→{B} (R)")
    ax.plot(tR, R_rev, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (R, rev)")
    if annotate and pR is not None and len(pR)==len(tR):
        for i,(x,y) in enumerate(zip(tR, R_fwd)):
            if i % max(1,stride)==0 and np.isfinite(y) and np.isfinite(pR[i]):
                ax.text(x, y, f"{pR[i]:.2f}", fontsize=7, va="bottom", ha="center")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel("GC bits/bin (R)")
    ax.legend(frameon=False)

    fig.suptitle(f"Feature-specific flow (RAW) — {sid}: {A}→{B}")
    fig.tight_layout(rect=[0,0.02,1,0.96]); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_int_pair(sid, A, B, int_win, band,
                  tC, C_sl, C_sl_p, C_sl_mu, C_sl_lo, C_sl_hi,
                  tR, R_sl, R_sl_p, R_sl_mu, R_sl_lo, R_sl_hi,
                  IC, pIC, IR, pIR, out_png: Path,
                  annotate_p=False, stride=4, show_band_span=False):
    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)

    # Category integrated
    ax = axes[0]
    if show_band_span:
        ax.axvspan(band[0], band[1], color="yellow", alpha=0.12, label="pre-registered band")
    ax.fill_between(tC, C_sl_lo, C_sl_hi, color="grey", alpha=0.15, label="integrated null 95% (forward)")
    ax.plot(tC, C_sl_mu, color="grey", lw=1, label="integrated null mean (forward)")
    ax.plot(tC, C_sl, lw=2, label=f"{A}→{B} (C, INT {int_win*1000:.0f}ms)")
    if annotate_p and C_sl_p is not None:
        for i,(x,y) in enumerate(zip(tC, C_sl)):
            if i % max(1,stride)==0 and np.isfinite(y) and np.isfinite(C_sl_p[i]):
                ax.text(x, y, f"{C_sl_p[i]:.2f}", fontsize=7, va="bottom", ha="center")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_ylabel("Integrated GC (C)")
    ax.legend(frameon=False)
    ax.text(0.01, 0.95, f"Band ∑GC: {IC:.4f}  p={pIC:.3g}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    # Direction integrated
    ax = axes[1]
    if show_band_span:
        ax.axvspan(band[0], band[1], color="yellow", alpha=0.12, label="pre-registered band")
    ax.fill_between(tR, R_sl_lo, R_sl_hi, color="grey", alpha=0.15, label="integrated null 95% (forward)")
    ax.plot(tR, R_sl_mu, color="grey", lw=1, label="integrated null mean (forward)")
    ax.plot(tR, R_sl, lw=2, label=f"{A}→{B} (R, INT {int_win*1000:.0f}ms)")
    if annotate_p and R_sl_p is not None:
        for i,(x,y) in enumerate(zip(tR, R_sl)):
            if i % max(1,stride)==0 and np.isfinite(y) and np.isfinite(R_sl_p[i]):
                ax.text(x, y, f"{R_sl_p[i]:.2f}", fontsize=7, va="bottom", ha="center")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel("Integrated GC (R)")
    ax.legend(frameon=False)

    fig.suptitle(f"Feature-specific flow (INT) — {sid}: {A}→{B}")
    fig.tight_layout(rect=[0,0.02,1,0.96]); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------- One pair ----------
def run_one_pair(sid: int, A: str, B: str,
                 axes_dir: Path, cache_dir: Path,
                 win: float, step: float, k: int, ridge: float,
                 perms: int, n_jobs: int, seed: int,
                 band_start: float, band_end: float,
                 int_win: float,
                 out_tag: str = "", skip_if_exists: bool = False,
                 annotate_p_raw: bool = False, annotate_p_int: bool = False, p_text_stride: int = 4,
                 show_band_span: bool = False):
    # Load axes (tag-aware)
    ZC_A, ZR_A, metaA = load_axes(sid, A, axes_dir, out_tag)
    ZC_B, ZR_B, metaB = load_axes(sid, B, axes_dir, out_tag)
    if ZC_A.shape[1] != ZC_B.shape[1]: raise SystemExit("Mismatched bins A vs B.")
    trials = load_trials(cache_dir, sid, B)

    # Output dirs
    out_dir = axes_dir / f"{sid}"
    if out_tag: out_dir = out_dir / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- CATEGORY (condition on R) ----------
    tC, C_fwd, C_rev, C_fnull, C_rnull, (C_lo, C_hi, C_rlo, C_rhi) = \
        flow_timeseries_full(
            ZA=ZC_A, ZB=ZC_B, ZA_other=ZR_A, ZB_other=ZR_B, trials=trials, meta=metaB,
            win_w=win, win_step=step, k=k, ridge=ridge, perms=perms, n_jobs=n_jobs, seed=seed+101
        )

    # RAW per-time p-values (forward)
    def pvals_from_null_point(obs, null_mat):
        if null_mat is None or null_mat.size == 0:
            return np.full_like(obs, np.nan, dtype=float)
        ge = (null_mat >= obs[None, :]).sum(axis=0)
        return (1 + ge) / (1 + null_mat.shape[0])
    C_p_raw = pvals_from_null_point(C_fwd, C_fnull)

    # Single pre-registered band integration & p-values (scalar)
    IC_obs = integrate_in_band(tC, C_fwd, band_start, band_end)
    IC_null = np.nansum(C_fnull[:, (tC>=band_start)&(tC<band_end)], axis=1) if C_fnull is not None and C_fnull.size else np.array([])
    pIC = float((1 + np.sum(IC_null >= IC_obs)) / (1 + IC_null.size)) if IC_null.size else np.nan

    # Sliding integrated series and per-time p-values
    C_fwd_sl = sliding_integrate(C_fwd, tC, int_win)
    C_null_sl = sliding_integrate_null(C_fnull, tC, int_win) if C_fnull is not None and C_fnull.size else np.empty((0,len(tC)))
    C_sl_mu = np.nanmean(C_null_sl, axis=0) if C_null_sl.size else np.full_like(C_fwd_sl, np.nan)
    C_sl_lo = np.nanpercentile(C_null_sl, 2.5, axis=0) if C_null_sl.size else np.full_like(C_fwd_sl, np.nan)
    C_sl_hi = np.nanpercentile(C_null_sl, 97.5, axis=0) if C_null_sl.size else np.full_like(C_fwd_sl, np.nan)
    C_p_int = pvals_from_null(C_fwd_sl, C_null_sl)

    # ---------- DIRECTION (condition on C) ----------
    if ZR_A.shape[2]==0 or ZR_B.shape[2]==0:
        tR = tC.copy()
        R_fwd = R_rev = np.full_like(tR, np.nan, dtype=float)
        R_lo = R_hi = R_rlo = R_rhi = None
        R_fnull = R_rnull = np.empty((0,len(tR)))
        IR_obs = np.nan; IR_null = np.array([]); pIR = np.nan
        R_fwd_sl = np.full_like(tR, np.nan); R_null_sl = np.empty((0,len(tR)))
        R_sl_mu = R_sl_lo = R_sl_hi = np.full_like(tR, np.nan)
        R_p_raw = np.full_like(tR, np.nan, dtype=float)
        R_p_int = np.full_like(tR, np.nan, dtype=float)
    else:
        tR, R_fwd, R_rev, R_fnull, R_rnull, (R_lo, R_hi, R_rlo, R_rhi) = \
            flow_timeseries_full(
                ZA=ZR_A, ZB=ZR_B, ZA_other=ZC_A, ZB_other=ZC_B, trials=trials, meta=metaB,
                win_w=win, win_step=step, k=k, ridge=ridge, perms=perms, n_jobs=n_jobs, seed=seed+202
            )
        R_p_raw = pvals_from_null_point(R_fwd, R_fnull)

        IR_obs = integrate_in_band(tR, R_fwd, band_start, band_end)
        IR_null = np.nansum(R_fnull[:, (tR>=band_start)&(tR<band_end)], axis=1) if R_fnull is not None and R_fnull.size else np.array([])
        pIR = float((1 + np.sum(IR_null >= IR_obs)) / (1 + IR_null.size)) if IR_null.size else np.nan

        R_fwd_sl = sliding_integrate(R_fwd, tR, int_win)
        R_null_sl = sliding_integrate_null(R_fnull, tR, int_win) if R_fnull is not None and R_fnull.size else np.empty((0,len(tR)))
        R_sl_mu = np.nanmean(R_null_sl, axis=0) if R_null_sl.size else np.full_like(R_fwd_sl, np.nan)
        R_sl_lo = np.nanpercentile(R_null_sl, 2.5, axis=0) if R_null_sl.size else np.full_like(R_fwd_sl, np.nan)
        R_sl_hi = np.nanpercentile(R_null_sl, 97.5, axis=0) if R_null_sl.size else np.full_like(R_fwd_sl, np.nan)
        R_p_int = pvals_from_null(R_fwd_sl, R_null_sl)

    # ----- Save everything (including full nulls and per-time p) -----
    out_npz = out_dir / f"flow_timeseriesINT_{A}to{B}.npz"
    out_raw_png = out_dir / f"flow_timeseriesRAW_{A}to{B}.png"
    out_int_png = out_dir / f"flow_timeseriesINT_{A}to{B}.png"
    if skip_if_exists and out_npz.exists() and out_raw_png.exists() and out_int_png.exists():
        print(f("[skip] {sid} {A}->{B} already exists in tag '{out_dir.name}'"))
        return

    np.savez_compressed(
        out_npz,
        # time axes
        tC=tC, tR=tR,
        # RAW curves + bands + full nulls + per-time p
        C_fwd=C_fwd, C_rev=C_rev, C_lo=C_lo, C_hi=C_hi, C_rlo=C_rlo, C_rhi=C_rhi, C_fnull=C_fnull, C_rnull=C_rnull,
        R_fwd=R_fwd, R_rev=R_rev, R_lo=R_lo, R_hi=R_hi, R_rlo=R_rlo, R_rhi=R_rhi, R_fnull=R_fnull, R_rnull=R_rnull,
        C_p_raw=C_p_raw, R_p_raw=R_p_raw,
        # Pre-registered band integration (scalar)
        band=np.array([band_start, band_end]),
        IC_fwd=IC_obs, IC_fwd_null=IC_null, pC_fwd=float(pIC),
        IR_fwd=IR_obs, IR_fwd_null=IR_null, pR_fwd=float(pIR),
        # Sliding integrated series + per-time p + full integrated nulls
        int_win=float(int_win),
        C_fwd_sl=C_fwd_sl, C_null_sl=C_null_sl, C_sl_mu=C_sl_mu, C_sl_lo=C_sl_lo, C_sl_hi=C_sl_hi, C_p_int=C_p_int,
        R_fwd_sl=R_fwd_sl, R_null_sl=R_null_sl, R_sl_mu=R_sl_mu, R_sl_lo=R_sl_lo, R_sl_hi=R_sl_hi, R_p_int=R_p_int,
        meta=json.dumps({"sid": sid, "A": A, "B": B, "win": win, "step": step, "k": k,
                         "ridge": ridge, "perms": perms, "n_jobs": n_jobs, "tag": out_tag})
    )

    # ----- Plots -----
    plot_raw_pair(sid, A, B,
                  tC, C_fwd, C_rev, (C_lo, C_hi, C_rlo, C_rhi),
                  tR, R_fwd, R_rev, (R_lo, R_hi, R_rlo, R_rhi),
                  out_raw_png,
                  pC=C_p_raw, pR=R_p_raw, annotate=annotate_p_raw, stride=p_text_stride)

    plot_int_pair(sid, A, B, int_win, (band_start, band_end),
                  tC, C_fwd_sl, C_p_int, C_sl_mu, C_sl_lo, C_sl_hi,
                  tR, R_fwd_sl, R_p_int, R_sl_mu, R_sl_lo, R_sl_hi,
                  IC_obs, pIC, IR_obs, pIR,
                  out_int_png,
                  annotate_p=annotate_p_int, stride=p_text_stride, show_band_span=show_band_span)

    print(f"[done] {sid} {A}->{B}  RAW+INT saved -> {out_dir}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--A", type=str, help="Source area")
    ap.add_argument("--B", type=str, help="Target area")
    ap.add_argument("--all_pairs", action="store_true")
    ap.add_argument("--axes_dir",  type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))

    # GC params
    ap.add_argument("--win",   type=float, default=0.16)
    ap.add_argument("--step",  type=float, default=0.02)
    ap.add_argument("--k",     type=int,   default=5)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int,   default=500)
    ap.add_argument("--n_jobs", type=int,  default=8)
    ap.add_argument("--seed",  type=int,   default=123)

    # Pre-registered band (scalar integrated test)
    ap.add_argument("--band_start", type=float, default=0.12)
    ap.add_argument("--band_end",   type=float, default=0.28)

    # Sliding integrated window (rectangular)
    ap.add_argument("--int_win", type=float, default=None,
                    help="Width (s) for sliding integration; default = band_end - band_start")

    # Tagging / skipping
    ap.add_argument("--out_tag", type=str, default="", help="Write under results/session/<sid>/<tag>/")
    ap.add_argument("--skip_if_exists", action="store_true", default=False)

    # Annotations & band span
    ap.add_argument("--annotate_p_raw", action="store_true", default=False,
                    help="Annotate per-time p-values on RAW forward curve")
    ap.add_argument("--annotate_p_int", action="store_true", default=False,
                    help="Annotate per-time p-values on INT forward curve")
    ap.add_argument("--p_text_stride", type=int, default=4,
                    help="Annotate every N points (default 4)")
    ap.add_argument("--show_band_span", action="store_true", default=False,
                    help="Draw the static pre-registered band span on INT plot")

    args = ap.parse_args()
    if args.int_win is None:
        args.int_win = max(1e-3, args.band_end - args.band_start)

    def pairs_for_session():
        areas = [a for a in detect_areas(args.cache_dir, args.sid) if a in VALID]
        return [(a,b) for a in areas for b in areas if a!=b]

    if args.all_pairs:
        pairs = pairs_for_session()
        print(f"[info] sid={args.sid} running {len(pairs)} ordered pairs: {pairs}")
        for (A,B) in pairs:
            run_one_pair(args.sid, A, B, args.axes_dir, args.cache_dir,
                         args.win, args.step, args.k, args.ridge,
                         args.perms, args.n_jobs, args.seed,
                         args.band_start, args.band_end, args.int_win,
                         out_tag=args.out_tag, skip_if_exists=args.skip_if_exists,
                         annotate_p_raw=args.annotate_p_raw, annotate_p_int=args.annotate_p_int,
                         p_text_stride=args.p_text_stride, show_band_span=args.show_band_span)
    else:
        if not args.A or not args.B:
            raise SystemExit("Provide --A and --B, or use --all_pairs.")
        run_one_pair(args.sid, args.A, args.B, args.axes_dir, args.cache_dir,
                     args.win, args.step, args.k, args.ridge,
                     args.perms, args_n_jobs=args.n_jobs, seed=args.seed,  # minor guard
                     band_start=args.band_start, band_end=args.band_end, int_win=args.int_win,
                     out_tag=args.out_tag, skip_if_exists=args.skip_if_exists,
                     annotate_p_raw=args.annotate_p_raw, annotate_p_int=args.annotate_p_int,
                     p_text_stride=args.p_text_stride, show_band_span=args.show_band_span)

if __name__ == "__main__":
    main()
