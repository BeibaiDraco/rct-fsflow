#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_induced_fsflow_timesliding.py

Single-session feature-specific flow with "communication-first" settings (toggleable):
  • Induced GC (subtract trial-mean per time bin; optional time smoothing)
  • Lagged only (exclude contemporaneous X_t)  [--no_x0 True by default]
  • Condition only on target's other feature (lighter conditioning)  [--cond_b_only True by default]
  • Optional PT covariate as exogenous regressor per trial (broadcast across time)
  • Integrated tests (pre-registered bands per feature) + sliding integrated series with per-time p-values
  • Saves full permutation null matrices (RAW & sliding-integrated)

Outputs per pair under results/session/<sid>/<tag>/:
  - induced_flow_RAW_<A>to<B>.png
  - induced_flow_INT_<A>to<B>.png
  - induced_flow_<A>to<B>.npz  (rich artifact)
  - (optional) induced_summary.csv appended with scalar integrals and p-values

Example:
  python 12_induced_fsflow_timesliding.py --sid 20200926 --all_pairs \
    --win 0.16 --step 0.02 --k 2 --ridge 1e-2 --perms 500 --n_jobs 16 \
    --bandC_start 0.12 --bandC_end 0.28 --bandR_start 0.08 --bandR_end 0.20 \
    --int_win 0.16 --evoked_sigma_ms 10 \
    --out_tag induced_k2_win160_p500 --skip_if_exists \
    --annotate_p --pt_cov
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
from scipy.ndimage import gaussian_filter1d

VALID = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

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

# ---------- Helpers ----------
def remove_evoked(Z: np.ndarray, smooth_sigma_bins: int = 0) -> np.ndarray:
    if Z.size == 0: return Z
    ev = np.nanmean(Z, axis=0)  # (T,d)
    if smooth_sigma_bins and ev.size:
        ev = np.stack([gaussian_filter1d(ev[:, j], sigma=smooth_sigma_bins, mode="nearest")
                       for j in range(ev.shape[1])], axis=1)
    return Z - ev[None, :, :]

def compute_PT(trials: pd.DataFrame) -> np.ndarray:
    sacc_on = np.asarray(trials.get("Align_to_sacc_on", np.nan), float)
    fix_off = np.asarray(trials.get("Align_to_fix_off", np.nan), float)
    cat_on  = np.asarray(trials.get("Align_to_cat_stim_on", np.nan), float)
    RT  = sacc_on - fix_off
    gap = cat_on  - fix_off
    return RT - gap  # PT

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

# ---------- VAR / GC ----------
def make_VAR_design(ZA: np.ndarray, ZB: np.ndarray,
                    ZA_other: np.ndarray|None,
                    ZB_other: np.ndarray|None,
                    Zextra: np.ndarray|None,
                    k: int, include_x0: bool=False):
    """
    Y_t ~ [Y_{t-1..t-k}, X_{t-1..t-k} (+X_t if include_x0), ZB_other (current+lags), Zextra (current)]
    Shapes: (nT,T,d)
    """
    nT, T, _ = ZA.shape
    rowsY, rowsF, rowsR = [], [], []
    for tr in range(nT):
        Xa, Yb = ZA[tr], ZB[tr]
        Ob = ZB_other[tr] if ZB_other is not None and ZB_other.shape[2]>0 else None
        Xz = Zextra[tr] if Zextra is not None and Zextra.shape[2]>0 else None
        for t in range(k, T):
            past_y = np.concatenate([Yb[t-i] for i in range(1,k+1)], axis=0) if k>0 else np.zeros((0,))
            hx = []
            if include_x0: hx.append(Xa[t])
            if k>0: hx.extend([Xa[t-i] for i in range(1,k+1)])
            hx = np.concatenate(hx, axis=0) if hx else np.zeros((0,))
            hz = []
            if Ob is not None:
                hz.append(Ob[t]); 
                if k>0: hz.extend([Ob[t-i] for i in range(1,k+1)])
            if Xz is not None:
                hz.append(Xz[t])
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

# ---------- Windows / integration ----------
def build_windows(meta: dict, win_w: float, win_step: float):
    t = time_axis(meta)
    anchors = np.arange(t[0], t[-1] - win_w + 1e-9, win_step)
    wins, centers = [], []
    for a in anchors:
        m = (t >= a) & (t < a + win_w)
        if m.sum() >= 2:
            wins.append(m); centers.append(a + win_w/2)
    return wins, np.array(centers,float)

def sliding_integrate(series: np.ndarray, centers: np.ndarray, int_win: float) -> np.ndarray:
    if series.size == 0: return series
    step = float(np.median(np.diff(centers))) if len(centers) >= 2 else int_win
    L = max(1, int(round(int_win / max(step, 1e-9))))
    kernel = np.ones(L, dtype=float)
    return np.convolve(series, kernel, mode="same")

def sliding_integrate_null(null_mat: np.ndarray, centers: np.ndarray, int_win: float) -> np.ndarray:
    if null_mat is None or null_mat.size == 0:
        return np.empty((0,0), dtype=float)
    out = np.empty_like(null_mat, dtype=float)
    for i in range(null_mat.shape[0]):
        out[i] = sliding_integrate(null_mat[i], centers, int_win)
    return out

def pvals_from_null(obs: np.ndarray, null_mat: np.ndarray) -> np.ndarray:
    if null_mat is None or null_mat.size == 0:
        return np.full_like(obs, np.nan, dtype=float)
    ge = (null_mat >= obs[None, :]).sum(axis=0)
    return (1 + ge) / (1 + null_mat.shape[0])

# ---------- Permutation worker (cond-B-only; lagged-only toggle) ----------
def flow_curves_for_perm(ZA: np.ndarray, ZB: np.ndarray,
                         ZB_other: np.ndarray,
                         Zextra: np.ndarray,
                         trials: pd.DataFrame, wins: list[np.ndarray],
                         k: int, ridge: float, seed: int,
                         include_x0: bool):
    rng = np.random.default_rng(seed)
    nT = ZA.shape[0]
    permA = permute_within_CR(trials, nT, rng)
    permB = permute_within_CR(trials, nT, rng)

    ZA_p = ZA[permA]
    ZB_p = ZB
    ZBo_p = ZB_other
    Zx_p  = Zextra if Zextra is not None else None

    ZA_q = ZA
    ZB_q = ZB[permB]
    ZBo_q = ZB_other[permB] if ZB_other is not None and ZB_other.shape[2]>0 else None
    Zx_q  = Zextra if Zextra is not None else None

    fwd = np.full(len(wins), np.nan); rev = np.full(len(wins), np.nan)
    for w,m in enumerate(wins):
        Y, Xf, Xr = make_VAR_design(ZA_p[:,m,:], ZB_p[:,m,:],
                                    ZA_other=None,
                                    ZB_other=ZBo_p[:,m,:] if ZBo_p is not None else None,
                                    Zextra=Zx_p[:,m,:] if Zx_p is not None else None,
                                    k=k, include_x0=include_x0)
        if Y.shape[0] > (k+2): fwd[w] = gc_bits(Y, Xf, Xr, ridge)
        Y2, Xf2, Xr2 = make_VAR_design(ZB_q[:,m,:], ZA_q[:,m,:],
                                       ZA_other=None,
                                       ZB_other=ZBo_q[:,m,:] if ZBo_q is not None else None,
                                       Zextra=Zx_q[:,m,:] if Zx_q is not None else None,
                                       k=k, include_x0=include_x0)
        if Y2.shape[0] > (k+2): rev[w] = gc_bits(Y2, Xf2, Xr2, ridge)
    return fwd, rev

def flow_timeseries_induced(ZA: np.ndarray, ZB: np.ndarray,
                            ZB_other: np.ndarray,
                            Zextra: np.ndarray,
                            trials: pd.DataFrame, meta: dict,
                            win_w: float, win_step: float,
                            k: int, ridge: float,
                            perms: int, n_jobs: int, seed: int,
                            include_x0: bool):
    wins, centers = build_windows(meta, win_w, win_step)

    fwd = np.full(len(wins), np.nan); rev = np.full(len(wins), np.nan)
    for w,m in enumerate(wins):
        Y, Xf, Xr = make_VAR_design(ZA[:,m,:], ZB[:,m,:],
                                    ZA_other=None,
                                    ZB_other=ZB_other[:,m,:] if ZB_other is not None and ZB_other.shape[2]>0 else None,
                                    Zextra=Zextra[:,m,:] if Zextra is not None and Zextra.shape[2]>0 else None,
                                    k=k, include_x0=include_x0)
        if Y.shape[0]>(k+2): fwd[w]=gc_bits(Y,Xf,Xr,ridge)
        Y2, Xf2, Xr2 = make_VAR_design(ZB[:,m,:], ZA[:,m,:],
                                       ZA_other=None,
                                       ZB_other=ZB_other[:,m,:] if ZB_other is not None and ZB_other.shape[2]>0 else None,
                                       Zextra=Zextra[:,m,:] if Zextra is not None and Zextra.shape[2]>0 else None,
                                       k=k, include_x0=include_x0)
        if Y2.shape[0]>(k+2): rev[w]=gc_bits(Y2,Xf2,Xr2,ridge)

    if perms>0:
        seeds = np.random.SeedSequence(seed).spawn(perms)
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(flow_curves_for_perm)(
                ZA, ZB, ZB_other, Zextra, trials, wins, k, ridge, int(s.generate_state(1)[0]), include_x0
            ) for s in seeds
        )
        fnull = np.vstack([r[0] for r in results])
        rnull = np.vstack([r[1] for r in results])
        f_lo = np.nanpercentile(fnull, 2.5, axis=0); f_hi = np.nanpercentile(fnull, 97.5, axis=0)
        r_lo = np.nanpercentile(rnull, 2.5, axis=0); r_hi = np.nanpercentile(rnull, 97.5, axis=0)
    else:
        fnull=rnull=None; f_lo=f_hi=r_lo=r_hi=None

    return centers, fwd, rev, fnull, rnull, (f_lo, f_hi, r_lo, r_hi)

# ---------- Plotters ----------
def plot_raw(sid, A, B,
             tC, C_fwd, C_rev, C_bands,
             tR, R_fwd, R_rev, R_bands,
             out_png: Path):
    C_flo, C_fhi, C_rlo, C_rhi = C_bands
    R_flo, R_fhi, R_rlo, R_rhi = R_bands
    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)
    ax = axes[0]
    if C_flo is not None: ax.fill_between(tC, C_flo, C_fhi, color="grey", alpha=0.15, label="perm 95% (forward)")
    if C_rlo is not None: ax.fill_between(tC, C_rlo, C_rhi, facecolor="none", edgecolor="grey", hatch="///", alpha=0.18, label="perm 95% (reverse)")
    ax.plot(tC, C_fwd, lw=2, label=f"{A}→{B} (C)")
    ax.plot(tC, C_rev, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (C, rev)")
    ax.axvline(0.0, color="k", ls=":", lw=1); ax.set_ylabel("GC bits/bin (C)"); ax.legend(frameon=False)
    ax = axes[1]
    if R_flo is not None: ax.fill_between(tR, R_flo, R_fhi, color="grey", alpha=0.15, label="perm 95% (forward)")
    if R_rlo is not None: ax.fill_between(tR, R_rlo, R_rhi, facecolor="none", edgecolor="grey", hatch="///", alpha=0.18, label="perm 95% (reverse)")
    ax.plot(tR, R_fwd, lw=2, label=f"{A}→{B} (R)")
    ax.plot(tR, R_rev, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (R, rev)")
    ax.axvline(0.0, color="k", ls=":", lw=1); ax.set_xlabel("time (s) from cat_stim_on"); ax.set_ylabel("GC bits/bin (R)"); ax.legend(frameon=False)
    fig.suptitle(f"Induced feature-specific flow (RAW) — {sid}: {A}→{B}"); fig.tight_layout(rect=[0,0.02,1,0.96]); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_int(sid, A, B, int_win,
             tC, C_sl, C_sl_mu, C_sl_lo, C_sl_hi, IC, pIC,
             tR, R_sl, R_sl_mu, R_sl_lo, R_sl_hi, IR, pIR,
             bandC, bandR, out_png: Path, annotate_p: bool=False, C_p=None, R_p=None):
    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)
    ax = axes[0]
    ax.fill_between(tC, C_sl_lo, C_sl_hi, color="grey", alpha=0.15, label="integrated null 95% (C)")
    ax.plot(tC, C_sl_mu, color="grey", lw=1, label="integrated null mean (C)")
    ax.plot(tC, C_sl, lw=2, label=f"{A}→{B} (C, INT {int_win*1000:.0f}ms)")
    if annotate_p and C_p is not None:
        for x,y,p in zip(tC, C_sl, C_p):
            if np.isfinite(p): ax.text(x, y, f"{p:.2f}", fontsize=7, va="bottom", ha="center")
    ax.axvline(0.0, color="k", ls=":", lw=1); ax.set_ylabel("Integrated GC (C)"); ax.legend(frameon=False)
    ax.text(0.01,0.95, f"∑GC(C)={IC:.4f}  p={pIC:.3g}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    ax = axes[1]
    ax.fill_between(tR, R_sl_lo, R_sl_hi, color="grey", alpha=0.15, label="integrated null 95% (R)")
    ax.plot(tR, R_sl_mu, color="grey", lw=1, label="integrated null mean (R)")
    ax.plot(tR, R_sl, lw=2, label=f"{A}→{B} (R, INT {int_win*1000:.0f}ms)")
    if annotate_p and R_p is not None:
        for x,y,p in zip(tR, R_sl, R_p):
            if np.isfinite(p): ax.text(x, y, f"{p:.2f}", fontsize=7, va="bottom", ha="center")
    ax.axvline(0.0, color="k", ls=":", lw=1); ax.set_xlabel("time (s) from cat_stim_on"); ax.set_ylabel("Integrated GC (R)"); ax.legend(frameon=False)
    fig.suptitle(f"Induced feature-specific flow (INT) — {sid}: {A}→{B}"); fig.tight_layout(rect=[0,0.02,1,0.96]); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------- One pair ----------
def run_one_pair(sid: int, A: str, B: str,
                 axes_dir: Path, cache_dir: Path,
                 win: float, step: float, k: int, ridge: float,
                 perms: int, n_jobs: int, seed: int,
                 bandC: tuple[float,float], bandR: tuple[float,float],
                 int_win: float, evoked_sigma_ms: float,
                 out_tag: str = "", skip_if_exists: bool = False, annotate_p: bool=False,
                 no_x0: bool=True, cond_b_only: bool=True, pt_cov: bool=False, save_csv: bool=True):
    # Load
    ZC_A, ZR_A, metaA = load_axes(sid, A, axes_dir, out_tag)
    ZC_B, ZR_B, metaB = load_axes(sid, B, axes_dir, out_tag)
    if ZC_A.shape[1] != ZC_B.shape[1]:
        raise SystemExit("Mismatched bins A vs B.")
    trials = load_trials(cache_dir, sid, B)
    bs = float(metaB.get("bin_size_s", 0.010))
    sig_bins = int(round(evoked_sigma_ms / (bs*1000.0))) if evoked_sigma_ms > 0 else 0

    # Induced
    ZC_A = remove_evoked(ZC_A, sig_bins); ZC_B = remove_evoked(ZC_B, sig_bins)
    ZR_A = remove_evoked(ZR_A, sig_bins); ZR_B = remove_evoked(ZR_B, sig_bins)

    # Exogenous PT
    Zextra = None
    if pt_cov:
        PT = compute_PT(trials)
        T = ZC_A.shape[1]
        Zextra = np.repeat(PT[:,None], T, axis=1)[:,:,None]

    include_x0 = (not no_x0)

    out_dir = axes_dir / f"{sid}"
    if out_tag: out_dir = out_dir / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # CATEGORY: A(ZC)->B(ZC), cond on B(ZR) only
    ZB_other_C = ZR_B if cond_b_only else np.concatenate([ZR_A, ZR_B], axis=2)
    tC, C_fwd, C_rev, C_fnull, C_rnull, (C_lo, C_hi, C_rlo, C_rhi) = \
        flow_timeseries_induced(ZC_A, ZC_B, ZB_other_C, Zextra, trials, metaB,
                                win, step, k, ridge, perms, n_jobs, seed+11, include_x0)

    IC = integrate_band(tC, C_fwd, *bandC)
    C_mask = (tC >= bandC[0]) & (tC < bandC[1])
    IC_null = np.nansum(C_fnull[:, C_mask], axis=1) if C_fnull is not None and C_fnull.size else np.array([])
    pIC = float((1 + np.sum(IC_null >= IC)) / (1 + IC_null.size)) if IC_null.size else np.nan

    C_sl = sliding_integrate(C_fwd, tC, int_win)
    C_null_sl = sliding_integrate_null(C_fnull, tC, int_win) if C_fnull is not None and C_fnull.size else np.empty((0,len(tC)))
    C_sl_mu = np.nanmean(C_null_sl, axis=0) if C_null_sl.size else np.full_like(C_sl, np.nan)
    C_sl_lo = np.nanpercentile(C_null_sl, 2.5, axis=0) if C_null_sl.size else np.full_like(C_sl, np.nan)
    C_sl_hi = np.nanpercentile(C_null_sl, 97.5, axis=0) if C_null_sl.size else np.full_like(C_sl, np.nan)
    C_p = pvals_from_null(C_sl, C_null_sl)

    # DIRECTION: A(ZR)->B(ZR), cond on B(ZC) only
    if ZR_A.shape[2]==0 or ZR_B.shape[2]==0:
        tR = tC.copy()
        R_fwd = R_rev = np.full_like(tR, np.nan)
        R_lo = R_hi = R_rlo = R_rhi = None
        R_fnull = R_rnull = np.empty((0,len(tR)))
        IR = np.nan; IR_null = np.array([]); pIR = np.nan
        R_sl = np.full_like(tR, np.nan); R_null_sl = np.empty((0,len(tR)))
        R_sl_mu = R_sl_lo = R_sl_hi = np.full_like(tR, np.nan)
        R_p = np.full_like(tR, np.nan, float)
    else:
        ZB_other_R = ZC_B if cond_b_only else np.concatenate([ZC_A, ZC_B], axis=2)
        tR, R_fwd, R_rev, R_fnull, R_rnull, (R_lo, R_hi, R_rlo, R_rhi) = \
            flow_timeseries_induced(ZR_A, ZR_B, ZB_other_R, Zextra, trials, metaB,
                                    win, step, k, ridge, perms, n_jobs, seed+22, include_x0)

        IR = integrate_band(tR, R_fwd, *bandR)
        R_mask = (tR >= bandR[0]) & (tR < bandR[1])
        IR_null = np.nansum(R_fnull[:, R_mask], axis=1) if R_fnull is not None and R_fnull.size else np.array([])
        pIR = float((1 + np.sum(IR_null >= IR)) / (1 + IR_null.size)) if IR_null.size else np.nan

        R_sl = sliding_integrate(R_fwd, tR, int_win)
        R_null_sl = sliding_integrate_null(R_fnull, tR, int_win) if R_fnull is not None and R_fnull.size else np.empty((0,len(tR)))
        R_sl_mu = np.nanmean(R_null_sl, axis=0) if R_null_sl.size else np.full_like(R_sl, np.nan)
        R_sl_lo = np.nanpercentile(R_null_sl, 2.5, axis=0) if R_null_sl.size else np.full_like(R_sl, np.nan)
        R_sl_hi = np.nanpercentile(R_null_sl, 97.5, axis=0) if R_null_sl.size else np.full_like(R_sl, np.nan)
        R_p = pvals_from_null(R_sl, R_null_sl)

    # Save NPZ
    out_npz = out_dir / f"induced_flow_{A}to{B}.npz"
    out_raw_png = out_dir / f"induced_flow_RAW_{A}to{B}.png"
    out_int_png = out_dir / f"induced_flow_INT_{A}to{B}.png"
    if skip_if_exists and out_npz.exists() and out_raw_png.exists() and out_int_png.exists():
        print(f"[skip] {sid} {A}->{B} already exists in {out_dir}")
        return

    np.savez_compressed(
        out_npz,
        tC=tC, C_fwd=C_fwd, C_rev=C_rev, C_lo=C_lo, C_hi=C_hi, C_rlo=C_rlo, C_rhi=C_rhi,
        tR=tR, R_fwd=R_fwd, R_rev=R_rev, R_lo=R_lo, R_hi=R_hi, R_rlo=R_rlo, R_rhi=R_rhi,
        C_fnull=C_fnull, C_rnull=C_rnull, R_fnull=R_fnull, R_rnull=R_rnull,
        int_win=float(int_win),
        C_fwd_sl=C_sl, C_null_sl=C_null_sl, C_sl_mu=C_sl_mu, C_sl_lo=C_sl_lo, C_sl_hi=C_sl_hi, C_p_t=C_p,
        R_fwd_sl=R_sl, R_null_sl=R_null_sl, R_sl_mu=R_sl_mu, R_sl_lo=R_sl_lo, R_sl_hi=R_sl_hi, R_p_t=R_p,
        bandC=np.array(bandC), IC_fwd=IC, IC_fwd_null=IC_null, pC_fwd=float(pIC),
        bandR=np.array(bandR), IR_fwd=IR, IR_fwd_null=IR_null, pR_fwd=float(pIR),
        meta=json.dumps({"sid": sid, "A": A, "B": B, "win": win, "step": step, "k": k,
                         "ridge": ridge, "perms": perms, "n_jobs": n_jobs,
                         "tag": out_tag, "evoked_sigma_ms": evoked_sigma_ms,
                         "no_x0": no_x0, "cond_b_only": cond_b_only, "pt_cov": pt_cov})
    )

    # Plots
    plot_raw(sid, A, B,
             tC, C_fwd, C_rev, (C_lo, C_hi, C_rlo, C_rhi),
             tR, R_fwd, R_rev, (R_lo, R_hi, R_rlo, R_rhi),
             out_raw_png)

    plot_int(sid, A, B, int_win,
             tC, C_sl, C_sl_mu, C_sl_lo, C_sl_hi, IC, pIC,
             tR, R_sl, R_sl_mu, R_sl_lo, R_sl_hi, IR, pIR,
             bandC, bandR, out_int_png, annotate_p=annotate_p, C_p=C_p, R_p=R_p)

    # Optional CSV summary (one row per pair)
    if save_csv:
        summ = pd.DataFrame([{
            "sid": sid, "A": A, "B": B,
            "IC": IC, "pIC": pIC, "IR": IR, "pIR": pIR,
            "win": win, "step": step, "k": k, "ridge": ridge,
            "int_win": int_win, "perms": perms,
            "bandC_start": bandC[0], "bandC_end": bandC[1],
            "bandR_start": bandR[0], "bandR_end": bandR[1],
            "no_x0": no_x0, "cond_b_only": cond_b_only, "pt_cov": pt_cov,
            "evoked_sigma_ms": evoked_sigma_ms
        }])
        csv_path = out_dir / "induced_summary.csv"
        hdr = (not csv_path.exists())
        summ.to_csv(csv_path, mode="a", header=hdr, index=False)

    print(f"[done] {sid} {A}->{B}  saved RAW+INT → {out_dir}")

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
    ap.add_argument("--k",     type=int,   default=2)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int,   default=500)
    ap.add_argument("--n_jobs", type=int,  default=8)
    ap.add_argument("--seed",  type=int,   default=123)
    # bands (separate for C and R)
    ap.add_argument("--bandC_start", type=float, default=0.12)
    ap.add_argument("--bandC_end",   type=float, default=0.28)
    ap.add_argument("--bandR_start", type=float, default=0.08)
    ap.add_argument("--bandR_end",   type=float, default=0.20)
    # sliding integration window
    ap.add_argument("--int_win", type=float, default=0.16)
    # evoked removal
    ap.add_argument("--evoked_sigma_ms", type=float, default=10.0)
    # toggles
    ap.add_argument("--no_x0", action="store_true", default=True)
    ap.add_argument("--cond_b_only", action="store_true", default=True)
    ap.add_argument("--pt_cov", action="store_true", default=False)
    # tagging / skipping
    ap.add_argument("--out_tag", type=str, default="")
    ap.add_argument("--skip_if_exists", action="store_true", default=False)
    ap.add_argument("--annotate_p", action="store_true", default=False)
    args = ap.parse_args()

    def pairs_for_session():
        areas = [a for a in detect_areas(args.cache_dir, args.sid) if a in VALID]
        return [(a,b) for a in areas for b in areas if a!=b]

    bandC = (args.bandC_start, args.bandC_end)
    bandR = (args.bandR_start, args.bandR_end)

    if args.all_pairs:
        pairs = pairs_for_session()
        print(f"[info] sid={args.sid} running {len(pairs)} ordered pairs: {pairs}")
        for (A,B) in pairs:
            run_one_pair(args.sid, A, B, args.axes_dir, args.cache_dir,
                         args.win, args.step, args.k, args.ridge,
                         args.perms, args.n_jobs, args.seed,
                         bandC, bandR, args.int_win, args.evoked_sigma_ms,
                         out_tag=args.out_tag, skip_if_exists=args.skip_if_exists,
                         annotate_p=args.annotate_p,
                         no_x0=args.no_x0, cond_b_only=args.cond_b_only, pt_cov=args.pt_cov)
    else:
        if not args.A or not args.B:
            raise SystemExit("Provide --A and --B, or use --all_pairs.")
        run_one_pair(args.sid, args.A, args.B, args.axes_dir, args.cache_dir,
                     args.win, args.step, args.k, args.ridge,
                     args.perms, args.n_jobs, args.seed,
                     bandC, bandR, args.int_win, args.evoked_sigma_ms,
                     out_tag=args.out_tag, skip_if_exists=args.skip_if_exists,
                     annotate_p=args.annotate_p,
                     no_x0=args.no_x0, cond_b_only=args.cond_b_only, pt_cov=args.pt_cov)

if __name__ == "__main__":
    main()
