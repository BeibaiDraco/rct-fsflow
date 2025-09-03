#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08i_fsflow_timesliding_integrated.py
Time-sliding, multivariate, feature-specific GC + integrated-band test (parallel permutations).

Adds:
  --band_start --band_end  (seconds): integrate GC over [band_start, band_end] and test
  --out_tag: write under results/session/<sid>/<tag>/
  --skip_if_exists: skip a pair if NPZ+PNG already exist in the tag

Usage examples:
  python 08i_fsflow_timesliding_integrated.py --sid 20200926 --A MLIP --B MFEF \
    --win 0.16 --step 0.02 --k 5 --perms 500 --n_jobs 16 --band_start 0.12 --band_end 0.28 \
    --out_tag win160_k5_perm500 --skip_if_exists

  python 08i_fsflow_timesliding_integrated.py --sid 20200926 --all_pairs \
    --win 0.16 --step 0.02 --k 5 --perms 500 --n_jobs 16 --band_start 0.12 --band_end 0.28 \
    --out_tag win160_k5_perm500 --skip_if_exists
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
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

# ---------- IO ----------
def load_axes(sid: int, area: str, axes_dir: Path, out_tag: str = ""):
    """
    Prefer tagged path results/session/<sid>/<tag>/axes_<AREA>.npz,
    then fall back to results/session/<sid>/axes_<AREA>.npz if tag not found.
    """
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

# ---------- Core: flow time-series with parallel nulls + integrated test ----------
def flow_timeseries_and_integrated(ZA: np.ndarray, ZB: np.ndarray,
                                   ZA_other: np.ndarray, ZB_other: np.ndarray,
                                   trials: pd.DataFrame, meta: dict,
                                   win_w: float, win_step: float,
                                   k: int, ridge: float,
                                   perms: int, n_jobs: int, seed: int,
                                   band_start: float, band_end: float):
    wins, centers = build_windows(meta, win_w, win_step)

    # true curves
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

    # permutations
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

    # integrated stats in the band
    mask = band_mask(centers, band_start, band_end)
    Ifwd = float(np.nansum(fwd[mask]))
    Irev = float(np.nansum(rev[mask]))
    Ifwd_null = np.nansum(fnull[:,mask], axis=1) if fnull is not None else np.array([])
    Irev_null = np.nansum(rnull[:,mask], axis=1) if rnull is not None else np.array([])

    # p-values (>= observed)
    pf = float((1 + np.sum(Ifwd_null >= Ifwd)) / (1 + len(Ifwd_null))) if Ifwd_null.size else np.nan
    pr = float((1 + np.sum(Irev_null >= Irev)) / (1 + len(Irev_null))) if Irev_null.size else np.nan

    return centers, fwd, rev, (f_lo,f_hi,r_lo,r_hi), Ifwd, Irev, Ifwd_null, Irev_null, pf, pr, mask

# ---------- Plot ----------
def plot_pair(sid, A, B, band, tC, C_fwd, C_rev, C_bands, I_C, pC,
              tR, R_fwd, R_rev, R_bands, I_R, pR, maskC, maskR, out_png: Path):
    C_flo, C_fhi, C_rlo, C_rhi = C_bands
    R_flo, R_fhi, R_rlo, R_rhi = R_bands
    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)

    # Category
    ax = axes[0]
    ax.axvspan(band[0], band[1], color="yellow", alpha=0.12, label="integration band")
    if C_flo is not None: ax.fill_between(tC, C_flo, C_fhi, color="grey", alpha=0.15, label="perm 95% (forward)")
    if C_rlo is not None: ax.fill_between(tC, C_rlo, C_rhi, facecolor="none", edgecolor="grey", hatch="///", alpha=0.18, label="perm 95% (reverse)")
    ax.plot(tC, C_fwd, lw=2, label=f"{A}→{B} (C)")
    ax.plot(tC, C_rev, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (C, rev)")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_ylabel("GC bits/bin (C)")
    ax.legend(frameon=False)
    ax.text(0.01, 0.95, f"Integrated C: {I_C:.4f}  p={pC:.3g}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    # Direction
    ax = axes[1]
    ax.axvspan(band[0], band[1], color="yellow", alpha=0.12, label="integration band")
    if R_flo is not None: ax.fill_between(tR, R_flo, R_fhi, color="grey", alpha=0.15, label="perm 95% (forward)")
    if R_rlo is not None: ax.fill_between(tR, R_rlo, R_rhi, facecolor="none", edgecolor="grey", hatch="///", alpha=0.18, label="perm 95% (reverse)")
    ax.plot(tR, R_fwd, lw=2, label=f"{A}→{B} (R)")
    ax.plot(tR, R_rev, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (R, rev)")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel("GC bits/bin (R)")
    ax.legend(frameon=False)
    ax.text(0.01, 0.95, f"Integrated R: {I_R:.4f}  p={pR:.3g}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    fig.suptitle(f"Feature-specific flow (with integrated test) — {sid}: {A}→{B}")
    fig.tight_layout(rect=[0,0.02,1,0.96]); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------- One pair ----------
def run_one_pair(sid: int, A: str, B: str,
                 axes_dir: Path, cache_dir: Path,
                 win: float, step: float, k: int, ridge: float,
                 perms: int, n_jobs: int, seed: int,
                 band_start: float, band_end: float,
                 out_tag: str = "", skip_if_exists: bool = False):
    # Load axes (tag-aware)
    ZC_A, ZR_A, metaA = load_axes(sid, A, axes_dir, out_tag)
    ZC_B, ZR_B, metaB = load_axes(sid, B, axes_dir, out_tag)
    if ZC_A.shape[1] != ZC_B.shape[1]: raise SystemExit("Mismatched bins A vs B.")
    trials = load_trials(cache_dir, sid, B)

    # C (condition on R)
    tC, C_fwd, C_rev, C_bands, IC_f, IC_r, IC_f_null, IC_r_null, pCf, pCr, maskC = \
        flow_timeseries_and_integrated(ZC_A, ZC_B, ZR_A, ZR_B, trials, metaB,
                                       win, step, k, ridge, perms, n_jobs, seed+11,
                                       band_start, band_end)
    # R (condition on C)
    if ZR_A.shape[2]==0 or ZR_B.shape[2]==0:
        tR = tC.copy(); R_fwd=R_rev=np.full_like(tR, np.nan, float)
        R_bands=(None,None,None,None)
        IR_f=IR_r=np.nan; IR_f_null=IR_r_null=np.array([]); pRf=pRr=np.nan; maskR = maskC
    else:
        tR, R_fwd, R_rev, R_bands, IR_f, IR_r, IR_f_null, IR_r_null, pRf, pRr, maskR = \
            flow_timeseries_and_integrated(ZR_A, ZR_B, ZC_A, ZC_B, trials, metaB,
                                           win, step, k, ridge, perms, n_jobs, seed+22,
                                           band_start, band_end)

    # Tagged output dir + skip-if-exists
    out_dir = axes_dir / f"{sid}"
    if out_tag:
        out_dir = out_dir / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"flow_timeseriesINT_{A}to{B}.npz"
    out_png = out_dir / f"flow_timeseriesINT_{A}to{B}.png"
    if skip_if_exists and out_npz.exists() and out_png.exists():
        print(f"[skip] {sid} {A}->{B} already exists in tag '{out_tag}'")
        return

    # Save + plot
    np.savez_compressed(
        out_npz,
        tC=tC, C_fwd=C_fwd, C_rev=C_rev, C_lo=C_bands[0], C_hi=C_bands[1], C_rlo=C_bands[2], C_rhi=C_bands[3],
        tR=tR, R_fwd=R_fwd, R_rev=R_rev, R_lo=R_bands[0], R_hi=R_bands[1], R_rlo=R_bands[2], R_rhi=R_bands[3],
        band=np.array([band_start, band_end]),
        IC_fwd=IC_f, IC_rev=IC_r, IC_fwd_null=IC_f_null, IC_rev_null=IC_r_null, pC_fwd=pCf, pC_rev=pCr,
        IR_fwd=IR_f, IR_rev=IR_r, IR_fwd_null=IR_f_null, IR_rev_null=IR_r_null, pR_fwd=pRf, pR_rev=pRr,
        meta=json.dumps({"sid": sid, "A": A, "B": B, "win": win, "step": step, "k": k,
                         "ridge": ridge, "perms": perms, "n_jobs": n_jobs, "tag": out_tag})
    )
    plot_pair(sid, A, B, (band_start, band_end),
              tC, C_fwd, C_rev, (C_bands[0],C_bands[1],C_bands[2],C_bands[3]), IC_f, pCf,
              tR, R_fwd, R_rev, (R_bands[0],R_bands[1],R_bands[2],R_bands[3]), IR_f, pRf,
              maskC, maskR, out_png)
    print(f"[done] {sid} {A}->{B} integrated C={IC_f:.4f} (p={pCf:.3g}), integrated R={IR_f:.4f} (p={pRf:.3g})  -> {out_npz}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--A", type=str, help="Source area")
    ap.add_argument("--B", type=str, help="Target area")
    ap.add_argument("--all_pairs", action="store_true")
    ap.add_argument("--axes_dir",  type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--win",   type=float, default=0.16)
    ap.add_argument("--step",  type=float, default=0.02)
    ap.add_argument("--k",     type=int,   default=5)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int,   default=500)
    ap.add_argument("--n_jobs", type=int,  default=8)
    ap.add_argument("--seed",  type=int,   default=123)
    ap.add_argument("--band_start", type=float, default=0.12)
    ap.add_argument("--band_end",   type=float, default=0.28)
    # NEW tagging / skipping
    ap.add_argument("--out_tag", type=str, default="", help="Write under results/session/<sid>/<tag>/")
    ap.add_argument("--skip_if_exists", action="store_true", default=False,
                    help="Skip pair if outputs already exist in the tag")
    args = ap.parse_args()

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
                         args.band_start, args.band_end,
                         out_tag=args.out_tag, skip_if_exists=args.skip_if_exists)
    else:
        if not args.A or not args.B:
            raise SystemExit("Provide --A and --B, or use --all_pairs.")
        run_one_pair(args.sid, args.A, args.B, args.axes_dir, args.cache_dir,
                     args.win, args.step, args.k, args.ridge,
                     args.perms, args.n_jobs, args.seed,
                     args.band_start, args.band_end,
                     out_tag=args.out_tag, skip_if_exists=args.skip_if_exists)

if __name__ == "__main__":
    main()
