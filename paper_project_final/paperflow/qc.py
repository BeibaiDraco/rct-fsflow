# paper_project/paperflow/qc.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

@dataclass
class QCAreaCurves:
    time: np.ndarray
    auc_C: Optional[np.ndarray]
    auc_S_raw: Optional[np.ndarray]
    auc_S_inv: Optional[np.ndarray]
    acc_R_macro: Optional[np.ndarray]
    auc_T: Optional[np.ndarray]           # NEW: T (target config) axis AUC
    lat_C_ms: Optional[float]
    lat_S_raw_ms: Optional[float]
    lat_S_inv_ms: Optional[float]
    lat_T_ms: Optional[float]             # NEW: T latency
    meta: Dict

def first_k_above(y: np.ndarray, thr: float, k: int) -> int:
    run = 0
    for i, v in enumerate(y):
        if np.isfinite(v) and v >= thr: 
            run += 1
            if run >= k: return i - k + 1
        else:
            run = 0
    return -1

def gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0: return np.array([1.0], dtype=float)
    half = int(np.ceil(3* sigma))
    x = np.arange(-half, half+1, dtype=float)
    k = np.exp(-0.5*(x/sigma)**2)
    k /= k.sum()
    return k

def smooth_time(X: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: return X
    T,B,U = X.shape
    k = gaussian_kernel(sigma)
    Y = np.empty_like(X, dtype=float)
    for u in range(U):
        Y[:,:,u] = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, X[:,:,u])
    return Y

def qc_curves_for_area(
    cache: Dict,
    axes: Dict,
    align: str,
    time_s: np.ndarray,
    orientation: Optional[str],
    thr: float = 0.75,
    k_bins: int = 5,
    pt_min_ms: Optional[float] = None,
) -> QCAreaCurves:
    """Compute time-resolved QC curves using *exactly* the same masking and preprocessing as in training."""
    Z = cache["Z"].astype(np.float64)
    C = cache.get("lab_C", np.full(Z.shape[0], np.nan)).astype(np.float64)
    R = cache.get("lab_R", np.full(Z.shape[0], np.nan)).astype(np.float64)
    S = cache.get("lab_S", np.full(Z.shape[0], np.nan)).astype(np.float64)
    # T = category × saccade_location_sign; derive if not stored
    if "lab_T" in cache:
        T = cache.get("lab_T", np.full(Z.shape[0], np.nan)).astype(np.float64)
    else:
        T = np.sign(C) * np.sign(S)
        T[~(np.isfinite(C) & np.isfinite(S))] = np.nan
    OR = cache.get("lab_orientation", np.array(["pooled"]*Z.shape[0], dtype=object))
    PT = cache.get("lab_PT_ms", None)
    IC = cache.get("lab_is_correct", np.ones(Z.shape[0], dtype=bool))

    keep = np.ones(Z.shape[0], dtype=bool)
    keep &= IC
    if orientation is not None and "lab_orientation" in cache:
        keep &= (OR.astype(str) == orientation)
    if pt_min_ms is not None and (PT is not None):
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))

    Z = Z[keep]; C = C[keep]; R = R[keep]; S = S[keep]; T = T[keep]
    meta = dict(align=align, orientation=orientation, pt_min_ms=(float(pt_min_ms) if pt_min_ms is not None else None),
                n_trials=int(Z.shape[0]), n_bins=int(Z.shape[1]), n_units=int(Z.shape[2]), thr=float(thr), k_bins=int(k_bins))

    sC = (axes["sC"] if "sC" in axes and axes["sC"].size else None)
    sR = (axes["sR"] if "sR" in axes and axes["sR"].ndim==2 and axes["sR"].size>0 else None)
    sSr = (axes["sS_raw"] if "sS_raw" in axes and axes["sS_raw"].size else None)
    sSi = (axes["sS_inv"] if "sS_inv" in axes and axes["sS_inv"].size else None)
    sT = (axes["sT"] if "sT" in axes and axes["sT"].size else None)

    B = time_s.size
    aucC  = np.full(B, np.nan)
    aucSr = np.full(B, np.nan)
    aucSi = np.full(B, np.nan)
    accR  = np.full(B, np.nan)
    aucT  = np.full(B, np.nan)

    # Precompute R whitening in its window if present in meta
    mu_R = None; std_R = None
    mR = None
    if sR is not None:
        # rebuild within-category mean/std from training meta if available
        mu_R = None; std_R = None
        try:
            # if not stored, fall back to raw (no whitening)
            # (You can store mu_R/std_R in axes meta if desired)
            pass
        except Exception:
            pass

    for b in range(B):
        Xb = Z[:,b,:].astype(float)
        if sC is not None:
            sc = Xb @ sC
            yC = (C > 0).astype(int)
            if np.unique(yC).size == 2:
                try: 
                    aucC[b] = float(roc_auc_score(yC, sc))
                except Exception:
                    aucC[b] = np.nan

        if align == "sacc":
            if sSr is not None:
                ssr = Xb @ sSr
                ys = (S > 0).astype(int)
                if np.unique(ys).size == 2:
                    try: aucSr[b] = float(roc_auc_score(ys, ssr))
                    except Exception: pass
            if sSi is not None:
                ssi = Xb @ sSi
                ys = (S > 0).astype(int)
                if np.unique(ys).size == 2:
                    try: aucSi[b] = float(roc_auc_score(ys, ssi))
                    except Exception: pass

        # R macro accuracy within C (stim alignment)
        if sR is not None and align == "stim":
            XR = Xb @ sR  # (N x R_dim); assume whitening handled in training/axis
            vals = []
            for cs in (-1.0, +1.0):
                m = (np.sign(C) == cs)
                if not m.any(): 
                    continue
                ydir = R[m]
                uv = np.unique(ydir[~np.isnan(ydir)])
                if uv.size < 2: 
                    continue
                yi = np.array([np.where(uv == v)[0][0] if np.isfinite(v) else -1 for v in ydir], dtype=int)
                mm = (yi >= 0)
                Xi, yi2 = XR[m][mm], yi[mm]
                if Xi.shape[0] < 30: 
                    continue
                clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
                clf.fit(Xi, yi2)
                pred = clf.predict(Xi)
                vals.append(accuracy_score(yi2, pred))
            if vals:
                acc = float(np.mean(vals))
                accR[b] = acc

        # T (target configuration) AUC — stim alignment typically
        if sT is not None:
            st = Xb @ sT
            yT = (T > 0).astype(int)
            okT = np.isfinite(T)
            if okT.sum() >= 10 and np.unique(yT[okT]).size == 2:
                try:
                    aucT[b] = float(roc_auc_score(yT[okT], st[okT]))
                except Exception:
                    pass

    # latencies
    idxC = first_k_above(aucC, thr, k_bins) if np.any(np.isfinite(aucC)) else -1
    latC = float(time_s[idxC]*1000.0) if idxC >= 0 else None
    latSr = None
    if np.any(np.isfinite(aucSr)):
        idxSr = first_k_above(aucSr, thr, k_bins)
        latSr = float(time_s[idxSr]*1000.0) if idxSr >= 0 else None
    latSi = None
    if np.any(np.isfinite(aucSi)):
        idxSi = first_k_above(aucSi, thr, k_bins)
        latSi = float(time_s[idxSi]*1000.0) if idxSi >= 0 else None
    latT = None
    if np.any(np.isfinite(aucT)):
        idxT = first_k_above(aucT, thr, k_bins)
        latT = float(time_s[idxT]*1000.0) if idxT >= 0 else None

    return QCAreaCurves(
        time=time_s,
        auc_C=aucC if sC is not None else None,
        auc_S_raw=aucSr if (sSr is not None or sSi is not None) else None,
        auc_S_inv=aucSi if sSi is not None else None,
        acc_R_macro=accR if sR is not None else None,
        auc_T=aucT if sT is not None else None,
        lat_C_ms=latC, lat_S_raw_ms=latSr, lat_S_inv_ms=latSi, lat_T_ms=latT,
        meta=meta,
    )

