# paper_project/paperflow/qc.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class QCAreaCurves:
    time: np.ndarray
    auc_C: Optional[np.ndarray]
    auc_S_raw: Optional[np.ndarray]
    auc_S_inv: Optional[np.ndarray]
    acc_R_macro: Optional[np.ndarray]
    lat_C_ms: Optional[float]
    lat_S_raw_ms: Optional[float]
    lat_S_inv_ms: Optional[float]
    meta: Dict
    dec_R_cv: Optional[np.ndarray] = None

def _first_k_above(y: np.ndarray, thr: float, k: int) -> int:
    consec = 0
    for i, v in enumerate(y):
        consec = consec + 1 if (np.isfinite(v) and v >= thr) else 0
        if consec >= k: return i - k + 1
    return -1

def _auc_binary(scores: np.ndarray, y_pm1: np.ndarray) -> float:
    if scores.ndim != 1: scores = scores.ravel()
    y = (y_pm1 > 0).astype(int)
    if np.unique(y).size < 2: return np.nan
    try: return float(roc_auc_score(y, scores))
    except Exception: return np.nan

def _macro_acc_multiclass(X: np.ndarray, y_int: np.ndarray) -> float:
    y = y_int.astype(int)
    classes = np.unique(y)
    if classes.size < 3: return np.nan
    try:
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X, y)
        pr = clf.predict(X)
        accs = []
        for c in classes:
            m = (y == c)
            if m.any(): accs.append(accuracy_score(y[m], pr[m]))
        return float(np.mean(accs)) if accs else np.nan
    except Exception:
        return np.nan

def _cv_multinomial_acc(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """Stratified CV accuracy (multinomial logistic) for 3-way direction."""
    y = y.astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 3 or counts.min() < 2:
        return np.nan
    k = max(2, min(n_splits, int(counts.min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=2000, class_weight="balanced"
        )
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else np.nan

def qc_curves_for_area(cache: Dict, axes: Dict, align: str,
                       time_s: np.ndarray, orientation: Optional[str],
                       compute_decoder_R: bool = False,
                       qc_r_residC: bool = False,
                       thr: float = 0.75, k_bins: int = 5) -> QCAreaCurves:
    Z = cache["Z"].astype(np.float64)
    C = cache.get("lab_C", np.full(Z.shape[0], np.nan)).astype(np.float64)
    R = cache.get("lab_R", np.full(Z.shape[0], np.nan)).astype(np.float64)
    S = cache.get("lab_S", np.full(Z.shape[0], np.nan)).astype(np.float64)
    OR = cache.get("lab_orientation", np.array(["pooled"]*Z.shape[0], dtype=object))
    IC = cache.get("lab_is_correct", np.ones(Z.shape[0], dtype=bool))
    time = time_s.copy()

    keep = IC.copy()
    if orientation is not None: keep &= (OR.astype(str) == orientation)
    Z, C, R, S = Z[keep], C[keep], R[keep], S[keep]
    N, B, U = Z.shape

    sC = axes["sC"] if ("sC" in axes and axes["sC"].size) else None
    sS_raw = axes["sS_raw"] if ("sS_raw" in axes and axes["sS_raw"].size) else None
    sS_inv = axes["sS_inv"] if ("sS_inv" in axes and axes["sS_inv"].size) else None
    sR = axes["sR"] if ("sR" in axes and axes["sR"].size and axes["sR"].ndim==2 and axes["sR"].shape[1]>=1) else None
    mu_R = axes.get("mu_R", None); std_R = axes.get("std_R", None)

    aucC = np.full(B, np.nan); aucSr = np.full(B, np.nan); aucSi = np.full(B, np.nan); accR = np.full(B, np.nan)
    decR = np.full(B, np.nan) if (align == "stim" and compute_decoder_R) else None

    for b in range(B):
        Xb = Z[:, b, :]

        if decR is not None:
            vals = []
            for cval in (-1.0, 1.0):
                mc = (np.sign(C) == (1 if cval>0 else -1))
                if not mc.any(): continue
                ydir = R[mc].copy()
                uniq = np.unique(ydir[~np.isnan(ydir)])
                if len(uniq) < 3: continue
                # map to 0..K-1
                mapv = {float(u): i for i,u in enumerate(sorted(uniq))}
                yi = np.array([mapv.get(float(v), np.nan) for v in ydir], dtype=float)
                mm = ~np.isnan(yi)
                Xi, yi = Xb[mc][mm], yi[mm].astype(int)
                if Xi.shape[0] < 30: continue
                vals.append(_cv_multinomial_acc(Xi, yi, n_splits=5))
            if vals:
                decR[b] = float(np.nanmean(vals))

        if sC is not None:
            scoresC = Xb @ sC
            aucC[b] = _auc_binary(scoresC, C)

        if sR is not None and align == "stim":
            XR = Xb.copy()
            # per-category center/whiten using training μ/σ
            if mu_R is not None and std_R is not None and XR.shape[1] == mu_R.shape[1]:
                XRcw = np.empty_like(XR)
                for idx, cval in enumerate((-1.0, 1.0)):
                    mc = (np.sign(C) == (1 if cval>0 else -1))
                    if mc.any():
                        XRcw[mc] = (XR[mc] - mu_R[idx]) / std_R[idx]
                XR = XRcw
            # residualize vs sC
            if qc_r_residC and sC is not None and sC.size:
                q = sC.reshape(-1,1) / (np.linalg.norm(sC) + 1e-12)
                XR = XR - (XR @ q) @ q.T
            ZR_b = XR @ sR
            # macro-ACC within category
            vals = []
            for cval in (-1.0, 1.0):
                mc = (np.sign(C) == (1 if cval>0 else -1))
                if mc.any():
                    ydir = R[mc].copy()
                    uniq = np.unique(ydir[~np.isnan(ydir)])
                    if len(uniq) >= 3:
                        mapv = {float(u): i for i,u in enumerate(sorted(uniq))}
                        yi = np.array([mapv.get(float(v), np.nan) for v in ydir], dtype=float)
                        mm = ~np.isnan(yi)
                        Xi, yi = ZR_b[mc][mm], yi[mm].astype(int)
                        if Xi.shape[0] >= 30:
                            vals.append(_macro_acc_multiclass(Xi, yi))
            if vals:
                accR[b] = float(np.nanmean(vals))

        if align == "sacc":
            if sS_raw is not None:
                scoresSr = Xb @ sS_raw
                aucSr[b] = _auc_binary(scoresSr, S)
            if sS_inv is not None:
                scoresSi = Xb @ sS_inv
                aucSi[b] = _auc_binary(scoresSi, S)

    latC_ms = None
    if np.isfinite(aucC).any():
        idx = _first_k_above(aucC, thr, k_bins); latC_ms = float(time[idx]*1000.0) if idx >= 0 else None
    latSr_ms = None
    if align == "sacc" and np.isfinite(aucSr).any():
        idx = _first_k_above(aucSr, thr, k_bins); latSr_ms = float(time[idx]*1000.0) if idx >= 0 else None
    latSi_ms = None
    if align == "sacc" and np.isfinite(aucSi).any():
        idx = _first_k_above(aucSi, thr, k_bins); latSi_ms = float(time[idx]*1000.0) if idx >= 0 else None

    meta = dict(n_trials=N, n_bins=B, n_units=U, align=align, orientation=orientation, thr=thr, k_bins=int(k_bins))
    return QCAreaCurves(time=time, auc_C=aucC if sC is not None else None,
                        auc_S_raw=aucSr if sS_raw is not None else None,
                        auc_S_inv=aucSi if sS_inv is not None else None,
                        acc_R_macro=accR if sR is not None else None,
                        lat_C_ms=latC_ms, lat_S_raw_ms=latSr_ms, lat_S_inv_ms=latSi_ms,
                        meta=meta,dec_R_cv=decR)

def save_qc(curves: QCAreaCurves, out_pdf: str, area: str):
    import matplotlib.pyplot as plt, os
    tms = curves.time * 1000.0
    plt.figure(figsize=(7.2,3.5))
    plt.axhline(0.5, ls="--", c="k", lw=0.8); plt.axvline(0, ls="--", c="k", lw=0.8)
    if curves.auc_C is not None:      plt.plot(tms, curves.auc_C,     lw=2, label="AUC(C | sC)")
    if curves.auc_S_inv is not None:  plt.plot(tms, curves.auc_S_inv, lw=2, label="AUC(S | sS inv)")
    if curves.auc_S_raw is not None:  plt.plot(tms, curves.auc_S_raw, lw=1.5, ls="--", label="AUC(S | sS raw)")
    if curves.acc_R_macro is not None:plt.plot(tms, curves.acc_R_macro, lw=2, label="ACC(R | sR) (macro, within C)")
    if getattr(curves, "dec_R_cv", None) is not None: plt.plot(tms, curves.dec_R_cv, lw=2, color="C4", alpha=0.9, label="CV-ACC(R) raw (5-fold)")
    plt.xlabel("Time (ms)"); plt.ylabel("AUC / ACC"); plt.title(f"{area} — QC curves"); plt.legend(loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_pdf); plt.savefig(os.path.splitext(out_pdf)[0] + ".png", dpi=300); plt.close()

def save_qc_json(curves: QCAreaCurves, out_json: str):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = dict(
        time=curves.time.tolist(),
        auc_C=(curves.auc_C.tolist() if curves.auc_C is not None else None),
        auc_S_raw=(curves.auc_S_raw.tolist() if curves.auc_S_raw is not None else None),
        auc_S_inv=(curves.auc_S_inv.tolist() if curves.auc_S_inv is not None else None),
        acc_R_macro=(curves.acc_R_macro.tolist() if curves.acc_R_macro is not None else None),
        dec_R_cv= (curves.dec_R_cv.tolist() if curves.dec_R_cv is not None else None),
        latencies_ms=dict(C=curves.lat_C_ms, S_raw=curves.lat_S_raw_ms, S_inv=curves.lat_S_inv_ms),
        meta=curves.meta
    )
    with open(out_json, "w") as f: json.dump(payload, f, indent=2)
