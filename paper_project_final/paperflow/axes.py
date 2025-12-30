# paper_project/paperflow/axes.py
from __future__ import annotations
import os, json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from paperflow.norm import get_Z, baseline_stats

# ---------- small helpers ----------

def window_mask(time_s: np.ndarray, win: Tuple[float,float]) -> np.ndarray:
    return (time_s >= win[0]) & (time_s <= win[1])

# 2) fix avg_over_window fallback
def avg_over_window(Z: np.ndarray, mask_t: np.ndarray) -> np.ndarray:
    # (trials, bins, units) -> (trials, units)
    if not np.any(mask_t):
        return np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
    return Z[:, mask_t, :].mean(axis=1).astype(np.float64)


def unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v); 
    if not np.isfinite(n) or n == 0: 
        return np.zeros_like(v)
    return v / n

def orthogonalize(u: np.ndarray, v: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[float]]:
    if v is None or u.size == 0: 
        return u, None
    v = unit_vec(v)
    # protect against degenerate v
    vn = np.linalg.norm(v)
    if not np.isfinite(vn) or vn == 0:
        return u, None
    dot = float(np.dot(u, v) / (np.linalg.norm(u)*vn + 1e-12))
    u2 = u - dot * v
    un = np.linalg.norm(u2)
    if not np.isfinite(un) or un == 0:
        return u, dot
    return u2 / un, dot

def auc_binary_scores(scores: np.ndarray, y_pm1: np.ndarray) -> float:
    """AUC for binary ±1 labels from 1D scores."""
    y = (y_pm1 > 0).astype(int)
    if np.unique(y).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return np.nan


# ---------- Unified binary classifier with CV ----------

def cv_fit_binary_linear(
    X: np.ndarray,
    y_pm1: np.ndarray,
    method: str,
    C_grid: List[float],
    sample_weight: Optional[np.ndarray] = None,
    lda_shrinkage: str = "auto",
) -> Tuple[np.ndarray, float, Any]:
    """
    Unified binary linear classifier with CV for logreg, svm, or lda.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y_pm1 : np.ndarray
        Labels ±1.
    method : str
        "logreg", "svm", or "lda".
    C_grid : list of float
        Regularization grid for logreg/svm.
    sample_weight : np.ndarray, optional
        Sample weights.
    lda_shrinkage : str
        "auto" or "none" for LDA.
    
    Returns
    -------
    w : np.ndarray
        Weight vector (n_features,).
    cv_score : float
        Best CV AUC score.
    best_param : any
        Best hyperparameter (C for logreg/svm, shrinkage for lda).
    """
    y = (y_pm1 > 0).astype(int)
    
    if np.unique(y).size < 2:
        return np.zeros(X.shape[1]), 0.5, None
    
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    if method == "logreg":
        best, bestC = -np.inf, None
        for Cval in C_grid:
            vals = []
            for tr, te in splitter.split(X, y):
                clf = LogisticRegression(
                    penalty="l2", C=Cval, solver="liblinear",
                    class_weight="balanced", max_iter=2000
                )
                try:
                    clf.fit(X[tr], y[tr], 
                           sample_weight=(None if sample_weight is None else sample_weight[tr]))
                    s = clf.decision_function(X[te])
                    vals.append(roc_auc_score(y[te], s))
                except Exception:
                    pass
            if vals:
                m = float(np.mean(vals))
                if m > best:
                    best, bestC = m, float(Cval)
        if bestC is None:
            bestC = 1.0
        clf = LogisticRegression(
            penalty="l2", C=bestC, solver="liblinear",
            class_weight="balanced", max_iter=2000
        )
        clf.fit(X, y, sample_weight=sample_weight)
        w = clf.coef_.ravel().astype(np.float64)
        return w, best if best > -np.inf else 0.5, bestC
    
    elif method == "svm":
        best, bestC = -np.inf, None
        n_samples, n_features = X.shape
        dual = (n_samples < n_features)
        for Cval in C_grid:
            vals = []
            for tr, te in splitter.split(X, y):
                clf = LinearSVC(
                    C=Cval, class_weight="balanced", max_iter=5000, dual=dual
                )
                try:
                    clf.fit(X[tr], y[tr])
                    s = clf.decision_function(X[te])
                    vals.append(roc_auc_score(y[te], s))
                except Exception:
                    pass
            if vals:
                m = float(np.mean(vals))
                if m > best:
                    best, bestC = m, float(Cval)
        if bestC is None:
            bestC = 1.0
        clf = LinearSVC(C=bestC, class_weight="balanced", max_iter=5000, dual=dual)
        clf.fit(X, y)
        w = clf.coef_.ravel().astype(np.float64)
        return w, best if best > -np.inf else 0.5, bestC
    
    elif method == "lda":
        shrink = lda_shrinkage if lda_shrinkage != "none" else None
        vals = []
        for tr, te in splitter.split(X, y):
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink)
            try:
                clf.fit(X[tr], y[tr])
                s = clf.decision_function(X[te])
                vals.append(roc_auc_score(y[te], s))
            except Exception:
                pass
        cv_score = float(np.mean(vals)) if vals else 0.5
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink)
        clf.fit(X, y)
        w = clf.coef_.ravel().astype(np.float64)
        return w, cv_score, shrink
    
    else:
        raise ValueError(f"Unknown method: {method!r}")


# Legacy wrapper for backward compatibility
def cv_logreg_binary(X: np.ndarray, y_pm1: np.ndarray, C_grid: List[float], sample_weight: Optional[np.ndarray]) -> tuple[LogisticRegression, float, float]:
    """Legacy wrapper - returns classifier object for backward compatibility."""
    y = (y_pm1 > 0).astype(int)
    best, bestC = -np.inf, None
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for Cval in C_grid:
        vals = []
        for tr, te in splitter.split(X, y):
            clf = LogisticRegression(penalty="l2", C=Cval, solver="liblinear", class_weight="balanced", max_iter=2000)
            clf.fit(X[tr], y[tr], sample_weight=(None if sample_weight is None else sample_weight[tr]))
            s = clf.decision_function(X[te])
            try: vals.append(roc_auc_score(y[te], s))
            except Exception: pass
        if vals:
            m = float(np.mean(vals))
            if m > best: best, bestC = m, float(Cval)
    if bestC is None: bestC = 1.0
    clf = LogisticRegression(penalty="l2", C=bestC, solver="liblinear", class_weight="balanced", max_iter=2000)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf, best, bestC

def multinomial_cv_acc(X: np.ndarray, y_int: np.ndarray, C_grid: List[float]) -> tuple[LogisticRegression, float, float]:
    y = y_int.astype(int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best, bestC = -np.inf, None
    for Cval in C_grid:
        vals = []
        for tr, te in splitter.split(X, y):
            clf = LogisticRegression(penalty="l2", C=Cval, solver="lbfgs", max_iter=2000, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pr = clf.predict(X[te])
            vals.append(accuracy_score(y[te], pr))
        m = float(np.mean(vals)) if vals else np.nan
        if m > best: best, bestC = m, float(Cval)
    if bestC is None: bestC = 1.0
    clf = LogisticRegression(penalty="l2", C=bestC, solver="lbfgs", max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf, best, bestC

def learn_category_invariant(Xmean: np.ndarray, C: np.ndarray, R: np.ndarray, C_reg: float, C_dim: int = 1) -> np.ndarray:
    """Hold-one-R-out stacking to suppress direction content in sC."""
    Rvals = np.unique(R[~np.isnan(R)])
    Ws = []
    for rv in Rvals:
        mtr = (~np.isnan(C)) & (R != rv)
        if mtr.sum() < 30 or np.unique(np.sign(C[mtr])).size < 2: 
            continue
        clf = LogisticRegression(penalty="l2", C=C_reg, solver="liblinear", class_weight="balanced", max_iter=2000)
        clf.fit(Xmean[mtr], (C[mtr] > 0).astype(int))
        Ws.append(clf.coef_.ravel())
    if not Ws:
        return np.zeros((Xmean.shape[1], 0))
    W = np.vstack(Ws)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)  # Vt: (k x U)
    Wc = Vt[:max(1, C_dim)].T
    # orthonormalize
    Q, _ = np.linalg.qr(Wc)
    return Q[:, :C_dim]

def select_topk_units_by_auc(Xmean: np.ndarray, y_pm1: np.ndarray, frac: float) -> np.ndarray:
    """Return boolean mask of top-k fraction units by |AUC-0.5| using univariate projections X[:,j]."""
    y = (y_pm1 > 0).astype(int)
    nU = Xmean.shape[1]
    if frac >= 0.999 or nU <= 1:
        return np.ones(nU, dtype=bool)
    scores = np.zeros(nU, dtype=float)
    for j in range(nU):
        try:
            scores[j] = abs(roc_auc_score(y, Xmean[:,j]))
        except Exception:
            scores[j] = 0.0
    k = max(1, int(np.ceil(frac * nU)))
    idx = np.argsort(scores)[-k:]
    mask = np.zeros(nU, dtype=bool); mask[idx] = True
    return mask


# ---------- Window search helpers ----------

def make_window_grid(
    search_range: Tuple[float, float],
    lens_ms: List[float],
    step_ms: float,
) -> List[Tuple[float, float]]:
    """
    Generate a grid of candidate windows for window search.
    
    Parameters
    ----------
    search_range : tuple of (float, float)
        Range to search within, in seconds.
    lens_ms : list of float
        Window lengths to try, in milliseconds.
    step_ms : float
        Step size for window starts, in milliseconds.
    
    Returns
    -------
    windows : list of (float, float)
        List of (t0, t1) tuples in seconds.
    """
    t_start, t_end = search_range
    step_s = step_ms / 1000.0
    windows = []
    
    for len_ms in lens_ms:
        len_s = len_ms / 1000.0
        t0 = t_start
        while t0 + len_s <= t_end + 1e-9:
            windows.append((t0, t0 + len_s))
            t0 += step_s
    
    return windows


def search_best_window_binary(
    Z: np.ndarray,
    y: np.ndarray,
    time_s: np.ndarray,
    candidates: List[Tuple[float, float]],
    method: str,
    C_grid: List[float],
    sample_weight: Optional[np.ndarray] = None,
    lda_shrinkage: str = "auto",
) -> Tuple[Tuple[float, float], float, Any, np.ndarray]:
    """
    Search for the best training window by CV score.
    
    Parameters
    ----------
    Z : np.ndarray
        Normalized data (trials, bins, units).
    y : np.ndarray
        Labels (±1).
    time_s : np.ndarray
        Time array in seconds.
    candidates : list of (float, float)
        Candidate windows.
    method : str
        Classifier method.
    C_grid : list of float
        Regularization grid.
    sample_weight : np.ndarray, optional
        Sample weights.
    lda_shrinkage : str
        LDA shrinkage setting.
    
    Returns
    -------
    best_win : tuple of (float, float)
        Best window.
    best_score : float
        Best CV score.
    best_param : any
        Best hyperparameter.
    best_w : np.ndarray
        Weight vector from best window.
    """
    best_win = None
    best_score = -np.inf
    best_param = None
    best_w = None
    
    ok = ~np.isnan(y)
    if ok.sum() < 20 or np.unique(np.sign(y[ok])).size < 2:
        # Not enough data
        if candidates:
            return candidates[0], 0.5, None, np.zeros(Z.shape[2])
        return (0.0, 0.1), 0.5, None, np.zeros(Z.shape[2])
    
    for win in candidates:
        mask_t = window_mask(time_s, win)
        if not np.any(mask_t):
            continue
        Xw = avg_over_window(Z[ok], mask_t)
        yw = y[ok]
        sw = sample_weight[ok] if sample_weight is not None else None
        
        w, score, param = cv_fit_binary_linear(Xw, yw, method, C_grid, sw, lda_shrinkage)
        
        if score > best_score:
            best_score = score
            best_win = win
            best_param = param
            best_w = w
    
    if best_win is None:
        best_win = candidates[0] if candidates else (0.0, 0.1)
        best_w = np.zeros(Z.shape[2])
    
    return best_win, best_score, best_param, best_w


# ---------- main trainer ----------

@dataclass
class AxisPack:
    sC: Optional[np.ndarray]
    sR: Optional[np.ndarray]         # (U, R_dim) or None
    sS_raw: Optional[np.ndarray]
    sS_inv: Optional[np.ndarray]
    sT: Optional[np.ndarray] = None  # target configuration axis (binary)
    meta: Dict = None
    sO: Optional[np.ndarray] = None  # context / orientation axis
    # Normalization parameters (stored when norm="baseline")
    norm_mu: Optional[np.ndarray] = None
    norm_sd: Optional[np.ndarray] = None

def train_axes_for_area(
    cache: Dict,
    feature_set: List[str],                # any of ["C","R","S","T","O"]
    time_s: np.ndarray,
    winC: Optional[Tuple[float,float]] = None,
    winR: Optional[Tuple[float,float]] = None,
    winS: Optional[Tuple[float,float]] = None,
    winT: Optional[Tuple[float,float]] = None,  # window for T axis (default: winC if None)
    orientation: Optional[str] = None,     # 'vertical'|'horizontal'|None
    C_dim: int = 1,
    R_dim: int = 2,
    S_dim: int = 1,                        # kept 1 by design (binary S)
    make_S_invariant: bool = True,
    C_grid: Optional[List[float]] = None,
    # selection knobs (for training only)
    select_mode: Optional[str] = None,     # None|'C'|'R'|'S'
    select_frac: float = 1.0,              # 0<..≤1.0
    # PT gating
    pt_min_ms: Optional[float] = None,     # if not None: filter to PT >= this (ms)
    # === NEW: normalization ===
    norm: str = "global",                   # "global"|"baseline"|"none"
    baseline_win: Optional[Tuple[float,float]] = None,  # required if norm="baseline"
    # === NEW: binary classifier selection ===
    clf_binary: str = "logreg",            # "logreg"|"svm"|"lda"
    lda_shrinkage: str = "auto",           # "auto"|"none"
    # === NEW: window search ===
    winC_candidates: Optional[List[Tuple[float,float]]] = None,
    winS_candidates: Optional[List[Tuple[float,float]]] = None,
    winT_candidates: Optional[List[Tuple[float,float]]] = None,
    winR_candidates: Optional[List[Tuple[float,float]]] = None,
) -> AxisPack:

    if C_grid is None:
        Cworth = [0.1, 0.3, 1.0, 3.0, 10.0]
    else:
        Cworth = list(C_grid)

    # Build trial mask FIRST
    N_total = cache["Z"].shape[0]
    C_raw = cache.get("lab_C", np.full(N_total, np.nan)).astype(np.float64)
    R_raw = cache.get("lab_R", np.full(N_total, np.nan)).astype(np.float64)
    S_raw = cache.get("lab_S", np.full(N_total, np.nan)).astype(np.float64)
    if "lab_T" in cache:
        Tcfg_raw = cache.get("lab_T", np.full(N_total, np.nan)).astype(np.float64)
    else:
        Tcfg_raw = np.sign(C_raw) * np.sign(S_raw)
        Tcfg_raw[~(np.isfinite(C_raw) & np.isfinite(S_raw))] = np.nan
    OR_raw = cache.get("lab_orientation", np.array(["pooled"] * N_total, dtype=object))
    PT_raw = cache.get("lab_PT_ms", None)
    IC_raw = cache.get("lab_is_correct", np.ones(N_total, dtype=bool))

    keep = np.ones(N_total, dtype=bool)
    keep &= IC_raw
    if orientation is not None and "lab_orientation" in cache:
        keep &= (OR_raw.astype(str) == orientation)
    if pt_min_ms is not None and (PT_raw is not None):
        keep &= np.isfinite(PT_raw) & (PT_raw >= float(pt_min_ms))

    # === Get normalized data ===
    norm_mu = None
    norm_sd = None
    
    if norm == "baseline":
        if baseline_win is None:
            raise ValueError("norm='baseline' requires baseline_win to be specified")
        # Compute baseline stats from raw X on masked trials
        if "X" not in cache:
            raise KeyError("Cache missing 'X' (raw counts). Cannot use norm='baseline'.")
        mu, sd = baseline_stats(cache["X"], time_s, baseline_win, keep)
        norm_mu = mu
        norm_sd = sd
        Z, norm_meta = get_Z(cache, time_s, keep, norm, baseline_win, 
                            axes_norm={"mu": mu, "sd": sd})
    else:
        Z, norm_meta = get_Z(cache, time_s, keep, norm, baseline_win, axes_norm=None)
    
    # Apply masks to labels
    C = C_raw[keep]
    R = R_raw[keep]
    S = S_raw[keep]
    Tcfg = Tcfg_raw[keep]
    OR = OR_raw[keep]
    
    # Use explicit winT if provided, else fall back to winC
    winT_used = winT if winT is not None else winC
    
    meta = dict(
        n_trials=int(Z.shape[0]), n_bins=int(Z.shape[1]), n_units=int(Z.shape[2]),
        orientation=orientation, pt_min_ms=(float(pt_min_ms) if pt_min_ms is not None else None),
        feature_set=feature_set, winC=winC, winR=winR, winS=winS, winT=winT_used,
        C_dim=int(C_dim), R_dim=int(R_dim), S_dim=int(S_dim),
        select_mode=(select_mode or "none"), select_frac=float(select_mode and select_frac or 1.0),
        sC_invariance=None,  # filled below if trained
        # === NEW: normalization metadata ===
        norm=norm,
        baseline_win=list(baseline_win) if baseline_win else None,
        norm_source="computed_from_training_trials" if norm == "baseline" else "cache",
        # === NEW: classifier metadata ===
        clf_binary=clf_binary,
        C_grid=Cworth,
        lda_shrinkage=lda_shrinkage if clf_binary == "lda" else None,
    )

    sC_vec = None
    sR_mat = None
    sS_raw_vec = None
    sS_inv = None
    sT_vec = None
    sO_vec = None

    # ---------- Train sC ----------
    if "C" in feature_set:
        # Determine window (search or fixed)
        if winC_candidates is not None and len(winC_candidates) > 0:
            # Window search
            yc = C.copy()
            best_win, best_score, best_param, wC = search_best_window_binary(
                Z, yc, time_s, winC_candidates, clf_binary, Cworth, None, lda_shrinkage
            )
            if wC is not None and np.linalg.norm(wC) > 0:
                sC_vec = unit_vec(wC)
                meta["winC_selected"] = list(best_win)
                meta["winC_cv_best"] = float(best_score)
                meta["winC_best_param"] = best_param
                meta["sC_auc_mean"] = float(best_score)
                meta["sC_C"] = best_param
                meta["sC_n"] = int((~np.isnan(yc)).sum())
                meta["sC_invariance"] = "none"
        elif winC is not None:
            mC = window_mask(time_s, winC)
            Xc_full = avg_over_window(Z, mC)  # (N x U)
            yc = C.copy()
            okc = ~np.isnan(yc)
            Xc = Xc_full[okc]; yC = yc[okc]
            # optional unit selection for C training
            Xc_used = Xc
            maskC = np.ones(Xc.shape[1], dtype=bool)
            if select_mode == "C" and select_frac < 0.999:
                maskC = select_topk_units_by_auc(Xc, yC, select_frac)
                Xc_used = Xc[:, maskC]
                meta["select_C_units"] = int(maskC.sum())
            if Xc_used.shape[0] >= 20 and np.unique(np.sign(yC)).size >= 2:
                cache_meta = cache.get("meta", {})
                if isinstance(cache_meta, str):
                    cache_meta = json.loads(cache_meta)
                elif hasattr(cache_meta, "item"):
                    cache_meta = json.loads(cache_meta.item())
                
                if cache_meta.get("align_event","stim") == "stim":
                    # direction-invariant via holdout R
                    if select_mode == "C":
                        Rc = R[okc]
                        Wc = learn_category_invariant(Xc_used, yC, Rc, C_reg=Cworth[2], C_dim=C_dim)
                        if Wc.shape[1] >= 1:
                            tmp = np.zeros((Xc.shape[1], Wc.shape[1]))
                            tmp[maskC,:] = Wc
                            sC_vec = unit_vec(tmp[:,0])
                            meta["sC_invariance"] = "holdoutR"
                            meta["sC_C"] = float(Cworth[2]); meta["sC_n"] = int(Xc_used.shape[0])
                    else:
                        # Use new unified classifier
                        wC, aucC, Cbest = cv_fit_binary_linear(Xc_used, yC, clf_binary, Cworth, None, lda_shrinkage)
                        if Xc_used.shape[1] != Xc.shape[1]:
                            tmp = np.zeros(Xc.shape[1], dtype=float)
                            tmp[maskC] = wC
                            wC = tmp
                        sC_vec = unit_vec(wC)
                        meta["sC_auc_mean"] = float(aucC); meta["sC_C"] = Cbest; meta["sC_n"] = int(Xc_used.shape[0]); meta["sC_invariance"] = "none"
                else:
                    # saccade-aligned C (for S invariance)
                    wC, aucC, Cbest = cv_fit_binary_linear(Xc_used, yC, clf_binary, Cworth, None, lda_shrinkage)
                    if select_mode == "C" and Xc_used.shape[1] != Xc.shape[1]:
                        tmp = np.zeros(Xc.shape[1], dtype=float); tmp[maskC] = wC; wC = tmp
                    sC_vec = unit_vec(wC)
                    meta["sC_auc_mean"] = float(aucC); meta["sC_C"] = Cbest; meta["sC_n"] = int(Xc.shape[0]); meta["sC_invariance"] = "sacc"
            # rectify sC orientation if trained
            if sC_vec is not None and Xc.shape[1] == sC_vec.shape[0]:
                sC_scores = (Xc @ sC_vec)
                aucC0 = auc_binary_scores(sC_scores, yC)
                if np.isfinite(aucC0) and aucC0 < 0.5:
                    sC_vec = -sC_vec
                    aucC0 = 1.0 - aucC0
                meta["sC_auc_proj"] = float(aucC0)

    # ---------- Train sR (stim only) ----------
    mu_R = None; std_R = None
    if "R" in feature_set:
        # Determine window
        winR_use = winR
        if winR_candidates is not None and len(winR_candidates) > 0:
            # For R, use multinomial CV accuracy to select window
            best_win_R = None
            best_acc_R = -np.inf
            for win in winR_candidates:
                mR = window_mask(time_s, win)
                if not np.any(mR):
                    continue
                Xr = avg_over_window(Z, mR)
                # Compute accuracy for this window
                accs = []
                for cval in (-1.0, +1.0):
                    m = (np.sign(C) == cval)
                    if not m.any():
                        continue
                    ydir = R[m]
                    cats_present = np.unique(ydir[~np.isnan(ydir)])
                    if cats_present.size < 2:
                        continue
                    yint = np.array([np.where(cats_present==v)[0][0] if np.isfinite(v) else -1 for v in ydir], dtype=int)
                    mm = (yint >= 0)
                    Xi, yi = Xr[m][mm], yint[mm]
                    if Xi.shape[0] < 30:
                        continue
                    _, acc, _ = multinomial_cv_acc(Xi, yi, Cworth)
                    accs.append(acc)
                if accs:
                    avg_acc = float(np.mean(accs))
                    if avg_acc > best_acc_R:
                        best_acc_R = avg_acc
                        best_win_R = win
            if best_win_R is not None:
                winR_use = best_win_R
                meta["winR_selected"] = list(best_win_R)
                meta["winR_cv_best"] = float(best_acc_R)
        
        if winR_use is not None:
            mR = window_mask(time_s, winR_use)
            Xr = avg_over_window(Z, mR)
            cats = np.array([-1.0, +1.0], dtype=float)
            mu_R = np.zeros((2, Xr.shape[1]), dtype=float)
            std_R = np.ones((2, Xr.shape[1]), dtype=float)
            for i,cval in enumerate(cats):
                m = (np.sign(C) == cval)
                if m.any():
                    mu_R[i]  = Xr[m].mean(axis=0)
                    std = Xr[m].std(axis=0, ddof=1)
                    std[std < 1e-8] = 1.0
                    std_R[i] = std
            Xcw = np.empty_like(Xr)
            for i,cval in enumerate(cats):
                m = (np.sign(C) == cval)
                if m.any():
                    Xcw[m] = (Xr[m] - mu_R[i]) / std_R[i]
            maskR = np.ones(Xcw.shape[1], dtype=bool)
            if select_mode == "R" and select_frac < 0.999:
                var = np.mean(np.abs(Xcw), axis=0)
                k = max(1, int(np.ceil(select_frac * Xcw.shape[1])))
                idx = np.argsort(var)[-k:]
                maskR[:] = False; maskR[idx] = True
                meta["select_R_units"] = int(maskR.sum())
            Xr_use = Xcw[:, maskR]
            if Xr_use.shape[0] >= 30 and Xr_use.shape[1] >= 1:
                blocks = []
                accs = []
                for cval in (-1.0, +1.0):
                    m = (np.sign(C) == cval)
                    if not m.any(): 
                        continue
                    ydir = R[m]
                    cats_present = np.unique(ydir[~np.isnan(ydir)])
                    if cats_present.size < 2: 
                        continue
                    yint = np.array([np.where(cats_present==v)[0][0] if np.isfinite(v) else -1 for v in ydir], dtype=int)
                    mm = (yint >= 0)
                    Xi, yi = Xr_use[m][mm], yint[mm]
                    if Xi.shape[0] < 30: 
                        continue
                    clfR, accm, Cbest = multinomial_cv_acc(Xi, yi, Cworth)
                    accs.append(accm)
                    blocks.append(clfR.coef_.astype(np.float64))
                if blocks:
                    W = np.vstack(blocks)
                    U_, S_, Vt = np.linalg.svd(W, full_matrices=False)
                    rd = max(1, int(R_dim))
                    sR_small = Vt[:rd].T
                    sR_full = np.zeros((Xr.shape[1], rd), dtype=float)
                    sR_full[maskR, :] = sR_small
                    if sC_vec is not None:
                        for j in range(rd):
                            sR_full[:,j], _ = orthogonalize(sR_full[:,j], sC_vec)
                            sR_full[:,j] = unit_vec(sR_full[:,j])
                    sR_mat = sR_full
                    meta["sR_dim"] = int(rd)
                    meta["sR_cv_acc_mean"] = float(np.nanmean(accs)) if accs else np.nan
                else:
                    sR_mat = None

    # ---------- Train sS (sacc only) ----------
    if "S" in feature_set:
        # Determine window
        if winS_candidates is not None and len(winS_candidates) > 0:
            # Window search for S
            ys = S.copy()
            ok = ~np.isnan(ys)
            yC_used = C.copy()
            ok2 = ok & ~np.isnan(yC_used)
            if ok2.sum() >= 20:
                # Build sample weights for balanced training
                yb = ys[ok2]
                w = np.ones_like(yb, dtype=float)
                for sign_s in (0,1):
                    for sign_c in (0,1):
                        m = (yb > 0) == bool(sign_s)
                        mc = (yC_used[ok2] > 0) == bool(sign_c)
                        mask_sw = m & mc
                        cnt = mask_sw.sum()
                        if cnt > 0:
                            w[mask_sw] = 1.0 / cnt
                
                # Create full sample weight array
                sample_weight_full = np.zeros(Z.shape[0])
                sample_weight_full[ok2] = w
                
                best_win, best_score, best_param, wS = search_best_window_binary(
                    Z[ok2], yb, time_s, winS_candidates, clf_binary, Cworth, w, lda_shrinkage
                )
                if wS is not None and np.linalg.norm(wS) > 0:
                    sS_raw_vec = unit_vec(wS)
                    sS_inv, cos = orthogonalize(sS_raw_vec, sC_vec) if make_S_invariant else (sS_raw_vec, None)
                    meta["winS_selected"] = list(best_win)
                    meta["winS_cv_best"] = float(best_score)
                    meta["winS_best_param"] = best_param
                    meta["sSraw_auc_mean"] = float(best_score)
                    meta["sS_C"] = best_param
                    meta["sS_n"] = int(ok2.sum())
                    meta["cos_sSraw_sC"] = (None if cos is None else float(cos))
        elif winS is not None:
            mS = window_mask(time_s, winS)
            Xs_full = avg_over_window(Z, mS)
            ys = S.copy()
            ok = ~np.isnan(ys)
            Xs = Xs_full[ok]; ys2 = ys[ok]
            maskS = np.ones(Xs.shape[1], dtype=bool)
            if select_mode == "S" and select_frac < 0.999:
                scores = np.array([abs(auc_binary_scores(Xs[:,j], ys2) - 0.5) for j in range(Xs.shape[1])])
                k = max(1, int(np.ceil(select_frac * Xs.shape[1])))
                idx = np.argsort(scores)[-k:]
                maskS[:] = False; maskS[idx] = True
                Xs = Xs[:, maskS]
                meta["select_S_units"] = int(maskS.sum())
            if Xs.shape[0] >= 20 and np.unique(np.sign(ys2)).size >= 2:
                yC_used = C[ok]
                ok2 = ~np.isnan(yC_used) & ~np.isnan(ys2)
                yb = ys2[ok2]; X_use = Xs[ok2]; w = np.ones_like(yb, dtype=float)
                for sign_s in (0,1):
                    for sign_c in (0,1):
                        m = (yb > 0) == bool(sign_s)
                        mc = (yC_used[ok2] > 0) == bool(sign_c)
                        mask_sw = m & mc
                        cnt = mask_sw.sum()
                        if cnt > 0:
                            w[mask_sw] = 1.0 / cnt
                wS, aucS, CbestS = cv_fit_binary_linear(X_use, yb, clf_binary, Cworth, w, lda_shrinkage)
                if not maskS.all():
                    wfull = np.zeros(Xs_full.shape[1], dtype=float); wfull[maskS] = wS; wS = wfull
                sS_raw_vec = unit_vec(wS)
                sS_inv, cos = orthogonalize(sS_raw_vec, sC_vec) if make_S_invariant else (sS_raw_vec, None)
                meta["sSraw_auc_mean"] = float(aucS); meta["sS_C"] = CbestS; meta["sS_n"] = int(Xs.shape[0])
                meta["cos_sSraw_sC"] = (None if cos is None else float(cos))

    # ---------- Train sT (target configuration) ----------
    winT_actual = winT if winT is not None else winC
    if "T" in feature_set:
        if winT_candidates is not None and len(winT_candidates) > 0:
            # Window search for T
            yt = Tcfg.copy()
            best_win, best_score, best_param, wT = search_best_window_binary(
                Z, yt, time_s, winT_candidates, clf_binary, Cworth, None, lda_shrinkage
            )
            if wT is not None and np.linalg.norm(wT) > 0:
                sT_vec = unit_vec(wT)
                # Orient to positive AUC
                okt = ~np.isnan(yt)
                if okt.any():
                    mask_t = window_mask(time_s, best_win)
                    Xt = avg_over_window(Z[okt], mask_t)
                    t_scores = Xt @ sT_vec
                    aucT0 = auc_binary_scores(t_scores, yt[okt])
                    if np.isfinite(aucT0) and aucT0 < 0.5:
                        sT_vec = -sT_vec
                        aucT0 = 1.0 - aucT0
                    meta["sT_auc_proj"] = float(aucT0) if np.isfinite(aucT0) else np.nan
                meta["winT_selected"] = list(best_win)
                meta["winT_cv_best"] = float(best_score)
                meta["winT_best_param"] = best_param
                meta["sT_auc_mean"] = float(best_score)
                meta["sT_C"] = best_param
                meta["sT_n"] = int((~np.isnan(yt)).sum())
        elif winT_actual is not None:
            mT = window_mask(time_s, winT_actual)
            Xt_full = avg_over_window(Z, mT)
            yt = Tcfg.copy()
            okt = ~np.isnan(yt)
            Xt = Xt_full[okt]; yT = yt[okt]
            if Xt.shape[0] >= 20 and np.unique(np.sign(yT)).size >= 2:
                wT, aucT, CbestT = cv_fit_binary_linear(Xt, yT, clf_binary, Cworth, None, lda_shrinkage)
                sT_vec = unit_vec(wT)
                t_scores = Xt @ sT_vec
                aucT0 = auc_binary_scores(t_scores, yT)
                if np.isfinite(aucT0) and aucT0 < 0.5:
                    sT_vec = -sT_vec
                    aucT0 = 1.0 - aucT0
                meta["sT_auc_mean"] = float(aucT)
                meta["sT_auc_proj"] = float(aucT0) if np.isfinite(aucT0) else np.nan
                meta["sT_C"] = CbestT
                meta["sT_n"] = int(Xt.shape[0])

    # ---------- Train sO (context / orientation axis) ----------
    if "O" in feature_set and winC is not None:
        mO = window_mask(time_s, winC)
        Xo_full = avg_over_window(Z, mO)

        OR_str = np.asarray(OR).astype(str)
        yO = np.full(Z.shape[0], np.nan, dtype=float)
        yO[OR_str == "vertical"] = +1.0
        yO[OR_str == "horizontal"] = -1.0

        ok = np.isfinite(yO)
        Xo = Xo_full[ok]
        yO2 = yO[ok]

        if Xo.shape[0] >= 20 and np.unique(yO2).size == 2:
            Ck = C[ok]
            w = np.ones_like(yO2, dtype=float)
            for c_sign in (-1.0, +1.0):
                for o_sign in (-1.0, +1.0):
                    m = (Ck == c_sign) & (yO2 == o_sign)
                    cnt = m.sum()
                    if cnt > 0:
                        w[m] = 1.0 / cnt

            wO, aucO, CbestO = cv_fit_binary_linear(Xo, yO2, clf_binary, Cworth, w, lda_shrinkage)

            if sC_vec is not None:
                wO, cos_sO_sC = orthogonalize(wO, sC_vec)
                meta["cos_sO_sC"] = float(cos_sO_sC) if cos_sO_sC is not None else None

            sO_vec = unit_vec(wO)
            meta["sO_auc_mean"] = float(aucO)
            meta["sO_C"] = CbestO
            meta["sO_n"] = int(Xo.shape[0])

    res = dict(meta)
    return AxisPack(
        sC=sC_vec, sR=sR_mat, sS_raw=sS_raw_vec, sS_inv=sS_inv, sT=sT_vec, 
        meta=res, sO=sO_vec,
        norm_mu=norm_mu, norm_sd=norm_sd
    )

def save_axes(out_dir: str, area: str, pack: AxisPack) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"axes_{area}.npz")
    np.savez_compressed(
        path,
        sC=(pack.sC if pack.sC is not None else np.array([])),
        sR=(pack.sR if pack.sR is not None else np.array([[]])),
        sS_raw=(pack.sS_raw if pack.sS_raw is not None else np.array([])),
        sS_inv=(pack.sS_inv if pack.sS_inv is not None else np.array([])),
        sT=(pack.sT if pack.sT is not None else np.array([])),
        sO=(pack.sO if pack.sO is not None else np.array([])),
        # === NEW: normalization parameters ===
        norm_mu=(pack.norm_mu if pack.norm_mu is not None else np.array([])),
        norm_sd=(pack.norm_sd if pack.norm_sd is not None else np.array([])),
        meta=json.dumps(pack.meta),
    )
    return path
