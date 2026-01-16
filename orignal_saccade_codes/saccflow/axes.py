# saccflow/axes.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

@dataclass
class AxisResult:
    vec: np.ndarray         # unit vector (units,)
    beta_raw: np.ndarray    # raw coefficients
    intercept: float
    auc_mean: float
    auc_std: float
    C_best: float
    n_used: int

@dataclass
class AxesPack:
    sC: Optional[AxisResult]
    sS: Optional[AxisResult]
    cos_sS_sC: Optional[float]

def _time_mask(time_s: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (time_s >= t0) & (time_s <= t1)

def _balanced_subset_by_other(y: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Downsample so each (other, y) cell contributes equally."""
    ok = ~np.isnan(y) & ~np.isnan(other)
    yv = np.sign(y[ok])  # ±1
    ov = np.sign(other[ok])
    idx = np.where(ok)[0]
    keep = []
    for o in [-1.0, 1.0]:
        m = (ov == o)
        if not np.any(m): 
            continue
        idx_o = idx[m]
        y_o = yv[m]
        n_pos = np.sum(y_o > 0)
        n_neg = np.sum(y_o < 0)
        n = min(n_pos, n_neg)
        if n == 0:
            continue
        # take first n of each class (shuffle outside if needed)
        pos_idx = idx_o[y_o > 0][:n]
        neg_idx = idx_o[y_o < 0][:n]
        keep.extend(pos_idx.tolist() + neg_idx.tolist())
    if not keep:
        return ok  # fallback: no change
    mk = np.zeros_like(ok, dtype=bool)
    mk[np.array(keep, dtype=int)] = True
    return mk

def _grid_logreg_auc(X: np.ndarray, y_pm1: np.ndarray, C_grid: List[float]) -> Tuple[LogisticRegression, float, float, float]:
    # Convert labels to {0,1}
    y = (y_pm1 > 0).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best = (None, -np.inf, 0.0, None)  # model, mean_auc, std, C
    for Cval in C_grid:
        aucs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(
                penalty="l2", C=Cval, solver="liblinear",
                class_weight="balanced", max_iter=2000
            )
            clf.fit(X[tr], y[tr])
            s = clf.decision_function(X[te])
            try:
                aucs.append(roc_auc_score(y[te], s))
            except ValueError:
                # single class in test fold
                continue
        if not aucs:
            continue
        m, sd = float(np.mean(aucs)), float(np.std(aucs))
        if m > best[1]:
            best = (None, m, sd, Cval)
    # refit with best C on all data
    if best[3] is None:
        # fall back to C=1.0
        best_C = 1.0
    else:
        best_C = best[3]
    clf = LogisticRegression(
        penalty="l2", C=best_C, solver="liblinear",
        class_weight="balanced", max_iter=2000
    )
    clf.fit(X, y)
    return clf, float(best[1]), float(best[2]), float(best_C)

def _train_axis(Z: np.ndarray, y: np.ndarray, time_s: np.ndarray,
                win: Tuple[float, float], balance_other: Optional[np.ndarray] = None,
                C_grid: Optional[List[float]] = None) -> Optional[AxisResult]:
    """
    Z: (trials, bins, units) z-scored counts
    y: (trials,) labels in ±1 (NaN allowed)
    time_s: (bins,)
    win: (t0, t1) seconds
    balance_other: (trials,) another ±1 label to balance within (e.g., balance S by C)
    """
    if C_grid is None:
        C_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
    mask_t = _time_mask(time_s, win[0], win[1])
    if not np.any(mask_t):
        return None
    Xw = Z[:, mask_t, :].mean(axis=1)  # (trials, units)
    ok = ~np.isnan(y)
    if balance_other is not None:
        bmask = _balanced_subset_by_other(y, balance_other)
        ok = ok & bmask
    X = Xw[ok]
    yv = y[ok]
    if X.shape[0] < 20 or np.unique(np.sign(yv[~np.isnan(yv)])).size < 2:
        return None
    clf, auc_m, auc_sd, C_best = _grid_logreg_auc(X, yv, C_grid)
    beta = clf.coef_.ravel().astype(np.float64)
    nrm = np.linalg.norm(beta)
    if nrm == 0 or not np.isfinite(nrm):
        return None
    vec = (beta / nrm).astype(np.float64)
    return AxisResult(vec=vec, beta_raw=beta, intercept=float(clf.intercept_[0]),
                      auc_mean=auc_m, auc_std=auc_sd, C_best=C_best, n_used=int(X.shape[0]))

def orthogonalize(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Remove component of u along v; return unit vector and cos(angle) before.
    """
    if u is None or v is None:
        return u, np.nan
    v = v / (np.linalg.norm(v) + 1e-12)
    cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    proj = np.dot(u, v) * v
    r = u - proj
    nr = np.linalg.norm(r)
    if nr < 1e-12:
        # if orthogonalization killed it, keep original direction
        return u, cosang
    return r / nr, cosang


def build_axes_for_area(cache: Dict,
                        win_C: Tuple[float, float],
                        win_S: Tuple[float, float],
                        correct_only: bool = True,
                        orientation: Optional[str] = None,   # 'vertical' | 'horizontal' | None
                        make_invariant: bool = True) -> Dict:
    """
    Train category axis (sC) and saccade axis (sS):
      - sC: logistic on C in window win_C
      - sS_raw: logistic on S in window win_S, balanced within Category
      - sS_inv: sS_raw orthogonalized to sC (category-invariant) if make_invariant=True
    Optionally restrict to trials of a single target configuration via `orientation`.
    """
    Z = cache["Z"]
    time = cache["time"]
    C = cache["C"]
    S = cache["S"]
    ori = cache.get("orientation", None)
    is_corr = cache.get("is_correct", None)

    keep = ~np.isnan(C) & ~np.isnan(S)
    if correct_only and (is_corr is not None):
        keep &= is_corr
    if orientation is not None and (ori is not None):
        keep &= (ori.astype(str) == orientation)

    Z = Z[keep]
    C = C[keep]
    S = S[keep]

    # sC
    resC = _train_axis(Z, C, time, win_C, balance_other=None)
    sC_vec = resC.vec if resC is not None else None

    # sS (RAW, balanced within category)
    resS_raw = _train_axis(Z, S, time, win_S, balance_other=C)
    sS_raw_vec = resS_raw.vec if resS_raw is not None else None

    # sS invariant (orthogonalize to sC)
    resS_inv = None
    cos_sSraw_sC = None
    if make_invariant and (sS_raw_vec is not None) and (sC_vec is not None):
        sS_orth, cosang = orthogonalize(sS_raw_vec, sC_vec)
        resS_inv = AxisResult(vec=sS_orth, beta_raw=resS_raw.beta_raw, intercept=resS_raw.intercept,
                              auc_mean=resS_raw.auc_mean, auc_std=resS_raw.auc_std,
                              C_best=resS_raw.C_best, n_used=resS_raw.n_used)
        cos_sSraw_sC = cosang

    return {
        "sC": resC,
        "sS_raw": resS_raw,
        "sS_inv": resS_inv,
        "cos_sSraw_sC": cos_sSraw_sC
    }

def auc_from_axis_projection(Z: np.ndarray, axis_vec: np.ndarray, y_pm1: np.ndarray,
                             bins_mask: np.ndarray) -> float:
    """
    Project Z along axis across masked bins, compute AUC for y.
    """
    if axis_vec is None:
        return np.nan
    Xw = Z[:, bins_mask, :].mean(axis=1)  # (trials, units)
    s = Xw @ axis_vec
    y = (y_pm1 > 0).astype(int)
    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, s))
