# paper_project/paperflow/flow_cv.py
"""
Cross-validated flow computation.

Key improvement: Uses held-out data to evaluate prediction, eliminating overfitting bias.
The null distribution should now be centered around ZERO, not ~4 bits.
"""

from __future__ import annotations
import json
import warnings
from typing import Dict, Tuple, Optional, List
import numpy as np

# -------------------- label helpers --------------------

def _label_vec(cache: Dict, key: str, mask: np.ndarray) -> Optional[np.ndarray]:
    v = cache.get(key, None)
    if v is None:
        return None
    return np.asarray(v).astype(float)[mask]

def _encode_joint_labels(cache: Dict, mask: np.ndarray, keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    vecs = []
    for k in keys:
        v = _label_vec(cache, k, mask)
        if v is None:
            return None
        vecs.append(np.asarray(np.round(v), dtype=int))
    base = 1000
    code = np.zeros_like(vecs[0], dtype=int)
    for i, v in enumerate(reversed(vecs)):
        code += (base**i) * v
    return code.astype(float)

# -------------------- trial masks & axes --------------------

def _trial_mask(cache: Dict, orientation: Optional[str], pt_min_ms: Optional[float]) -> np.ndarray:
    N = cache["Z"].shape[0]
    ok = np.ones(N, dtype=bool)
    if "lab_is_correct" in cache:
        ok &= cache["lab_is_correct"].astype(bool)
    if orientation is not None and "lab_orientation" in cache:
        ok &= (cache["lab_orientation"].astype(str) == orientation)
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        ok &= np.isfinite(PT) & (PT >= float(pt_min_ms))
    return ok

def _axis_matrix(axes_npz: Dict, feature: str) -> Optional[np.ndarray]:
    if feature == "C":
        a = axes_npz.get("sC", np.array([]))
    elif feature == "R":
        a = axes_npz.get("sR", np.array([[]]))
    elif feature == "S":
        a = axes_npz.get("sS_inv", np.array([]))
        if a.size == 0:
            a = axes_npz.get("sS_raw", np.array([]))
    else:
        a = np.array([])
    if a.size == 0:
        return None
    a = np.array(a)
    if a.ndim == 1:
        a = a[:, None]
    return a

def _project_multi(cache: Dict, A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Z = cache["Z"][mask].astype(np.float64)
    return np.tensordot(Z, A, axes=([2], [0]))

# -------------------- induced removal --------------------

def _induce_by_strata_multi(Y: np.ndarray, labels: Optional[np.ndarray]) -> np.ndarray:
    if labels is None:
        return Y
    out = Y.copy()
    lab = labels.copy()
    good = np.isfinite(lab)
    for u in np.unique(lab[good]):
        m = good & (lab == u)
        if np.any(m):
            out[m] -= out[m].mean(axis=0, keepdims=True)
    return out

# -------------------- lag building --------------------

def _build_lag_stack_multi(Y: np.ndarray, L: int) -> np.ndarray:
    N, B, K = Y.shape
    if L <= 0:
        return np.zeros((N, B, 0), dtype=float)
    Xlags = np.zeros((N, B, L * K), dtype=float)
    for l in range(1, L + 1):
        Xlags[:, l:, (l - 1) * K: l * K] = Y[:, :-l, :]
    return Xlags

# -------------------- cross-validated ridge regression --------------------

def _cv_ridge_sse(Y: np.ndarray, X: np.ndarray, ridge: float, n_folds: int = 5, 
                   rng: np.random.Generator = None) -> float:
    """
    Cross-validated sum of squared errors.
    
    Key: Train on (n_folds-1) folds, evaluate on held-out fold.
    This eliminates overfitting bias.
    
    Returns total SSE across all held-out samples.
    """
    N = Y.shape[0]
    if N < n_folds:
        n_folds = N
    
    # Create fold indices
    if rng is not None:
        indices = rng.permutation(N)
    else:
        indices = np.arange(N)
    
    fold_size = N // n_folds
    sse_total = 0.0
    n_test_total = 0
    
    for fold in range(n_folds):
        # Define test indices for this fold
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else N
        
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        if len(train_idx) < X.shape[1] + 1:
            continue  # Not enough training samples
        
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_te, Y_te = X[test_idx], Y[test_idx]
        
        # Fit ridge regression on training data
        XtX = X_tr.T @ X_tr
        p = XtX.shape[0]
        A = XtX + ridge * np.eye(p)
        XtY = X_tr.T @ Y_tr
        
        try:
            B = np.linalg.solve(A, XtY)
        except np.linalg.LinAlgError:
            continue
        
        # Evaluate on test data
        Y_pred = X_te @ B
        R = Y_te - Y_pred
        sse_total += np.sum(R * R)
        n_test_total += len(test_idx)
    
    return sse_total, n_test_total

def _ridge_sse_train(Y: np.ndarray, X: np.ndarray, ridge: float) -> float:
    """Training SSE (for reference, has overfitting bias)."""
    XtX = X.T @ X
    p = XtX.shape[0]
    A = XtX + ridge * np.eye(p)
    XtY = X.T @ Y
    try:
        B = np.linalg.solve(A, XtY)
        R = Y - X @ B
        return float(np.sum(R * R))
    except:
        return np.nan

# -------------------- permutation --------------------

def _permute_within_strata(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = labels.shape[0]
    perm = np.arange(N)
    good = np.isfinite(labels)
    if not np.any(good):
        return rng.permutation(N)
    for v in np.unique(labels[good]):
        idx = np.where(good & (labels == v))[0]
        if idx.size > 1:
            perm[idx] = rng.permutation(idx)
    return perm

def _circular_shift_time(Y: np.ndarray, rng: np.random.Generator, min_shift: int = 1) -> np.ndarray:
    N, B, K = Y.shape
    Y_shifted = np.empty_like(Y)
    for n in range(N):
        shift = rng.integers(min_shift, max(min_shift + 1, B))
        Y_shifted[n] = np.roll(Y[n], shift, axis=0)
    return Y_shifted

# -------------------- main function --------------------

def compute_flow_cv(
    cacheA: Dict, cacheB: Dict,
    axesA: Dict, axesB: Dict,
    feature: str,
    align: str,
    orientation: Optional[str],
    pt_min_ms: Optional[float],
    lags_ms: float,
    ridge: float,
    n_folds: int = 5,
    perms: int = 500,
    induced: bool = True,
    include_B_lags: bool = True,
    seed: int = 0,
    null_method: str = "circular_shift",
) -> Dict[str, np.ndarray]:
    """
    Cross-validated flow computation.
    
    Key difference from original:
    - Uses held-out data to evaluate prediction
    - Eliminates overfitting bias
    - Null distribution should be centered around ZERO
    
    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds (default 5)
    null_method : str
        'circular_shift' (recommended) or 'trial_shuffle'
    
    Returns
    -------
    Dict with:
        time : array of time points
        bits_AtoB : observed flow (cross-validated)
        bits_AtoB_train : training flow (for comparison, has overfitting)
        null_mean_AtoB, null_std_AtoB, p_AtoB : null statistics
        ... and same for BtoA direction
    """
    
    # 1) Mask trials
    maskA = _trial_mask(cacheA, orientation, pt_min_ms)
    maskB = _trial_mask(cacheB, orientation, pt_min_ms)
    mask = maskA & maskB
    if not np.any(mask):
        raise ValueError("No trials remain after masking.")
    
    # 2) Time grid
    time = cacheA["time"].astype(float)
    metaA = cacheA["meta"]
    if not isinstance(metaA, dict):
        metaA = json.loads(metaA.item() if hasattr(metaA, "item") else str(metaA))
    bin_s = float(metaA.get("bin_s", (time[1] - time[0] if time.size > 1 else 0.01)))
    
    Bbins = time.size
    L = int(max(1, round(lags_ms / (bin_s * 1000.0))))
    start_b = L
    rng = np.random.default_rng(seed)
    
    # 3) Axes
    Aaxis = _axis_matrix(axesA, feature)
    Baxis = _axis_matrix(axesB, feature)
    if Aaxis is None or Baxis is None:
        raise ValueError(f"Missing {feature} axis.")
    K_A = Aaxis.shape[1]
    K_B = Baxis.shape[1]
    
    # 4) Project onto axes
    YA_full = _project_multi(cacheA, Aaxis, mask)
    YB_full = _project_multi(cacheB, Baxis, mask)
    
    # 5) Induced removal
    labs_induce = None
    if induced:
        if feature == "C":
            labs_induce = _label_vec(cacheA, "lab_R", mask)
        elif feature == "R":
            labs_induce = _label_vec(cacheA, "lab_C", mask)
        elif feature == "S":
            labs_induce = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
    
    YA = _induce_by_strata_multi(YA_full, labs_induce) if induced else YA_full
    YB = _induce_by_strata_multi(YB_full, labs_induce) if induced else YB_full
    
    # 6) Build lag stacks
    XA_lags = _build_lag_stack_multi(YA, L)
    XB_lags = _build_lag_stack_multi(YB, L)
    
    # 7) Compute observed flow (cross-validated)
    N = YA.shape[0]
    ones = np.ones((N, 1), dtype=float)
    
    bits_AtoB_cv = np.full(Bbins, np.nan)
    bits_BtoA_cv = np.full(Bbins, np.nan)
    bits_AtoB_train = np.full(Bbins, np.nan)  # For comparison
    bits_BtoA_train = np.full(Bbins, np.nan)
    
    for b in range(start_b, Bbins):
        # A → B direction
        Yt = YB[:, b, :]
        
        X_red = ones.copy()
        if include_B_lags and XB_lags.shape[2] > 0:
            X_red = np.concatenate([X_red, XB_lags[:, b, :]], axis=1)
        X_full = np.concatenate([X_red, XA_lags[:, b, :]], axis=1)
        
        # Cross-validated SSE
        sse_red_cv, n_red = _cv_ridge_sse(Yt, X_red, ridge, n_folds, rng)
        sse_full_cv, n_full = _cv_ridge_sse(Yt, X_full, ridge, n_folds, rng)
        
        if sse_full_cv > 0 and sse_red_cv > 0 and n_red > 0:
            # Use n_test for proper scaling
            bits_AtoB_cv[b] = 0.5 * n_full * np.log2(sse_red_cv / sse_full_cv)
        
        # Training SSE (for comparison)
        sse_red_tr = _ridge_sse_train(Yt, X_red, ridge)
        sse_full_tr = _ridge_sse_train(Yt, X_full, ridge)
        if sse_full_tr > 0 and sse_red_tr > 0:
            bits_AtoB_train[b] = 0.5 * N * np.log2(sse_red_tr / sse_full_tr)
        
        # B → A direction
        Yt_rev = YA[:, b, :]
        
        X_red_r = ones.copy()
        if include_B_lags and XA_lags.shape[2] > 0:
            X_red_r = np.concatenate([X_red_r, XA_lags[:, b, :]], axis=1)
        X_full_r = np.concatenate([X_red_r, XB_lags[:, b, :]], axis=1)
        
        sse_red_cv_r, n_red_r = _cv_ridge_sse(Yt_rev, X_red_r, ridge, n_folds, rng)
        sse_full_cv_r, n_full_r = _cv_ridge_sse(Yt_rev, X_full_r, ridge, n_folds, rng)
        
        if sse_full_cv_r > 0 and sse_red_cv_r > 0 and n_red_r > 0:
            bits_BtoA_cv[b] = 0.5 * n_full_r * np.log2(sse_red_cv_r / sse_full_cv_r)
        
        sse_red_tr_r = _ridge_sse_train(Yt_rev, X_red_r, ridge)
        sse_full_tr_r = _ridge_sse_train(Yt_rev, X_full_r, ridge)
        if sse_full_tr_r > 0 and sse_red_tr_r > 0:
            bits_BtoA_train[b] = 0.5 * N * np.log2(sse_red_tr_r / sse_full_tr_r)
    
    # 8) Null distribution
    P = int(perms)
    null_AtoB = np.full((P, Bbins), np.nan)
    null_BtoA = np.full((P, Bbins), np.nan)
    
    # Strata for trial shuffle
    perm_labels = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
    if perm_labels is None:
        strata = np.zeros(N, dtype=float)
    else:
        strata = perm_labels
    
    for p in range(P):
        # Generate null data
        if null_method == "trial_shuffle":
            perm_idx_A = _permute_within_strata(strata, rng)
            perm_idx_B = _permute_within_strata(strata, rng)
            YA_null = YA[perm_idx_A]
            YB_null = YB[perm_idx_B]
        elif null_method == "circular_shift":
            YA_null = _circular_shift_time(YA, rng, min_shift=L + 1)
            YB_null = _circular_shift_time(YB, rng, min_shift=L + 1)
        else:
            raise ValueError(f"Unknown null_method: {null_method}")
        
        XA_lags_null = _build_lag_stack_multi(YA_null, L)
        XB_lags_null = _build_lag_stack_multi(YB_null, L)
        
        for b in range(start_b, Bbins):
            # A → B null
            Yt = YB[:, b, :]  # Target is NOT permuted
            X_red = ones.copy()
            if include_B_lags and XB_lags.shape[2] > 0:
                X_red = np.concatenate([X_red, XB_lags[:, b, :]], axis=1)
            X_full = np.concatenate([X_red, XA_lags_null[:, b, :]], axis=1)
            
            sse_red_cv, n_red = _cv_ridge_sse(Yt, X_red, ridge, n_folds, rng)
            sse_full_cv, n_full = _cv_ridge_sse(Yt, X_full, ridge, n_folds, rng)
            
            if sse_full_cv > 0 and sse_red_cv > 0 and n_full > 0:
                null_AtoB[p, b] = 0.5 * n_full * np.log2(sse_red_cv / sse_full_cv)
            
            # B → A null
            Yt_rev = YA[:, b, :]
            X_red_r = ones.copy()
            if include_B_lags and XA_lags.shape[2] > 0:
                X_red_r = np.concatenate([X_red_r, XA_lags[:, b, :]], axis=1)
            X_full_r = np.concatenate([X_red_r, XB_lags_null[:, b, :]], axis=1)
            
            sse_red_cv_r, n_red_r = _cv_ridge_sse(Yt_rev, X_red_r, ridge, n_folds, rng)
            sse_full_cv_r, n_full_r = _cv_ridge_sse(Yt_rev, X_full_r, ridge, n_folds, rng)
            
            if sse_full_cv_r > 0 and sse_red_cv_r > 0 and n_full_r > 0:
                null_BtoA[p, b] = 0.5 * n_full_r * np.log2(sse_red_cv_r / sse_full_cv_r)
    
    # 9) Compute null statistics
    null_mean_AtoB = np.nanmean(null_AtoB, axis=0)
    null_std_AtoB = np.nanstd(null_AtoB, axis=0, ddof=1)
    null_mean_BtoA = np.nanmean(null_BtoA, axis=0)
    null_std_BtoA = np.nanstd(null_BtoA, axis=0, ddof=1)
    
    # P-values (one-sided: observed > null)
    p_AtoB = np.full(Bbins, np.nan)
    p_BtoA = np.full(Bbins, np.nan)
    
    for b in range(start_b, Bbins):
        if np.isfinite(bits_AtoB_cv[b]):
            null_vals = null_AtoB[:, b]
            valid = np.isfinite(null_vals)
            if np.any(valid):
                count = np.sum(null_vals[valid] >= bits_AtoB_cv[b])
                p_AtoB[b] = (1 + count) / (1 + np.sum(valid))
        
        if np.isfinite(bits_BtoA_cv[b]):
            null_vals = null_BtoA[:, b]
            valid = np.isfinite(null_vals)
            if np.any(valid):
                count = np.sum(null_vals[valid] >= bits_BtoA_cv[b])
                p_BtoA[b] = (1 + count) / (1 + np.sum(valid))
    
    return dict(
        time=time,
        # Cross-validated (unbiased)
        bits_AtoB=bits_AtoB_cv,
        bits_BtoA=bits_BtoA_cv,
        # Training (biased, for comparison)
        bits_AtoB_train=bits_AtoB_train,
        bits_BtoA_train=bits_BtoA_train,
        # Null statistics
        null_mean_AtoB=null_mean_AtoB,
        null_std_AtoB=null_std_AtoB,
        null_mean_BtoA=null_mean_BtoA,
        null_std_BtoA=null_std_BtoA,
        p_AtoB=p_AtoB,
        p_BtoA=p_BtoA,
        # Full null distribution
        null_all_AtoB=null_AtoB,
        null_all_BtoA=null_BtoA,
        # Metadata
        meta=dict(
            feature=feature, align=align, orientation=orientation,
            pt_min_ms=pt_min_ms, lags_ms=lags_ms, ridge=ridge,
            n_folds=n_folds, perms=perms, null_method=null_method,
            induced=induced, include_B_lags=include_B_lags,
            N=int(mask.sum()), K_A=int(K_A), K_B=int(K_B),
        )
    )
