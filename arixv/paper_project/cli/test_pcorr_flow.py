#!/usr/bin/env python
"""
Flow analysis using PARTIAL CORRELATION.

Advantages:
- Null is centered at ZERO (much cleaner)
- No ΔLL awkwardness
- Easy to interpret: r > 0 means A helps predict B
- Bounded between -1 and 1

Method:
1. Residualize B(t) against B-lags: B_resid = B(t) - predicted
2. Residualize A-lags against B-lags: A_resid = A-lags - predicted  
3. partial_corr = correlation(B_resid, A_resid)

For multi-dimensional A (e.g., R has 4 dims), use canonical correlation or average.
"""

import numpy as np
import json
import os
import sys
from typing import Dict, Tuple, Optional
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------- helpers --------------------

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

def _build_lag_stack_multi(Y: np.ndarray, L: int) -> np.ndarray:
    N, B, K = Y.shape
    if L <= 0:
        return np.zeros((N, B, 0), dtype=float)
    Xlags = np.zeros((N, B, L * K), dtype=float)
    for l in range(1, L + 1):
        Xlags[:, l:, (l - 1) * K: l * K] = Y[:, :-l, :]
    return Xlags

def _circular_shift_time(Y: np.ndarray, rng: np.random.Generator, min_shift: int = 1) -> np.ndarray:
    N, B, K = Y.shape
    Y_shifted = np.empty_like(Y)
    for n in range(N):
        shift = rng.integers(min_shift, max(min_shift + 1, B))
        Y_shifted[n] = np.roll(Y[n], shift, axis=0)
    return Y_shifted

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


# -------------------- partial correlation --------------------

def residualize(Y: np.ndarray, X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """
    Residualize Y against X using ridge regression.
    Y: (N,) or (N, K)
    X: (N, P)
    Returns: Y_resid (same shape as Y)
    """
    if Y.ndim == 1:
        Y = Y[:, None]
        squeeze = True
    else:
        squeeze = False
    
    N, K = Y.shape
    P = X.shape[1]
    
    # Ridge regression
    XtX = X.T @ X + ridge * np.eye(P)
    XtY = X.T @ Y
    try:
        B = np.linalg.solve(XtX, XtY)
        Y_pred = X @ B
        Y_resid = Y - Y_pred
    except:
        Y_resid = Y  # fallback
    
    if squeeze:
        Y_resid = Y_resid[:, 0]
    
    return Y_resid


def partial_correlation(Y: np.ndarray, X_interest: np.ndarray, X_control: np.ndarray, 
                        ridge: float = 1e-3) -> float:
    """
    Compute partial correlation between Y and X_interest, controlling for X_control.
    
    Y: (N,) target
    X_interest: (N,) or (N, K_interest) - predictors of interest
    X_control: (N, K_control) - control predictors
    
    For multi-dimensional X_interest, returns the squared multiple partial correlation
    (proportion of Y variance explained by X_interest after controlling for X_control).
    
    Returns: partial correlation (scalar between -1 and 1 for 1D, 0 to 1 for multi-D)
    """
    N = Y.shape[0]
    
    if X_interest.ndim == 1:
        X_interest = X_interest[:, None]
    
    K_interest = X_interest.shape[1]
    
    # Residualize Y against control
    Y_resid = residualize(Y, X_control, ridge)
    
    # Residualize X_interest against control
    X_interest_resid = residualize(X_interest, X_control, ridge)
    
    if K_interest == 1:
        # Simple partial correlation
        r, _ = stats.pearsonr(Y_resid.flatten(), X_interest_resid.flatten())
        return r
    else:
        # Multiple partial correlation: R² from regressing Y_resid on X_interest_resid
        XtX = X_interest_resid.T @ X_interest_resid + ridge * np.eye(K_interest)
        XtY = X_interest_resid.T @ Y_resid
        try:
            B = np.linalg.solve(XtX, XtY)
            Y_pred = X_interest_resid @ B
            SS_res = np.sum((Y_resid - Y_pred) ** 2)
            SS_tot = np.sum(Y_resid ** 2)
            R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
            # Return signed sqrt(R²) based on first principal direction
            sign = np.sign(np.corrcoef(Y_resid.flatten(), Y_pred.flatten())[0, 1])
            return sign * np.sqrt(max(0, R2))
        except:
            return 0.0


def compute_flow_partial_corr(
    cacheA: Dict, cacheB: Dict,
    axesA: Dict, axesB: Dict,
    feature: str,
    align: str,
    orientation: Optional[str],
    pt_min_ms: Optional[float],
    lags_ms: float,
    ridge: float = 1e-3,
    perms: int = 500,
    induced: bool = True,
    include_B_lags: bool = True,
    seed: int = 0,
    null_method: str = "circular_shift",
) -> Dict:
    """
    Compute flow using partial correlation.
    
    Returns partial correlation between A-lags and B(t), controlling for B-lags.
    
    Null should be centered at ZERO.
    """
    
    # 1) Mask trials
    maskA = _trial_mask(cacheA, orientation, pt_min_ms)
    maskB = _trial_mask(cacheB, orientation, pt_min_ms)
    mask = maskA & maskB
    if not np.any(mask):
        raise ValueError("No trials remain.")
    
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
    
    # 4) Project
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
    
    # 6) Lag stacks
    XA_lags = _build_lag_stack_multi(YA, L)
    XB_lags = _build_lag_stack_multi(YB, L)
    
    N = YA.shape[0]
    ones = np.ones((N, 1), dtype=float)
    
    # Strata for shuffling
    perm_labels = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
    strata = perm_labels if perm_labels is not None else np.zeros(N)
    
    # 7) Compute observed partial correlations
    pcorr_AtoB = np.full(Bbins, np.nan)
    pcorr_BtoA = np.full(Bbins, np.nan)
    null_AtoB = np.full((perms, Bbins), np.nan)
    null_BtoA = np.full((perms, Bbins), np.nan)
    
    for b in range(start_b, Bbins):
        # --- A → B ---
        # Target: B at time t (first dimension if multi-D)
        Yt = YB[:, b, 0] if YB.shape[2] > 0 else YB[:, b, :].flatten()
        
        # Control: B-lags (+ intercept)
        X_control = ones.copy()
        if include_B_lags and XB_lags.shape[2] > 0:
            X_control = np.concatenate([X_control, XB_lags[:, b, :]], axis=1)
        
        # Interest: A-lags
        X_interest = XA_lags[:, b, :]
        
        if X_interest.shape[1] > 0:
            pcorr_AtoB[b] = partial_correlation(Yt, X_interest, X_control, ridge)
        
        # --- B → A ---
        Yt_rev = YA[:, b, 0] if YA.shape[2] > 0 else YA[:, b, :].flatten()
        
        X_control_r = ones.copy()
        if include_B_lags and XA_lags.shape[2] > 0:
            X_control_r = np.concatenate([X_control_r, XA_lags[:, b, :]], axis=1)
        
        X_interest_r = XB_lags[:, b, :]
        
        if X_interest_r.shape[1] > 0:
            pcorr_BtoA[b] = partial_correlation(Yt_rev, X_interest_r, X_control_r, ridge)
        
        # --- Null permutations ---
        for p in range(perms):
            if null_method == "circular_shift":
                YA_null = _circular_shift_time(YA, rng, min_shift=L+1)
                YB_null = _circular_shift_time(YB, rng, min_shift=L+1)
            else:  # trial_shuffle
                perm_idx_A = _permute_within_strata(strata, rng)
                perm_idx_B = _permute_within_strata(strata, rng)
                YA_null = YA[perm_idx_A]
                YB_null = YB[perm_idx_B]
            
            XA_lags_null = _build_lag_stack_multi(YA_null, L)
            XB_lags_null = _build_lag_stack_multi(YB_null, L)
            
            # A → B null
            X_interest_null = XA_lags_null[:, b, :]
            if X_interest_null.shape[1] > 0:
                null_AtoB[p, b] = partial_correlation(Yt, X_interest_null, X_control, ridge)
            
            # B → A null
            X_interest_null_r = XB_lags_null[:, b, :]
            if X_interest_null_r.shape[1] > 0:
                null_BtoA[p, b] = partial_correlation(Yt_rev, X_interest_null_r, X_control_r, ridge)
    
    # 8) Statistics
    null_mean_AtoB = np.nanmean(null_AtoB, axis=0)
    null_std_AtoB = np.nanstd(null_AtoB, axis=0, ddof=1)
    null_mean_BtoA = np.nanmean(null_BtoA, axis=0)
    null_std_BtoA = np.nanstd(null_BtoA, axis=0, ddof=1)
    
    # P-values (two-sided for correlation)
    p_AtoB = np.full(Bbins, np.nan)
    p_BtoA = np.full(Bbins, np.nan)
    
    for b in range(start_b, Bbins):
        if np.isfinite(pcorr_AtoB[b]):
            null_vals = null_AtoB[:, b]
            valid = np.isfinite(null_vals)
            if np.any(valid):
                # One-sided: how often is null >= observed
                count = np.sum(null_vals[valid] >= pcorr_AtoB[b])
                p_AtoB[b] = (1 + count) / (1 + np.sum(valid))
        
        if np.isfinite(pcorr_BtoA[b]):
            null_vals = null_BtoA[:, b]
            valid = np.isfinite(null_vals)
            if np.any(valid):
                count = np.sum(null_vals[valid] >= pcorr_BtoA[b])
                p_BtoA[b] = (1 + count) / (1 + np.sum(valid))
    
    return dict(
        time=time,
        pcorr_AtoB=pcorr_AtoB,
        pcorr_BtoA=pcorr_BtoA,
        null_mean_AtoB=null_mean_AtoB,
        null_std_AtoB=null_std_AtoB,
        null_mean_BtoA=null_mean_BtoA,
        null_std_BtoA=null_std_BtoA,
        null_all_AtoB=null_AtoB,
        null_all_BtoA=null_BtoA,
        p_AtoB=p_AtoB,
        p_BtoA=p_BtoA,
    )


# -------------------- main --------------------

def load_cache(out_root: str, align: str, sid: str, area: str):
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    return cache

def load_axes(out_root: str, align: str, sid: str, area: str, tag: str = None):
    if tag:
        path = os.path.join(out_root, align, sid, "axes", tag, f"axes_{area}.npz")
    else:
        path = os.path.join(out_root, align, sid, "axes", f"axes_{area}.npz")
    return np.load(path, allow_pickle=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--align", default="stim")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs=2, required=True)
    ap.add_argument("--feature", default="C")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--orientation", default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--lags_ms", type=float, default=50.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--null_method", default="circular_shift")
    args = ap.parse_args()
    
    areaA, areaB = args.areas
    
    print(f"Loading {args.sid}: {areaA} → {areaB}")
    cacheA = load_cache(args.out_root, args.align, args.sid, areaA)
    cacheB = load_cache(args.out_root, args.align, args.sid, areaB)
    axesA = load_axes(args.out_root, args.align, args.sid, areaA, args.tag)
    axesB = load_axes(args.out_root, args.align, args.sid, areaB, args.tag)
    
    print(f"Computing partial correlation flow ({args.perms} perms)...")
    
    results = compute_flow_partial_corr(
        cacheA, cacheB, axesA, axesB,
        feature=args.feature,
        align=args.align,
        orientation=args.orientation,
        pt_min_ms=args.pt_min_ms,
        lags_ms=args.lags_ms,
        ridge=args.ridge,
        perms=args.perms,
        null_method=args.null_method,
        seed=42,
    )
    
    time = results['time'] * 1000
    obs = results['pcorr_AtoB']
    null_mean = results['null_mean_AtoB']
    null_std = results['null_std_AtoB']
    null_all = results['null_all_AtoB']
    p_values = results['p_AtoB']
    
    t_max_idx = np.nanargmax(obs)
    t_max = time[t_max_idx]
    
    # Plot
    fig, axes_arr = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Observed with null band
    ax = axes_arr[0, 0]
    ax.fill_between(time, null_mean - 2*null_std, null_mean + 2*null_std, 
                    alpha=0.3, color='gray', label='Null ±2σ')
    ax.plot(time, null_mean, 'k--', lw=1, label='Null mean')
    ax.plot(time, obs, 'b-', lw=2, label='Observed')
    ax.axhline(0, color='r', ls=':', lw=1, label='Zero')
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('Partial Correlation')
    ax.set_xlabel('Time (ms)')
    ax.legend(fontsize=8)
    ax.set_title(f'Flow: {areaA} → {areaB} ({args.feature})')
    
    # Panel 2: Histogram at peak
    ax = axes_arr[0, 1]
    null_at_peak = null_all[:, t_max_idx]
    null_at_peak = null_at_peak[np.isfinite(null_at_peak)]
    
    ax.hist(null_at_peak, bins=30, alpha=0.7, color='gray', density=True)
    ax.axvline(obs[t_max_idx], color='b', lw=2, label=f'Obs: {obs[t_max_idx]:.3f}')
    ax.axvline(0, color='r', ls=':', lw=1, label='Zero')
    ax.axvline(np.mean(null_at_peak), color='k', ls='--', label=f'Null: {np.mean(null_at_peak):.3f}')
    ax.set_xlabel('Partial Correlation')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_title(f'Null Distribution at Peak (t={t_max:.0f} ms)')
    
    # Panel 3: Both directions
    ax = axes_arr[1, 0]
    ax.plot(time, results['pcorr_AtoB'], 'b-', lw=2, label=f'{areaA} → {areaB}')
    ax.plot(time, results['pcorr_BtoA'], 'r-', lw=2, label=f'{areaB} → {areaA}')
    ax.axhline(0, color='k', ls=':', lw=0.5)
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('Partial Correlation')
    ax.set_xlabel('Time (ms)')
    ax.legend()
    ax.set_title('Bidirectional Flow')
    
    # Panel 4: P-values
    ax = axes_arr[1, 1]
    ax.semilogy(time, p_values, 'b-', lw=1.5, label=f'{areaA}→{areaB}')
    ax.semilogy(time, results['p_BtoA'], 'r-', lw=1.5, label=f'{areaB}→{areaA}')
    ax.axhline(0.05, color='gray', ls='--', lw=1, label='p=0.05')
    ax.axhline(0.01, color='gray', ls=':', lw=1, label='p=0.01')
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('P-value')
    ax.set_xlabel('Time (ms)')
    ax.set_ylim([0.001, 1])
    ax.legend(fontsize=8)
    ax.set_title('P-values')
    
    plt.tight_layout()
    
    out_dir = os.path.join(args.out_root, args.align, args.sid, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pcorr_flow_{areaA}_{areaB}_{args.feature}.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Partial Correlation)")
    print("="*70)
    print(f"\nAt peak (t={t_max:.0f} ms):")
    print(f"  Observed:   {obs[t_max_idx]:.4f}")
    print(f"  Null mean:  {null_mean[t_max_idx]:.4f}")
    print(f"  Null std:   {null_std[t_max_idx]:.4f}")
    print(f"  P-value:    {p_values[t_max_idx]:.4f}")
    print(f"  Z-score:    {(obs[t_max_idx] - null_mean[t_max_idx]) / null_std[t_max_idx]:.2f}")
    
    overall_null = np.nanmean(null_mean)
    print(f"\nOverall null mean: {overall_null:.4f}")
    
    if abs(overall_null) < 0.02:
        print("  ✓ Null is near zero - good!")
    else:
        print(f"  ⚠ Null mean is {overall_null:.4f} - expected ~0")
    
    n_total = np.sum(np.isfinite(p_values))
    n_05 = np.sum(p_values < 0.05)
    n_01 = np.sum(p_values < 0.01)
    
    print(f"\nSignificant time points ({areaA}→{areaB}):")
    print(f"  p < 0.05: {n_05}/{n_total} ({100*n_05/n_total:.1f}%)")
    print(f"  p < 0.01: {n_01}/{n_total} ({100*n_01/n_total:.1f}%)")
    
    # Reverse direction
    p_rev = results['p_BtoA']
    n_05_r = np.sum(p_rev < 0.05)
    n_01_r = np.sum(p_rev < 0.01)
    print(f"\nSignificant time points ({areaB}→{areaA}):")
    print(f"  p < 0.05: {n_05_r}/{n_total} ({100*n_05_r/n_total:.1f}%)")
    print(f"  p < 0.01: {n_01_r}/{n_total} ({100*n_01_r/n_total:.1f}%)")


if __name__ == "__main__":
    main()
