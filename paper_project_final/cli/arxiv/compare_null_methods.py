#!/usr/bin/env python
"""
Compare null methods (trial_shuffle vs circular_shift) with full distribution analysis.
This saves all null samples so we can examine the tail behavior.
"""

import numpy as np
import json
import os
import sys
import warnings
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============== Core functions (copied from flow.py with modifications) ==============

def _label_vec(cache: Dict, key: str, mask: np.ndarray) -> Optional[np.ndarray]:
    v = cache.get(key, None)
    if v is None:
        return None
    v = np.asarray(v).astype(float)
    return v[mask]

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
    elif feature == "T":
        a = axes_npz.get("sT", np.array([]))
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
    if A is None or A.shape[0] != Z.shape[2]:
        raise ValueError("Axis size mismatch.")
    return np.tensordot(Z, A, axes=([2],[0]))

def _induce_by_strata_multi(Y: np.ndarray, labels: Optional[np.ndarray]) -> np.ndarray:
    if labels is None:
        return Y
    out = Y.copy()
    lab = labels.copy()
    good = np.isfinite(lab)
    uniq = np.unique(lab[good])
    for u in uniq:
        m = good & (lab == u)
        if not np.any(m): 
            continue
        mu = out[m].mean(axis=0, keepdims=True)
        out[m] -= mu
    return out

def _build_lag_stack_multi(Y: np.ndarray, L: int) -> np.ndarray:
    N, B, K = Y.shape
    if L <= 0:
        return np.zeros((N, B, 0), dtype=float)
    Xlags = np.zeros((N, B, L*K), dtype=float)
    for l in range(1, L+1):
        Xlags[:, l:, (l-1)*K : l*K] = Y[:, :-l, :]
    return Xlags

def _ridge_sse_multi(Y: np.ndarray, X: np.ndarray, lam: float) -> float:
    XtX = X.T @ X
    p = XtX.shape[0]
    A = XtX + lam * np.eye(p)
    try:
        XtY = X.T @ Y
        B = np.linalg.solve(A, XtY)
        R = Y - X @ B
        return float(np.sum(R*R))
    except:
        return np.nan

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
        shift = rng.integers(min_shift, B)
        Y_shifted[n] = np.roll(Y[n], shift, axis=0)
    return Y_shifted


def compute_flow_with_full_null(
    cacheA: Dict, cacheB: Dict,
    axesA: Dict, axesB: Dict,
    feature: str,
    orientation: Optional[str],
    pt_min_ms: Optional[float],
    lags_ms: float,
    ridge: float,
    perms: int = 500,
    induced: bool = True,
    include_B_lags: bool = True,
    seed: int = 0,
    null_method: str = "trial_shuffle",
) -> Dict[str, np.ndarray]:
    """
    Compute flow and return FULL null distribution (all permutation samples).
    """
    
    # 1) mask trials
    maskA = _trial_mask(cacheA, orientation, pt_min_ms)
    maskB = _trial_mask(cacheB, orientation, pt_min_ms)
    mask = maskA & maskB
    if not np.any(mask):
        raise ValueError("No trials remain after masking.")

    # 2) time / binning
    time = cacheA["time"].astype(float)
    metaA = cacheA["meta"]
    if not isinstance(metaA, dict):
        metaA = json.loads(metaA.item() if hasattr(metaA, "item") else str(metaA))
    bin_s = float(metaA.get("bin_s", (time[1]-time[0] if time.size>1 else 0.01)))

    Bbins = time.size
    L = int(max(1, round(lags_ms / (bin_s*1000.0))))
    start_b = L
    rng = np.random.default_rng(seed)

    # 3) axes
    Aaxis = _axis_matrix(axesA, feature)
    Baxis = _axis_matrix(axesB, feature)
    if Aaxis is None or Baxis is None:
        raise ValueError(f"Missing {feature} axis.")

    # 4) projections
    YA_full = _project_multi(cacheA, Aaxis, mask)
    YB_full = _project_multi(cacheB, Baxis, mask)

    # 5) induced removal
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

    # 6) lag stacks
    XA_lags = _build_lag_stack_multi(YA, L)
    XB_lags = _build_lag_stack_multi(YB, L)

    # 7) Observed ΔLL
    N = YA.shape[0]
    ones = np.ones((N, 1), dtype=float)
    bits_AtoB = np.full(Bbins, np.nan)

    for b in range(start_b, Bbins):
        Yt = YB[:, b, :]
        X_red = ones
        if include_B_lags and XB_lags.shape[2] > 0:
            X_red = np.concatenate([X_red, XB_lags[:, b, :]], axis=1)
        sse_red = _ridge_sse_multi(Yt, X_red, ridge)
        X_full = np.concatenate([X_red, XA_lags[:, b, :]], axis=1)
        sse_full = _ridge_sse_multi(Yt, X_full, ridge)
        if sse_full > 0 and sse_red > 0:
            bits_AtoB[b] = 0.5 * N * np.log2(sse_red / sse_full)

    # 8) Null distribution (save ALL samples)
    P = int(perms)
    null_all = np.full((P, Bbins), np.nan)  # Full distribution!
    
    # Strata for trial_shuffle
    perm_labels = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
    if perm_labels is None:
        strata = np.zeros(N, dtype=float)
    else:
        strata = perm_labels

    for p in range(P):
        if null_method == "trial_shuffle":
            perm_idx = _permute_within_strata(strata, rng)
            YA_null = YA[perm_idx]
        elif null_method == "circular_shift":
            YA_null = _circular_shift_time(YA, rng, min_shift=L+1)
        else:
            raise ValueError(f"Unknown null_method: {null_method}")

        XA_lags_null = _build_lag_stack_multi(YA_null, L)

        for b in range(start_b, Bbins):
            Yt = YB[:, b, :]
            X_red = ones
            if include_B_lags and XB_lags.shape[2] > 0:
                X_red = np.concatenate([X_red, XB_lags[:, b, :]], axis=1)
            sse_red = _ridge_sse_multi(Yt, X_red, ridge)
            X_full = np.concatenate([X_red, XA_lags_null[:, b, :]], axis=1)
            sse_full = _ridge_sse_multi(Yt, X_full, ridge)
            if sse_full > 0 and sse_red > 0:
                null_all[p, b] = 0.5 * N * np.log2(sse_red / sse_full)

    # Compute summary stats
    null_mean = np.nanmean(null_all, axis=0)
    null_std = np.nanstd(null_all, axis=0, ddof=1)
    
    # P-values
    p_values = np.full(Bbins, np.nan)
    for b in range(start_b, Bbins):
        if np.isfinite(bits_AtoB[b]):
            null_vals = null_all[:, b]
            valid = np.isfinite(null_vals)
            if np.any(valid):
                count = np.sum(null_vals[valid] >= bits_AtoB[b])
                p_values[b] = (1 + count) / (1 + np.sum(valid))

    return dict(
        time=time,
        bits_AtoB=bits_AtoB,
        null_all=null_all,  # Full distribution!
        null_mean=null_mean,
        null_std=null_std,
        p_values=p_values,
        null_method=null_method,
    )


def load_cache(out_root: str, align: str, sid: str, area: str) -> Dict:
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    return cache

def load_axes(out_root: str, align: str, sid: str, area: str, tag: str = None) -> Dict:
    if tag:
        path = os.path.join(out_root, align, sid, "axes", tag, f"axes_{area}.npz")
    else:
        path = os.path.join(out_root, align, sid, "axes", f"axes_{area}.npz")
    return np.load(path, allow_pickle=True)


def plot_null_comparison(results_shuffle, results_shift, out_path: str):
    """Plot detailed comparison of null distributions."""
    
    time = results_shuffle['time'] * 1000
    obs = results_shuffle['bits_AtoB']
    
    # Find time of max observed flow
    t_max_idx = np.nanargmax(obs)
    t_max = time[t_max_idx]
    
    fig = plt.figure(figsize=(14, 10))
    
    # Panel 1: Time course comparison
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time, obs, 'k-', lw=2, label='Observed')
    
    for res, color, name in [(results_shuffle, 'C0', 'trial_shuffle'), 
                              (results_shift, 'C1', 'circular_shift')]:
        mu = res['null_mean']
        std = res['null_std']
        pctl_95 = np.nanpercentile(res['null_all'], 95, axis=0)
        pctl_99 = np.nanpercentile(res['null_all'], 99, axis=0)
        
        ax1.fill_between(time, mu - std, mu + std, alpha=0.2, color=color)
        ax1.plot(time, mu, '--', color=color, label=f'{name} mean')
        ax1.plot(time, pctl_95, ':', color=color, lw=1, label=f'{name} 95th')
    
    ax1.axvline(0, color='k', ls=':', lw=0.5)
    ax1.axvline(t_max, color='red', ls='--', lw=1, alpha=0.5)
    ax1.set_ylabel('ΔLL (bits)')
    ax1.set_xlabel('Time (ms)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_title('Flow Time Course with Null Distributions')
    
    # Panel 2: Histogram at peak time
    ax2 = fig.add_subplot(2, 2, 2)
    
    null_shuffle = results_shuffle['null_all'][:, t_max_idx]
    null_shift = results_shift['null_all'][:, t_max_idx]
    obs_val = obs[t_max_idx]
    
    bins = np.linspace(0, max(np.nanmax(null_shuffle), np.nanmax(null_shift), obs_val) * 1.1, 40)
    
    ax2.hist(null_shuffle[np.isfinite(null_shuffle)], bins=bins, alpha=0.5, 
             color='C0', label='trial_shuffle', density=True)
    ax2.hist(null_shift[np.isfinite(null_shift)], bins=bins, alpha=0.5, 
             color='C1', label='circular_shift', density=True)
    ax2.axvline(obs_val, color='k', lw=2, label=f'Observed ({obs_val:.1f})')
    
    ax2.set_xlabel('ΔLL (bits)')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=8)
    ax2.set_title(f'Null Distributions at Peak (t={t_max:.0f} ms)')
    
    # Panel 3: P-values comparison
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.semilogy(time, results_shuffle['p_values'], 'C0-', label='trial_shuffle')
    ax3.semilogy(time, results_shift['p_values'], 'C1-', label='circular_shift')
    ax3.axhline(0.05, color='r', ls='--', lw=1, label='p=0.05')
    ax3.axhline(0.01, color='r', ls=':', lw=1, label='p=0.01')
    ax3.axvline(0, color='k', ls=':', lw=0.5)
    ax3.set_ylabel('P-value')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylim([0.001, 1])
    ax3.legend(fontsize=8)
    ax3.set_title('P-values Over Time')
    
    # Panel 4: Tail comparison (ECDF)
    ax4 = fig.add_subplot(2, 2, 4)
    
    for res, color, name in [(results_shuffle, 'C0', 'trial_shuffle'), 
                              (results_shift, 'C1', 'circular_shift')]:
        null_vals = res['null_all'][:, t_max_idx]
        null_vals = null_vals[np.isfinite(null_vals)]
        sorted_vals = np.sort(null_vals)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax4.plot(sorted_vals, 1 - ecdf, color=color, label=name)  # Survival function
    
    ax4.axvline(obs_val, color='k', lw=2, ls='--', label='Observed')
    ax4.set_xlabel('ΔLL (bits)')
    ax4.set_ylabel('P(null ≥ x)')
    ax4.set_yscale('log')
    ax4.set_ylim([0.001, 1])
    ax4.legend(fontsize=8)
    ax4.set_title(f'Tail Probabilities at Peak (t={t_max:.0f} ms)')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace('.png', '.pdf'))
    print(f"Saved to {out_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY AT PEAK TIME (t={:.0f} ms)".format(t_max))
    print("="*60)
    print(f"  Observed: {obs_val:.2f} bits")
    print()
    
    for res, name in [(results_shuffle, 'trial_shuffle'), (results_shift, 'circular_shift')]:
        null_vals = res['null_all'][:, t_max_idx]
        null_vals = null_vals[np.isfinite(null_vals)]
        print(f"  {name}:")
        print(f"    Mean:   {np.mean(null_vals):.2f} bits")
        print(f"    Std:    {np.std(null_vals):.2f} bits")
        print(f"    95th:   {np.percentile(null_vals, 95):.2f} bits")
        print(f"    99th:   {np.percentile(null_vals, 99):.2f} bits")
        print(f"    Max:    {np.max(null_vals):.2f} bits")
        print(f"    P-value: {res['p_values'][t_max_idx]:.4f}")
        print()
    
    # Significance counts
    print("SIGNIFICANT TIME POINTS:")
    for res, name in [(results_shuffle, 'trial_shuffle'), (results_shift, 'circular_shift')]:
        p = res['p_values']
        n_total = np.sum(np.isfinite(p))
        n_05 = np.sum(p < 0.05)
        n_01 = np.sum(p < 0.01)
        print(f"  {name}: {n_05}/{n_total} (p<0.05), {n_01}/{n_total} (p<0.01)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare null methods with full distribution analysis")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--align", default="stim")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--feature", default="C", choices=["C", "R", "S", "T"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--orientation", default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--lags_ms", type=float, default=50.0)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int, default=500)
    args = ap.parse_args()
    
    areaA, areaB = args.areas
    
    print(f"Loading data for {args.sid}: {areaA} → {areaB}")
    cacheA = load_cache(args.out_root, args.align, args.sid, areaA)
    cacheB = load_cache(args.out_root, args.align, args.sid, areaB)
    axesA = load_axes(args.out_root, args.align, args.sid, areaA, args.tag)
    axesB = load_axes(args.out_root, args.align, args.sid, areaB, args.tag)
    
    print(f"\nRunning trial_shuffle ({args.perms} perms)...")
    results_shuffle = compute_flow_with_full_null(
        cacheA, cacheB, axesA, axesB,
        feature=args.feature,
        orientation=args.orientation,
        pt_min_ms=args.pt_min_ms,
        lags_ms=args.lags_ms,
        ridge=args.ridge,
        perms=args.perms,
        null_method="trial_shuffle",
        seed=0,
    )
    
    print(f"Running circular_shift ({args.perms} perms)...")
    results_shift = compute_flow_with_full_null(
        cacheA, cacheB, axesA, axesB,
        feature=args.feature,
        orientation=args.orientation,
        pt_min_ms=args.pt_min_ms,
        lags_ms=args.lags_ms,
        ridge=args.ridge,
        perms=args.perms,
        null_method="circular_shift",
        seed=0,
    )
    
    # Save and plot
    out_dir = os.path.join(args.out_root, args.align, args.sid, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"null_method_comparison_{areaA}_{areaB}_{args.feature}.png")
    plot_null_comparison(results_shuffle, results_shift, out_path)


if __name__ == "__main__":
    main()
