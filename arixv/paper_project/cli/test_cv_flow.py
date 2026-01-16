#!/usr/bin/env python
"""
Compare old (training) vs new (cross-validated) flow computation.

This demonstrates:
1. Training-based flow has ~4 bit null (overfitting bias)
2. Cross-validated flow has ~0 bit null (unbiased)
3. The EXCESS (observed - null) should be similar, but CV gives cleaner significance
"""

import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperflow.flow_cv import compute_flow_cv


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
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int, default=200, help="Fewer perms for speed")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()
    
    areaA, areaB = args.areas
    
    print(f"Loading {args.sid}: {areaA} → {areaB}")
    cacheA = load_cache(args.out_root, args.align, args.sid, areaA)
    cacheB = load_cache(args.out_root, args.align, args.sid, areaB)
    axesA = load_axes(args.out_root, args.align, args.sid, areaA, args.tag)
    axesB = load_axes(args.out_root, args.align, args.sid, areaB, args.tag)
    
    print(f"Computing flow with {args.n_folds}-fold CV and {args.perms} permutations...")
    print("(This may take a few minutes)")
    
    results = compute_flow_cv(
        cacheA, cacheB, axesA, axesB,
        feature=args.feature,
        align=args.align,
        orientation=args.orientation,
        pt_min_ms=args.pt_min_ms,
        lags_ms=args.lags_ms,
        ridge=args.ridge,
        n_folds=args.n_folds,
        perms=args.perms,
        null_method="circular_shift",
        seed=42,
    )
    
    time = results['time'] * 1000  # ms
    
    # Extract results
    obs_cv = results['bits_AtoB']
    obs_train = results['bits_AtoB_train']
    null_mean_cv = results['null_mean_AtoB']
    null_std_cv = results['null_std_AtoB']
    null_all = results['null_all_AtoB']
    p_values = results['p_AtoB']
    
    # Compute training null mean (approximately = CV observed - CV excess)
    # Actually, let's compute it from the training values
    # The null for training is ~4 bits (overfitting bias)
    
    # Find peak
    t_max_idx = np.nanargmax(obs_cv)
    t_max = time[t_max_idx]
    
    # Plot comparison
    fig, axes_arr = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Training vs CV observed
    ax = axes_arr[0, 0]
    ax.plot(time, obs_train, 'r-', lw=2, label='Training (biased)', alpha=0.7)
    ax.plot(time, obs_cv, 'b-', lw=2, label='Cross-validated (unbiased)')
    ax.axhline(0, color='k', ls=':', lw=0.5)
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.axvline(t_max, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel('ΔLL (bits)')
    ax.set_xlabel('Time (ms)')
    ax.legend()
    ax.set_title('Observed Flow: Training vs Cross-Validated')
    
    # Panel 2: CV with null band
    ax = axes_arr[0, 1]
    ax.fill_between(time, null_mean_cv - 2*null_std_cv, null_mean_cv + 2*null_std_cv, 
                    alpha=0.2, color='gray', label='Null ±2σ')
    ax.plot(time, null_mean_cv, 'k--', lw=1, label='Null mean')
    ax.plot(time, obs_cv, 'b-', lw=2, label='Observed (CV)')
    ax.axhline(0, color='k', ls=':', lw=0.5)
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('ΔLL (bits)')
    ax.set_xlabel('Time (ms)')
    ax.legend()
    ax.set_title('Cross-Validated Flow with Null Distribution')
    
    # Panel 3: Null distribution histogram at peak
    ax = axes_arr[1, 0]
    null_at_peak = null_all[:, t_max_idx]
    null_at_peak = null_at_peak[np.isfinite(null_at_peak)]
    
    ax.hist(null_at_peak, bins=30, alpha=0.7, color='gray', density=True, label='Null (CV)')
    ax.axvline(obs_cv[t_max_idx], color='b', lw=2, label=f'Observed: {obs_cv[t_max_idx]:.1f}')
    ax.axvline(0, color='k', ls=':', lw=1)
    ax.set_xlabel('ΔLL (bits)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(f'Null Distribution at Peak (t={t_max:.0f} ms)')
    
    # Panel 4: P-values
    ax = axes_arr[1, 1]
    ax.semilogy(time, p_values, 'b-', lw=1.5)
    ax.axhline(0.05, color='r', ls='--', lw=1, label='p=0.05')
    ax.axhline(0.01, color='r', ls=':', lw=1, label='p=0.01')
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('P-value')
    ax.set_xlabel('Time (ms)')
    ax.set_ylim([0.001, 1])
    ax.legend()
    ax.set_title('P-values Over Time (Cross-Validated)')
    
    plt.tight_layout()
    
    out_dir = os.path.join(args.out_root, args.align, args.sid, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cv_flow_{areaA}_{areaB}_{args.feature}.png")
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace('.png', '.pdf'))
    print(f"\nSaved figure to {out_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nAt peak time (t={t_max:.0f} ms):")
    print(f"  Training observed:  {obs_train[t_max_idx]:.2f} bits (includes ~4 bit overfitting bias)")
    print(f"  CV observed:        {obs_cv[t_max_idx]:.2f} bits (unbiased)")
    print(f"  CV null mean:       {null_mean_cv[t_max_idx]:.2f} bits (should be ~0)")
    print(f"  CV null std:        {null_std_cv[t_max_idx]:.2f} bits")
    print(f"  P-value:            {p_values[t_max_idx]:.4f}")
    
    # Check if null is centered around zero
    overall_null_mean = np.nanmean(null_mean_cv)
    print(f"\nOverall null mean: {overall_null_mean:.2f} bits")
    if abs(overall_null_mean) < 0.5:
        print("  ✓ Null is centered near zero (overfitting bias removed)")
    else:
        print(f"  ⚠ Null is not centered at zero - may indicate remaining bias")
    
    # Count significant time points
    n_total = np.sum(np.isfinite(p_values))
    n_05 = np.sum(p_values < 0.05)
    n_01 = np.sum(p_values < 0.01)
    n_001 = np.sum(p_values < 0.001)
    
    print(f"\nSignificant time points:")
    print(f"  p < 0.05:  {n_05}/{n_total} ({100*n_05/n_total:.1f}%)")
    print(f"  p < 0.01:  {n_01}/{n_total} ({100*n_01/n_total:.1f}%)")
    print(f"  p < 0.001: {n_001}/{n_total} ({100*n_001/n_total:.1f}%)")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
The key difference:

OLD METHOD (Training):
  - SSE computed on same data used to fit regression
  - Adding parameters ALWAYS reduces SSE (overfitting)
  - Null ≈ 4 bits (overfitting bias)
  - Must compare observed to null (which is also ~4 bits)

NEW METHOD (Cross-Validated):
  - SSE computed on HELD-OUT data
  - Random predictors DON'T help on held-out data
  - Null ≈ 0 bits (no overfitting bias)
  - Observed directly measures predictive information
  
If CV observed >> 0, there is REAL predictive information flow.
If CV observed ≈ 0, there is NO flow (A doesn't help predict B).
""")


if __name__ == "__main__":
    main()
