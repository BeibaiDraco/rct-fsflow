#!/usr/bin/env python
"""
Diagnostic script to analyze null distribution behavior in flow analysis.

Run this on a single session to understand:
1. Whether permutation is actually shuffling trials
2. How different null methods compare
3. Whether the null is overestimated
"""

import numpy as np
import json
import os
import sys
from typing import Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the flow functions (adjust path as needed)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def diagnose_labels(cache: Dict, mask: np.ndarray):
    """Check if labels are present and valid."""
    print("\n=== LABEL DIAGNOSTICS ===")
    
    for key in ["lab_C", "lab_R", "lab_S", "lab_orientation", "lab_PT_ms"]:
        if key in cache:
            vals = cache[key][mask] if mask is not None else cache[key]
            vals = np.asarray(vals)
            n_total = len(vals)
            
            if vals.dtype.kind in ['f', 'i']:  # numeric
                n_valid = np.sum(np.isfinite(vals))
                unique = np.unique(vals[np.isfinite(vals)])
                print(f"  {key}: {n_valid}/{n_total} valid, unique values: {unique[:10]}{'...' if len(unique) > 10 else ''}")
            else:  # string/object
                n_valid = np.sum(vals.astype(str) != '')
                unique = np.unique(vals)
                print(f"  {key}: {n_valid}/{n_total} valid, unique values: {unique[:10]}")
        else:
            print(f"  {key}: MISSING!")
    
def diagnose_strata(cache: Dict, mask: np.ndarray, perm_within: str = "CR"):
    """Check how many trials are in each stratum."""
    print(f"\n=== STRATA DIAGNOSTICS (perm_within={perm_within}) ===")
    
    N = mask.sum()
    
    # Get labels
    C = cache.get("lab_C", np.full(len(mask), np.nan))[mask].astype(float)
    R = cache.get("lab_R", np.full(len(mask), np.nan))[mask].astype(float)
    
    if perm_within == "CR":
        # Encode joint C×R
        valid = np.isfinite(C) & np.isfinite(R)
        if not np.any(valid):
            print("  WARNING: No trials have valid C AND R labels!")
            print("  This means NO SHUFFLING will occur!")
            return
        
        codes = C[valid] * 1000 + R[valid]
        unique_codes, counts = np.unique(codes, return_counts=True)
        print(f"  {len(unique_codes)} strata with sizes: {sorted(counts, reverse=True)}")
        print(f"  Valid trials: {valid.sum()}/{N}")
        
        if np.any(counts == 1):
            n_singletons = np.sum(counts == 1)
            print(f"  WARNING: {n_singletons} strata have only 1 trial (cannot shuffle)")
            
    elif perm_within == "C":
        valid = np.isfinite(C)
        unique, counts = np.unique(C[valid], return_counts=True)
        print(f"  C strata: {dict(zip(unique, counts))}")
        print(f"  Valid trials: {valid.sum()}/{N}")
        
    elif perm_within == "R":
        valid = np.isfinite(R)
        unique, counts = np.unique(R[valid], return_counts=True)
        print(f"  R strata: {dict(zip(unique, counts))}")
        print(f"  Valid trials: {valid.sum()}/{N}")
        
    elif perm_within == "none":
        print(f"  All {N} trials in single stratum (full shuffle)")


def compare_null_methods(cacheA: Dict, cacheB: Dict, axesA: Dict, axesB: Dict,
                         feature: str, align: str, orientation: str,
                         pt_min_ms: float, lags_ms: float, ridge: float,
                         perms: int = 100, seed: int = 0) -> Dict:
    """
    Compare different null methods at a single representative time bin.
    """
    from paperflow.flow import compute_flow_timecourse_for_pair
    
    results = {}
    
    for null_method in ["trial_shuffle", "circular_shift"]:
        print(f"\n  Running {null_method}...")
        res = compute_flow_timecourse_for_pair(
            cacheA=cacheA, cacheB=cacheB,
            axesA=axesA, axesB=axesB,
            feature=feature, align=align,
            orientation=orientation,
            pt_min_ms=pt_min_ms,
            lags_ms=lags_ms, ridge=ridge,
            perms=perms, induced=True,
            include_B_lags=True, seed=seed,
            perm_within="CR",
            null_method=null_method,
        )
        results[null_method] = res
        
    return results


def plot_null_comparison(results: Dict, out_path: str):
    """Plot comparison of null methods."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    time = results["trial_shuffle"]["time"] * 1000  # ms
    
    # A -> B
    ax = axes[0, 0]
    obs = results["trial_shuffle"]["bits_AtoB"]
    ax.plot(time, obs, 'k-', lw=2, label='Observed')
    
    for method, color in [("trial_shuffle", "C0"), ("circular_shift", "C1")]:
        mu = results[method]["null_mean_AtoB"]
        sd = results[method]["null_std_AtoB"]
        ax.fill_between(time, mu-sd, mu+sd, alpha=0.3, color=color, label=f'{method} null ±1σ')
        ax.plot(time, mu, color=color, ls='--')
    
    ax.axhline(0, ls=':', c='k', lw=0.5)
    ax.axvline(0, ls=':', c='k', lw=0.5)
    ax.set_ylabel('ΔLL (bits)')
    ax.set_title('A → B flow')
    ax.legend(fontsize=8)
    
    # B -> A
    ax = axes[0, 1]
    obs = results["trial_shuffle"]["bits_BtoA"]
    ax.plot(time, obs, 'k-', lw=2, label='Observed')
    
    for method, color in [("trial_shuffle", "C0"), ("circular_shift", "C1")]:
        mu = results[method]["null_mean_BtoA"]
        sd = results[method]["null_std_BtoA"]
        ax.fill_between(time, mu-sd, mu+sd, alpha=0.3, color=color)
        ax.plot(time, mu, color=color, ls='--')
    
    ax.axhline(0, ls=':', c='k', lw=0.5)
    ax.axvline(0, ls=':', c='k', lw=0.5)
    ax.set_ylabel('ΔLL (bits)')
    ax.set_title('B → A flow')
    
    # P-values A -> B
    ax = axes[1, 0]
    for method, color in [("trial_shuffle", "C0"), ("circular_shift", "C1")]:
        p = results[method]["p_AtoB"]
        ax.semilogy(time, p, color=color, label=method)
    ax.axhline(0.05, ls='--', c='r', lw=1, label='p=0.05')
    ax.axhline(0.01, ls=':', c='r', lw=1, label='p=0.01')
    ax.axvline(0, ls=':', c='k', lw=0.5)
    ax.set_ylabel('p-value')
    ax.set_xlabel('Time (ms)')
    ax.set_title('P-values (A → B)')
    ax.legend(fontsize=8)
    ax.set_ylim([1e-3, 1])
    
    # P-values B -> A
    ax = axes[1, 1]
    for method, color in [("trial_shuffle", "C0"), ("circular_shift", "C1")]:
        p = results[method]["p_BtoA"]
        ax.semilogy(time, p, color=color, label=method)
    ax.axhline(0.05, ls='--', c='r', lw=1)
    ax.axhline(0.01, ls=':', c='r', lw=1)
    ax.axvline(0, ls=':', c='k', lw=0.5)
    ax.set_ylabel('p-value')
    ax.set_xlabel('Time (ms)')
    ax.set_title('P-values (B → A)')
    ax.set_ylim([1e-3, 1])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace('.png', '.pdf'))
    print(f"\nSaved comparison plot to {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Diagnose null distribution issues")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--align", choices=["stim", "sacc"], default="stim")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--feature", choices=["C", "R", "S"], default="C")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--orientation", default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--lags_ms", type=float, default=50.0)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int, default=100, help="Permutations for comparison (use fewer for speed)")
    ap.add_argument("--compare", action="store_true", help="Compare different null methods")
    args = ap.parse_args()
    
    areaA, areaB = args.areas
    
    print(f"=== NULL DISTRIBUTION DIAGNOSTICS ===")
    print(f"Session: {args.sid}")
    print(f"Areas: {areaA} → {areaB}")
    print(f"Feature: {args.feature}")
    print(f"Alignment: {args.align}")
    
    # Load caches
    cacheA = load_cache(args.out_root, args.align, args.sid, areaA)
    cacheB = load_cache(args.out_root, args.align, args.sid, areaB)
    
    # Compute mask
    N = cacheA["Z"].shape[0]
    mask = np.ones(N, dtype=bool)
    if "lab_is_correct" in cacheA:
        mask &= cacheA["lab_is_correct"].astype(bool)
    if args.orientation and "lab_orientation" in cacheA:
        mask &= (cacheA["lab_orientation"].astype(str) == args.orientation)
    if args.pt_min_ms and "lab_PT_ms" in cacheA:
        PT = cacheA["lab_PT_ms"].astype(float)
        mask &= np.isfinite(PT) & (PT >= args.pt_min_ms)
    
    print(f"\nTrials: {mask.sum()}/{N} pass filters")
    
    # Diagnose labels
    diagnose_labels(cacheA, mask)
    
    # Diagnose strata
    for perm_within in ["CR", "C", "R", "none"]:
        diagnose_strata(cacheA, mask, perm_within)
    
    if args.compare:
        print("\n=== COMPARING NULL METHODS ===")
        axesA = load_axes(args.out_root, args.align, args.sid, areaA, args.tag)
        axesB = load_axes(args.out_root, args.align, args.sid, areaB, args.tag)
        
        results = compare_null_methods(
            cacheA, cacheB, axesA, axesB,
            feature=args.feature, align=args.align,
            orientation=args.orientation,
            pt_min_ms=args.pt_min_ms,
            lags_ms=args.lags_ms, ridge=args.ridge,
            perms=args.perms
        )
        
        # Print summary statistics
        print("\n=== NULL DISTRIBUTION SUMMARY ===")
        for method in ["trial_shuffle", "circular_shift"]:
            mu_A = np.nanmean(results[method]["null_mean_AtoB"])
            mu_B = np.nanmean(results[method]["null_mean_BtoA"])
            print(f"  {method}:")
            print(f"    Mean null A→B: {mu_A:.4f} bits")
            print(f"    Mean null B→A: {mu_B:.4f} bits")
        
        # Save plot
        out_dir = os.path.join(args.out_root, args.align, args.sid, "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"null_comparison_{areaA}_{areaB}_{args.feature}.png")
        plot_null_comparison(results, out_path)


if __name__ == "__main__":
    main()
