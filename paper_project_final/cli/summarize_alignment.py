#!/usr/bin/env python3
"""
Summarize Alignment Analysis: Aggregate per-session results and generate figures.

This script collects per-session alignment results from:
  out/sacc/alignment/{tag}/AI_*.json
  out/sacc/alignment/{tag}/axis_shuffle_*.json
  out/sacc/alignment/{tag}/axis_cov_*.json

And produces:
  - Group-level statistics with sign-flip tests
  - Summary JSON files
  - Publication-ready figures

THREE ANALYSES:
  A) Alignment Index (PCA subspaces + covariance null) - Tests subspace orthogonality
  B) Axis Shuffle (trained axes + label-shuffle null) - Tests label-specificity
  C) Axis Covariance (trained axes + covariance null) - Tests axis orthogonality (DAVE'S QUESTION)
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_session_results(results_dir: Path, analysis_type: str) -> List[Dict]:
    """Load all per-session results for a given analysis type."""
    results = []
    
    pattern = f"{analysis_type}_*.json"
    for json_path in sorted(results_dir.glob(pattern)):
        try:
            with open(json_path) as f:
                result = json.load(f)
            results.append(result)
        except Exception as e:
            print(f"[warn] Failed to load {json_path}: {e}")
    
    return results


def signflip_exact_pvalue_group(session_deltas: np.ndarray, 
                                 alternative: str = "two-sided") -> Dict:
    """
    Exact sign-flip test on per-session deltas.
    
    H0: Δ is symmetric around 0 (no systematic difference from chance)
    """
    x = np.asarray(session_deltas, dtype=float)
    x = x[np.isfinite(x)]
    S = x.size
    
    if S < 3:
        return dict(p=np.nan, obs_mean=np.nan, n_sessions=int(S))
    
    obs = float(np.mean(x))
    n = 1 << S  # 2^S
    
    # For large S, use Monte Carlo approximation
    if S > 20:
        n_mc = 100000
        rng = np.random.default_rng(42)
        null = np.empty(n_mc, dtype=float)
        for i in range(n_mc):
            signs = rng.choice([-1.0, 1.0], size=S)
            null[i] = np.mean(signs * x)
        
        if alternative == "greater":
            p = (1 + np.sum(null >= obs)) / (1 + n_mc)
        elif alternative == "less":
            p = (1 + np.sum(null <= obs)) / (1 + n_mc)
        else:
            p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_mc)
    else:
        null = np.empty(n, dtype=float)
        for mask in range(n):
            tot = 0.0
            for i in range(S):
                sign = -1.0 if ((mask >> i) & 1) else 1.0
                tot += sign * x[i]
            null[mask] = tot / S
        
        if alternative == "greater":
            p = (1 + np.sum(null >= obs)) / (1 + n)
        elif alternative == "less":
            p = (1 + np.sum(null <= obs)) / (1 + n)
        else:
            p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n)
    
    return dict(p=float(p), obs_mean=obs, n_sessions=int(S))


def plot_alignment_index_summary(results: List[Dict], out_dir: Path, tag: str):
    """Create summary plots for Alignment Index analysis."""
    if not results:
        print("[warn] No AI results to plot")
        return
    
    # Separate by monkey
    M_results = [r for r in results if r["monkey"] == "M"]
    S_results = [r for r in results if r["monkey"] == "S"]
    
    for monkey, monkey_results in [("M", M_results), ("S", S_results), ("all", results)]:
        if not monkey_results:
            continue
        
        AI_obs_arr = np.array([r["AI_obs"] for r in monkey_results])
        AI_null_arr = np.array([r["null_AI_mean"] for r in monkey_results])
        delta_arr = np.array([r["delta_AI"] for r in monkey_results])
        angle_obs_arr = np.array([r["mean_angle_obs_deg"] for r in monkey_results])
        angle_null_arr = np.array([r["null_angle_mean_deg"] for r in monkey_results])
        
        # Figure 1: AI scatter (obs vs null mean)
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.scatter(AI_null_arr, AI_obs_arr, c='blue', s=80, alpha=0.7, edgecolors='k')
        
        # Unity line
        lim = [0, max(max(AI_null_arr), max(AI_obs_arr)) * 1.1]
        ax.plot(lim, lim, 'k--', alpha=0.5, label='y=x')
        
        # Mean
        mean_obs = np.mean(AI_obs_arr)
        mean_null = np.mean(AI_null_arr)
        ax.scatter([mean_null], [mean_obs], c='red', s=200, marker='*', 
                   edgecolors='darkred', linewidths=2, zorder=10, label='Mean')
        
        ax.set_xlabel('Null mean AI (covariance-matched)', fontsize=14)
        ax.set_ylabel('Observed AI', fontsize=14)
        ax.set_title(f'Monkey {monkey}: Alignment Index\n'
                     f'N={len(monkey_results)}, Δ_AI={np.mean(delta_arr):.3f}±{np.std(delta_arr)/np.sqrt(len(delta_arr)):.3f}',
                     fontsize=12)
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        
        plt.tight_layout()
        fig.savefig(out_dir / f"AI_scatter_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"AI_scatter_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
        
        # Figure 2: Angle comparison
        fig, ax = plt.subplots(figsize=(5, 6))
        
        positions = [1, 2]
        bp = ax.boxplot([angle_null_arr, angle_obs_arr], positions=positions, 
                        widths=0.5, patch_artist=True)
        
        bp['boxes'][0].set_facecolor('lightgray')
        bp['boxes'][1].set_facecolor('lightblue')
        
        # Individual points with jitter
        jitter = 0.1
        rng = np.random.default_rng(42)
        for i, (pos, data) in enumerate([(1, angle_null_arr), (2, angle_obs_arr)]):
            x = pos + rng.uniform(-jitter, jitter, len(data))
            ax.scatter(x, data, c='k', s=30, alpha=0.5, zorder=5)
        
        # Connect paired observations
        for i in range(len(angle_null_arr)):
            ax.plot([1, 2], [angle_null_arr[i], angle_obs_arr[i]], 'k-', alpha=0.2)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Null\n(cov-matched)', 'Observed'], fontsize=12)
        ax.set_ylabel('Mean Principal Angle (degrees)', fontsize=14)
        ax.set_title(f'Monkey {monkey}: Subspace Angle\n'
                     f'Δθ = {np.mean(angle_obs_arr - angle_null_arr):.1f}°', fontsize=12)
        ax.set_ylim(0, 90)
        
        plt.tight_layout()
        fig.savefig(out_dir / f"AI_angle_boxplot_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"AI_angle_boxplot_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
        
        # Figure 3: Delta distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.hist(delta_arr, bins=15, color='steelblue', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--', linewidth=2, label='Δ=0')
        ax.axvline(np.mean(delta_arr), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean={np.mean(delta_arr):.3f}')
        
        ax.set_xlabel('Δ_AI = obs - null_mean', fontsize=14)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Monkey {monkey}: AI Effect Size', fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(out_dir / f"AI_delta_hist_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"AI_delta_hist_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
    
    print(f"[ok] Saved AI plots to {out_dir}")


def plot_axis_shuffle_summary(results: List[Dict], out_dir: Path, tag: str):
    """Create summary plots for axis-shuffle analysis."""
    if not results:
        print("[warn] No axis-shuffle results to plot")
        return
    
    # Separate by monkey
    M_results = [r for r in results if r["monkey"] == "M"]
    S_results = [r for r in results if r["monkey"] == "S"]
    
    for monkey, monkey_results in [("M", M_results), ("S", S_results), ("all", results)]:
        if not monkey_results:
            continue
        
        a_obs_arr = np.array([r["a_obs"] for r in monkey_results])
        null_mean_arr = np.array([r["null_mean"] for r in monkey_results])
        theta_obs_arr = np.array([r["theta_obs_deg"] for r in monkey_results])
        theta_null_arr = np.array([r["null_angle_mean_deg"] for r in monkey_results])
        delta_arr = np.array([r["delta"] for r in monkey_results])
        
        # Figure 1: Alignment scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.scatter(null_mean_arr, a_obs_arr, c='blue', s=80, alpha=0.7, edgecolors='k')
        
        lim = [0, max(max(null_mean_arr), max(a_obs_arr)) * 1.1]
        ax.plot(lim, lim, 'k--', alpha=0.5, label='y=x')
        
        mean_obs = np.mean(a_obs_arr)
        mean_null = np.mean(null_mean_arr)
        ax.scatter([mean_null], [mean_obs], c='red', s=200, marker='*', 
                   edgecolors='darkred', linewidths=2, zorder=10, label='Mean')
        
        ax.set_xlabel('Null mean |cos(θ)| (label-shuffle)', fontsize=14)
        ax.set_ylabel('Observed |cos(θ)|', fontsize=14)
        ax.set_title(f'Monkey {monkey}: Axis Alignment (shuffle null)\n'
                     f'N={len(monkey_results)}, Δ={np.mean(delta_arr):.3f}±{np.std(delta_arr)/np.sqrt(len(delta_arr)):.3f}',
                     fontsize=12)
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        
        plt.tight_layout()
        fig.savefig(out_dir / f"axis_scatter_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"axis_scatter_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
        
        # Figure 2: Angle boxplot
        fig, ax = plt.subplots(figsize=(5, 6))
        
        positions = [1, 2]
        bp = ax.boxplot([theta_null_arr, theta_obs_arr], positions=positions, 
                        widths=0.5, patch_artist=True)
        
        bp['boxes'][0].set_facecolor('lightgray')
        bp['boxes'][1].set_facecolor('lightblue')
        
        jitter = 0.1
        rng = np.random.default_rng(42)
        for i, (pos, data) in enumerate([(1, theta_null_arr), (2, theta_obs_arr)]):
            x = pos + rng.uniform(-jitter, jitter, len(data))
            ax.scatter(x, data, c='k', s=30, alpha=0.5, zorder=5)
        
        for i in range(len(theta_null_arr)):
            ax.plot([1, 2], [theta_null_arr[i], theta_obs_arr[i]], 'k-', alpha=0.2)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Null\n(label-shuffle)', 'Observed'], fontsize=12)
        ax.set_ylabel('Angle θ (degrees)', fontsize=14)
        ax.set_title(f'Monkey {monkey}: Axis Angle\n'
                     f'Δθ = {np.mean(theta_obs_arr - theta_null_arr):.1f}°', fontsize=12)
        ax.set_ylim(0, 90)
        
        plt.tight_layout()
        fig.savefig(out_dir / f"axis_angle_boxplot_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"axis_angle_boxplot_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
    
    print(f"[ok] Saved axis-shuffle plots to {out_dir}")


def plot_axis_covariance_summary(results: List[Dict], out_dir: Path, tag: str):
    """Create summary plots for axis-covariance analysis (Analysis C - DAVE'S QUESTION)."""
    if not results:
        print("[warn] No axis-covariance results to plot")
        return
    
    # Separate by monkey
    M_results = [r for r in results if r["monkey"] == "M"]
    S_results = [r for r in results if r["monkey"] == "S"]
    
    for monkey, monkey_results in [("M", M_results), ("S", S_results), ("all", results)]:
        if not monkey_results:
            continue
        
        a_obs_arr = np.array([r["a_obs"] for r in monkey_results])
        null_mean_arr = np.array([r["null_mean"] for r in monkey_results])
        theta_obs_arr = np.array([r["theta_obs_deg"] for r in monkey_results])
        theta_null_arr = np.array([r["null_angle_mean_deg"] for r in monkey_results])
        delta_arr = np.array([r["delta"] for r in monkey_results])
        z_scores = np.array([r["z_score"] for r in monkey_results])
        expected_cos_arr = np.array([r.get("expected_cos", np.nan) for r in monkey_results])
        
        # Figure 1: Alignment scatter (obs vs null mean)
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.scatter(null_mean_arr, a_obs_arr, c='blue', s=80, alpha=0.7, edgecolors='k')
        
        lim = [0, max(max(null_mean_arr), max(a_obs_arr)) * 1.1]
        ax.plot(lim, lim, 'k--', alpha=0.5, label='y=x')
        
        mean_obs = np.mean(a_obs_arr)
        mean_null = np.mean(null_mean_arr)
        ax.scatter([mean_null], [mean_obs], c='red', s=200, marker='*', 
                   edgecolors='darkred', linewidths=2, zorder=10, label='Mean')
        
        # Mark theoretical expectation
        if np.any(np.isfinite(expected_cos_arr)):
            ax.axhline(np.nanmean(expected_cos_arr), color='green', linestyle=':', 
                      alpha=0.7, label=f'E[|cos|]≈{np.nanmean(expected_cos_arr):.3f}')
        
        ax.set_xlabel('Null mean |cos(θ)| (covariance-constrained)', fontsize=14)
        ax.set_ylabel('Observed |cos(θ)|', fontsize=14)
        ax.set_title(f'Monkey {monkey}: Axis Alignment (covariance null)\n'
                     f'N={len(monkey_results)}, Δ={np.mean(delta_arr):.3f}±{np.std(delta_arr)/np.sqrt(len(delta_arr)):.3f}',
                     fontsize=12)
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        
        plt.tight_layout()
        fig.savefig(out_dir / f"axis_cov_scatter_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"axis_cov_scatter_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
        
        # Figure 2: Angle boxplot
        fig, ax = plt.subplots(figsize=(5, 6))
        
        positions = [1, 2]
        bp = ax.boxplot([theta_null_arr, theta_obs_arr], positions=positions, 
                        widths=0.5, patch_artist=True)
        
        bp['boxes'][0].set_facecolor('lightgray')
        bp['boxes'][1].set_facecolor('lightgreen')  # Different color to distinguish from shuffle
        
        jitter = 0.1
        rng = np.random.default_rng(42)
        for i, (pos, data) in enumerate([(1, theta_null_arr), (2, theta_obs_arr)]):
            x = pos + rng.uniform(-jitter, jitter, len(data))
            ax.scatter(x, data, c='k', s=30, alpha=0.5, zorder=5)
        
        for i in range(len(theta_null_arr)):
            ax.plot([1, 2], [theta_null_arr[i], theta_obs_arr[i]], 'k-', alpha=0.2)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Null\n(covariance)', 'Observed'], fontsize=12)
        ax.set_ylabel('Angle θ (degrees)', fontsize=14)
        ax.set_title(f'Monkey {monkey}: Axis Angle (cov-null)\n'
                     f'Δθ = {np.mean(theta_obs_arr - theta_null_arr):.1f}°', fontsize=12)
        ax.set_ylim(0, 90)
        
        plt.tight_layout()
        fig.savefig(out_dir / f"axis_cov_angle_boxplot_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"axis_cov_angle_boxplot_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
        
        # Figure 3: Z-score histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.hist(z_scores, bins=15, color='seagreen', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--', linewidth=2, label='z=0')
        ax.axvline(np.mean(z_scores), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean z={np.mean(z_scores):.2f}')
        
        ax.set_xlabel('z-score = (obs - null_mean) / null_std', fontsize=14)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Monkey {monkey}: Axis Alignment Effect Size', fontsize=12)
        ax.legend()
        
        # Add interpretation
        if np.mean(z_scores) < -1:
            ax.text(0.02, 0.98, 'More orthogonal\nthan chance', transform=ax.transAxes,
                   fontsize=10, va='top', ha='left', color='green', fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(out_dir / f"axis_cov_zscore_hist_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"axis_cov_zscore_hist_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
        
        # Figure 4: Delta histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.hist(delta_arr, bins=15, color='seagreen', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--', linewidth=2, label='Δ=0')
        ax.axvline(np.mean(delta_arr), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean={np.mean(delta_arr):.3f}')
        
        ax.set_xlabel('Δ = |cos(θ)|_obs - null_mean', fontsize=14)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Monkey {monkey}: Axis Alignment Δ (covariance null)', fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(out_dir / f"axis_cov_delta_hist_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"axis_cov_delta_hist_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
    
    print(f"[ok] Saved axis-covariance plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize alignment analysis results across sessions"
    )
    
    parser.add_argument("--out_root", default="out", help="Output root directory")
    parser.add_argument("--tag", default="alignment", help="Analysis tag")
    parser.add_argument("--mode", choices=["AI", "axis_shuffle", "axis_cov", "all", "both"], default="all",
                        help="Which analysis to summarize: 'AI', 'axis_shuffle', 'axis_cov', 'all' (all three), or 'both' (A+B only)")
    
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    results_dir = out_root / "sacc" / "alignment" / args.tag
    
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")
    
    # Create summary output directory
    summary_dir = out_root / "sacc" / "summary" / f"alignment_{args.tag}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = summary_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("SUMMARIZING ALIGNMENT ANALYSIS RESULTS")
    print("="*70)
    print(f"[info] Results dir: {results_dir}")
    print(f"[info] Summary dir: {summary_dir}")
    
    # Process Alignment Index results
    if args.mode in ["AI", "both", "all"]:
        print(f"\n{'='*70}")
        print("[Analysis A] ALIGNMENT INDEX RESULTS (PCA subspaces)")
        print("="*70)
        
        results_AI = load_session_results(results_dir, "AI")
        
        if results_AI:
            print(f"[info] Loaded {len(results_AI)} sessions")
            
            deltas = np.array([r["delta_AI"] for r in results_AI])
            
            # Group-level test: are subspaces less aligned than chance?
            test_less = signflip_exact_pvalue_group(deltas, "less")
            test_greater = signflip_exact_pvalue_group(deltas, "greater")
            
            print(f"\n  Mean AI_obs: {np.mean([r['AI_obs'] for r in results_AI]):.4f}")
            print(f"  Mean AI_null: {np.mean([r['null_AI_mean'] for r in results_AI]):.4f}")
            print(f"  Mean Δ_AI: {np.mean(deltas):.4f} ± {np.std(deltas)/np.sqrt(len(deltas)):.4f}")
            print(f"\n  Group-level tests (sign-flip on Δ_AI):")
            print(f"    p(less aligned than chance, Δ<0): {test_less['p']:.4f}")
            print(f"    p(more aligned than chance, Δ>0): {test_greater['p']:.4f}")
            
            # Interpretation
            if test_less['p'] < 0.05:
                print(f"\n  >>> SIGNIFICANT: Category and saccade subspaces are LESS aligned")
                print(f"      than expected given the neural manifold constraints.")
            elif test_greater['p'] < 0.05:
                print(f"\n  >>> SIGNIFICANT: Category and saccade subspaces are MORE aligned")
                print(f"      than expected given the neural manifold constraints.")
            else:
                print(f"\n  >>> NOT SIGNIFICANT: Alignment is consistent with chance")
                print(f"      given the neural manifold constraints.")
            
            # Save summary
            summary_AI = {
                "analysis": "alignment_index",
                "description": "SC-style Alignment Index with covariance-matched null",
                "hypothesis": "Tests whether category and saccade subspaces are less aligned than chance",
                "created": datetime.now().isoformat(),
                "n_sessions": len(results_AI),
                "mean_AI_obs": float(np.mean([r["AI_obs"] for r in results_AI])),
                "mean_AI_null": float(np.mean([r["null_AI_mean"] for r in results_AI])),
                "mean_delta_AI": float(np.mean(deltas)),
                "sem_delta_AI": float(np.std(deltas) / np.sqrt(len(deltas))),
                "mean_angle_obs_deg": float(np.mean([r["mean_angle_obs_deg"] for r in results_AI])),
                "mean_angle_null_deg": float(np.mean([r["null_angle_mean_deg"] for r in results_AI])),
                "group_p_less": test_less["p"],
                "group_p_greater": test_greater["p"],
                "per_session": results_AI,
            }
            
            with open(summary_dir / "summary_alignment_index.json", "w") as f:
                json.dump(summary_AI, f, indent=2)
            
            # Save NPZ with all deltas
            np.savez_compressed(
                summary_dir / "results_alignment_index.npz",
                sids=np.array([r["sid"] for r in results_AI]),
                deltas=deltas,
                AI_obs=np.array([r["AI_obs"] for r in results_AI]),
                AI_null_mean=np.array([r["null_AI_mean"] for r in results_AI]),
            )
            
            # Plot
            plot_alignment_index_summary(results_AI, figs_dir, args.tag)
        else:
            print("[warn] No AI results found")
    
    # Process axis-shuffle results
    if args.mode in ["axis_shuffle", "both", "all"]:
        print(f"\n{'='*70}")
        print("[Analysis B] AXIS-SHUFFLE RESULTS (label-specificity test)")
        print("="*70)
        
        results_axis = load_session_results(results_dir, "axis_shuffle")
        
        if results_axis:
            print(f"[info] Loaded {len(results_axis)} sessions")
            
            deltas = np.array([r["delta"] for r in results_axis])
            
            # Group-level test
            test_greater = signflip_exact_pvalue_group(deltas, "greater")
            test_less = signflip_exact_pvalue_group(deltas, "less")
            
            print(f"\n  Mean |cos(θ)|_obs: {np.mean([r['a_obs'] for r in results_axis]):.4f}")
            print(f"  Mean |cos(θ)|_null: {np.mean([r['null_mean'] for r in results_axis]):.4f}")
            print(f"  Mean Δ: {np.mean(deltas):.4f} ± {np.std(deltas)/np.sqrt(len(deltas)):.4f}")
            print(f"\n  Group-level tests (sign-flip on Δ):")
            print(f"    p(more aligned than shuffle, Δ>0): {test_greater['p']:.4f}")
            print(f"    p(less aligned than shuffle, Δ<0): {test_less['p']:.4f}")
            
            print(f"\n  NOTE: The shuffle null tests whether alignment is LABEL-SPECIFIC.")
            print(f"        It is NOT appropriate for testing 'orthogonal by design'.")
            print(f"        Use Analysis C (axis_cov) for that question.")
            
            # Save summary
            summary_axis = {
                "analysis": "axis_shuffle",
                "description": "Axis angle with label-shuffle null",
                "hypothesis": "Tests whether C-S axis alignment is label-specific (greater than shuffle)",
                "note": "NOT appropriate for testing orthogonality; use axis_cov (Analysis C) instead",
                "created": datetime.now().isoformat(),
                "n_sessions": len(results_axis),
                "mean_a_obs": float(np.mean([r["a_obs"] for r in results_axis])),
                "mean_null": float(np.mean([r["null_mean"] for r in results_axis])),
                "mean_delta": float(np.mean(deltas)),
                "sem_delta": float(np.std(deltas) / np.sqrt(len(deltas))),
                "mean_theta_obs_deg": float(np.mean([r["theta_obs_deg"] for r in results_axis])),
                "mean_theta_null_deg": float(np.mean([r["null_angle_mean_deg"] for r in results_axis])),
                "group_p_greater": test_greater["p"],
                "group_p_less": test_less["p"],
                "per_session": results_axis,
            }
            
            with open(summary_dir / "summary_axis_shuffle.json", "w") as f:
                json.dump(summary_axis, f, indent=2)
            
            np.savez_compressed(
                summary_dir / "results_axis_shuffle.npz",
                sids=np.array([r["sid"] for r in results_axis]),
                deltas=deltas,
                a_obs=np.array([r["a_obs"] for r in results_axis]),
                null_mean=np.array([r["null_mean"] for r in results_axis]),
            )
            
            # Plot
            plot_axis_shuffle_summary(results_axis, figs_dir, args.tag)
        else:
            print("[warn] No axis-shuffle results found")
    
    # Process axis-covariance results (DAVE'S QUESTION)
    if args.mode in ["axis_cov", "all"]:
        print(f"\n{'='*70}")
        print("[Analysis C] AXIS-COVARIANCE RESULTS (DAVE'S QUESTION)")
        print("="*70)
        print("This is the proper test: 'Are trained axes more orthogonal than")
        print("expected given the neural manifold geometry?'")
        
        results_cov = load_session_results(results_dir, "axis_cov")
        
        if results_cov:
            print(f"[info] Loaded {len(results_cov)} sessions")
            
            deltas = np.array([r["delta"] for r in results_cov])
            z_scores = np.array([r["z_score"] for r in results_cov])
            
            # Group-level test: are axes less aligned than covariance-constrained chance?
            test_less = signflip_exact_pvalue_group(deltas, "less")
            test_greater = signflip_exact_pvalue_group(deltas, "greater")
            
            print(f"\n  Mean |cos(θ)|_obs: {np.mean([r['a_obs'] for r in results_cov]):.4f}")
            print(f"  Mean |cos(θ)|_null: {np.mean([r['null_mean'] for r in results_cov]):.4f}")
            print(f"  Mean Δ: {np.mean(deltas):.4f} ± {np.std(deltas)/np.sqrt(len(deltas)):.4f}")
            print(f"  Mean z-score: {np.mean(z_scores):.2f}")
            print(f"\n  Group-level tests (sign-flip on Δ):")
            print(f"    p(more orthogonal than geometry-constrained chance, Δ<0): {test_less['p']:.4f}")
            print(f"    p(more aligned than geometry-constrained chance, Δ>0): {test_greater['p']:.4f}")
            
            # Interpretation
            if test_less['p'] < 0.05:
                print(f"\n  >>> SIGNIFICANT: Category and saccade AXES are MORE ORTHOGONAL")
                print(f"      than expected given the neural manifold constraints!")
                print(f"      This supports Dave's claim that C and S live in separate subspaces.")
            elif test_greater['p'] < 0.05:
                print(f"\n  >>> SIGNIFICANT: Category and saccade AXES are MORE ALIGNED")
                print(f"      than expected given the neural manifold constraints.")
            else:
                print(f"\n  >>> NOT SIGNIFICANT: Axis alignment is consistent with")
                print(f"      geometry-constrained chance.")
            
            # Report angle statistics
            theta_obs_arr = np.array([r["theta_obs_deg"] for r in results_cov])
            theta_null_arr = np.array([r["null_angle_mean_deg"] for r in results_cov])
            print(f"\n  Mean angle observed: {np.mean(theta_obs_arr):.1f}°")
            print(f"  Mean angle null: {np.mean(theta_null_arr):.1f}°")
            print(f"  Δθ: {np.mean(theta_obs_arr - theta_null_arr):.1f}° (positive = more orthogonal)")
            
            # Save summary
            summary_cov = {
                "analysis": "axis_covariance",
                "description": "Axis angle with covariance-constrained null (DAVE'S QUESTION)",
                "hypothesis": "Tests whether C-S axis alignment is LESS than geometry-constrained chance",
                "interpretation": "Small p_orth means axes are more orthogonal than expected given manifold",
                "created": datetime.now().isoformat(),
                "n_sessions": len(results_cov),
                "mean_a_obs": float(np.mean([r["a_obs"] for r in results_cov])),
                "mean_null": float(np.mean([r["null_mean"] for r in results_cov])),
                "mean_delta": float(np.mean(deltas)),
                "sem_delta": float(np.std(deltas) / np.sqrt(len(deltas))),
                "mean_z_score": float(np.mean(z_scores)),
                "mean_theta_obs_deg": float(np.mean(theta_obs_arr)),
                "mean_theta_null_deg": float(np.mean(theta_null_arr)),
                "mean_delta_theta_deg": float(np.mean(theta_obs_arr - theta_null_arr)),
                "mean_D_eff": float(np.mean([r.get("D_eff", np.nan) for r in results_cov])),
                "group_p_orth": test_less["p"],  # More orthogonal than chance
                "group_p_align": test_greater["p"],  # More aligned than chance
                "per_session": results_cov,
            }
            
            with open(summary_dir / "summary_axis_covariance.json", "w") as f:
                json.dump(summary_cov, f, indent=2)
            
            np.savez_compressed(
                summary_dir / "results_axis_covariance.npz",
                sids=np.array([r["sid"] for r in results_cov]),
                deltas=deltas,
                z_scores=z_scores,
                a_obs=np.array([r["a_obs"] for r in results_cov]),
                null_mean=np.array([r["null_mean"] for r in results_cov]),
                theta_obs=theta_obs_arr,
                theta_null=theta_null_arr,
            )
            
            # Plot
            plot_axis_covariance_summary(results_cov, figs_dir, args.tag)
        else:
            print("[warn] No axis-covariance results found")
    
    print(f"\n{'='*70}")
    print(f"[done] Summary saved to {summary_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
