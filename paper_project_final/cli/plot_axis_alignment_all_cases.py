#!/usr/bin/env python3
"""
Generate summary figures for all four axis alignment cases.

Creates comparison plots showing:
1. Observed angle vs null distribution for each case
2. Cross-case comparison of alignment
3. Correct-only vs all-trials comparison
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_case_results(results_dir: Path, prefix: str = "axis_cov_") -> List[Dict]:
    """Load all session results from a case directory."""
    results = []
    for json_file in sorted(results_dir.glob(f"{prefix}*.json")):
        with open(json_file) as f:
            results.append(json.load(f))
    return results


def load_cross_results(results_dir: Path, tag: str) -> List[Dict]:
    """Load cross-alignment results."""
    results = []
    for json_file in sorted(results_dir.glob(f"cross_{tag}_*.json")):
        with open(json_file) as f:
            results.append(json.load(f))
    return results


def plot_angle_comparison(results_by_case: Dict[str, List[Dict]], 
                          out_path: Path, 
                          title: str = "C-S Axis Angle by Case"):
    """
    Create boxplot comparing observed angles across cases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    case_names = list(results_by_case.keys())
    data = []
    null_means = []
    
    for case_name in case_names:
        results = results_by_case[case_name]
        angles = [r["theta_obs_deg"] for r in results if "theta_obs_deg" in r]
        null_mean = np.mean([r["null_angle_mean_deg"] for r in results if "null_angle_mean_deg" in r])
        data.append(angles)
        null_means.append(null_mean)
    
    positions = np.arange(len(case_names))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    # Color boxes
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(case_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add null expectation markers
    for i, null_mean in enumerate(null_means):
        ax.scatter([i], [null_mean], marker='_', s=300, c='black', zorder=5, linewidths=2)
    
    # Add individual points
    for i, angles in enumerate(data):
        jitter = np.random.uniform(-0.15, 0.15, len(angles))
        ax.scatter(positions[i] + jitter, angles, alpha=0.5, s=20, c='gray', zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace('_', '\n') for name in case_names], rotation=0)
    ax.set_ylabel("Angle between C and S axes (degrees)")
    ax.set_title(title)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5, label='Orthogonal (90°)')
    
    # Legend
    null_patch = mpatches.Patch(color='black', label='Null expectation')
    ax.legend(handles=[null_patch], loc='upper right')
    
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_pvalue_comparison(results_by_case: Dict[str, List[Dict]], 
                           out_path: Path,
                           title: str = "P-value (more orthogonal than chance)"):
    """
    Create boxplot of p_orth values across cases.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    case_names = list(results_by_case.keys())
    data = []
    
    for case_name in case_names:
        results = results_by_case[case_name]
        pvals = [r["p_orth"] for r in results if "p_orth" in r]
        data.append(pvals)
    
    positions = np.arange(len(case_names))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(case_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, pvals in enumerate(data):
        jitter = np.random.uniform(-0.15, 0.15, len(pvals))
        ax.scatter(positions[i] + jitter, pvals, alpha=0.5, s=20, c='gray', zorder=3)
    
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='p=0.5')
    
    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace('_', '\n') for name in case_names], rotation=0)
    ax.set_ylabel("p(more orthogonal than chance)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_obs_vs_null(results_by_case: Dict[str, List[Dict]], 
                     out_path: Path,
                     title: str = "Observed vs Null |cos(θ)|"):
    """
    Scatter plot of observed vs null alignment index.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    for i, (case_name, results) in enumerate(results_by_case.items()):
        obs = [r["a_obs"] for r in results if "a_obs" in r]
        null = [r["null_mean"] for r in results if "null_mean" in r]
        ax.scatter(null, obs, c=colors[i % len(colors)], 
                   marker=markers[i % len(markers)], s=50, alpha=0.7,
                   label=case_name.replace('_', ' '))
    
    # Diagonal line
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='obs = null')
    
    ax.set_xlabel("Null mean |cos(θ)|")
    ax.set_ylabel("Observed |cos(θ)|")
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_four_cases_summary(results_by_case: Dict[str, List[Dict]], 
                            out_path: Path,
                            title: str = "Four Cases Summary"):
    """
    Create a 2x2 panel figure summarizing all four cases.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    case_order = ['case_i', 'case_ii', 'case_iii', 'case_iv']
    case_titles = [
        'Case i: Sacc C&S (horizontal)',
        'Case ii: Sacc C&S (pooled)',
        'Case iii: Stim-C vs Sacc-S (vert/horiz)',
        'Case iv: Stim-C vs Sacc-S (pooled)'
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, (case_key, case_title, color) in enumerate(zip(case_order, case_titles, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Find matching case (correctonly)
        matching = [k for k in results_by_case.keys() if case_key in k and 'correctonly' in k]
        if not matching:
            ax.text(0.5, 0.5, f'No data for {case_key}', ha='center', va='center')
            ax.set_title(case_title)
            continue
        
        results = results_by_case[matching[0]]
        
        obs_angles = [r["theta_obs_deg"] for r in results if "theta_obs_deg" in r]
        null_angles = [r["null_angle_mean_deg"] for r in results if "null_angle_mean_deg" in r]
        p_orths = [r["p_orth"] for r in results if "p_orth" in r]
        
        if not obs_angles:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(case_title)
            continue
        
        # Histogram of observed angles
        ax.hist(obs_angles, bins=15, alpha=0.7, color=color, edgecolor='white', label='Observed')
        ax.axvline(np.mean(obs_angles), color=color, linestyle='-', linewidth=2)
        ax.axvline(np.mean(null_angles), color='black', linestyle='--', linewidth=2, label=f'Null mean')
        ax.axvline(90, color='gray', linestyle=':', alpha=0.7)
        
        # Stats text
        n_sig = sum(1 for p in p_orths if p < 0.05)
        stats_text = f"θ_obs = {np.mean(obs_angles):.1f}° ± {np.std(obs_angles):.1f}°\n"
        stats_text += f"θ_null = {np.mean(null_angles):.1f}°\n"
        stats_text += f"p_orth median = {np.median(p_orths):.3f}\n"
        stats_text += f"Significant: {n_sig}/{len(p_orths)}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Count")
        ax.set_title(case_title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, 100)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures for axis alignment analysis")
    parser.add_argument("--out_root", default="out", help="Output root (out or out_nofilter)")
    parser.add_argument("--trial_mode", default="both", choices=["correctonly", "alltrials", "both"])
    args = parser.parse_args()
    
    project = Path("/project/bdoiron/dracoxu/rct-fsflow/paper_project_final")
    
    # Define case locations
    CASES = {
        "correctonly": {
            "case_i_correctonly": ("out/sacc/alignment/case_i_correctonly", "axis_cov_"),
            "case_ii_correctonly": ("out/sacc/alignment/case_ii_correctonly", "axis_cov_"),
            "case_iii_correctonly": ("out/cross_alignment/case_iii_correctonly", "cross_case_iii_correctonly_"),
            "case_iv_correctonly": ("out/cross_alignment/case_iv_correctonly", "cross_case_iv_correctonly_"),
        },
        "alltrials": {
            "case_i_alltrials": ("out_nofilter/sacc/alignment/case_i_alltrials", "axis_cov_"),
            "case_ii_alltrials": ("out_nofilter/sacc/alignment/case_ii_alltrials", "axis_cov_"),
            "case_iii_alltrials": ("out_nofilter/cross_alignment/case_iii_alltrials", "cross_case_iii_alltrials_"),
            "case_iv_alltrials": ("out_nofilter/cross_alignment/case_iv_alltrials", "cross_case_iv_alltrials_"),
        }
    }
    
    # Load results
    all_results = {}
    
    for trial_mode, cases in CASES.items():
        if args.trial_mode != "both" and args.trial_mode != trial_mode:
            continue
        
        for case_name, (rel_path, prefix) in cases.items():
            results_dir = project / rel_path
            if not results_dir.exists():
                print(f"[skip] {results_dir} not found")
                continue
            
            results = load_case_results(results_dir, prefix)
            if results:
                all_results[case_name] = results
                print(f"[loaded] {case_name}: {len(results)} sessions")
    
    if not all_results:
        print("[error] No results found!")
        return
    
    # Create output directory
    figs_dir = project / "out" / "axis_alignment_all_cases_figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    print("\n[generating figures...]")
    
    # 1. Angle comparison (correctonly)
    correctonly_results = {k: v for k, v in all_results.items() if 'correctonly' in k}
    if correctonly_results:
        plot_angle_comparison(correctonly_results, 
                              figs_dir / "angle_comparison_correctonly.pdf",
                              "C-S Axis Angle by Case (Correct Trials Only)")
        
        plot_pvalue_comparison(correctonly_results,
                               figs_dir / "pvalue_comparison_correctonly.pdf",
                               "P-value Distribution (Correct Trials Only)")
        
        plot_obs_vs_null(correctonly_results,
                         figs_dir / "obs_vs_null_correctonly.pdf",
                         "Observed vs Null Alignment (Correct Trials Only)")
        
        plot_four_cases_summary(correctonly_results,
                                figs_dir / "four_cases_summary_correctonly.pdf",
                                "Four Cases Summary (Correct Trials Only)")
    
    # 2. Angle comparison (alltrials)
    alltrials_results = {k: v for k, v in all_results.items() if 'alltrials' in k}
    if alltrials_results:
        plot_angle_comparison(alltrials_results,
                              figs_dir / "angle_comparison_alltrials.pdf",
                              "C-S Axis Angle by Case (All Trials)")
        
        plot_pvalue_comparison(alltrials_results,
                               figs_dir / "pvalue_comparison_alltrials.pdf",
                               "P-value Distribution (All Trials)")
        
        plot_obs_vs_null(alltrials_results,
                         figs_dir / "obs_vs_null_alltrials.pdf",
                         "Observed vs Null Alignment (All Trials)")
        
        plot_four_cases_summary(alltrials_results,
                                figs_dir / "four_cases_summary_alltrials.pdf",
                                "Four Cases Summary (All Trials)")
    
    # 3. Combined comparison (if both modes have data)
    if correctonly_results and alltrials_results:
        plot_angle_comparison(all_results,
                              figs_dir / "angle_comparison_all.pdf",
                              "C-S Axis Angle: All Cases")
    
    print(f"\n[done] Figures saved to: {figs_dir}")


if __name__ == "__main__":
    main()
