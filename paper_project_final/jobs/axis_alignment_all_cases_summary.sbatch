#!/bin/bash
#SBATCH --job-name=align_sum
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_all_summary_%j.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_all_summary_%j.err

# =============================================================================
# AXIS ALIGNMENT SUMMARY - ALL FOUR CASES
#
# Aggregates results from axis_alignment_all_cases.sbatch across all sessions.
# Run this AFTER all per-session jobs complete.
#
# Usage:
#   sbatch jobs/axis_alignment_all_cases_summary.sbatch
# =============================================================================

set -euo pipefail

BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

export PAPER_HOME="$PROJECT"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"

echo "=============================================================="
echo "AXIS ALIGNMENT SUMMARY - ALL FOUR CASES"
echo "=============================================================="

# Create summary script inline
python3 << 'PYTHON_SCRIPT'
import json
import numpy as np
from pathlib import Path
import pandas as pd

PROJECT = Path("/project/bdoiron/dracoxu/rct-fsflow/paper_project_final")

# Define all cases to summarize
CASES = {
    # Correct-only
    "case_i_correctonly": {
        "base": "out/sacc/alignment",
        "prefix": "axis_cov_",
        "desc": "Case i: Sacc C&S, horizontal (correct-only)"
    },
    "case_ii_correctonly": {
        "base": "out/sacc/alignment",
        "prefix": "axis_cov_",
        "desc": "Case ii: Sacc C&S, pooled (correct-only)"
    },
    "case_iii_correctonly": {
        "base": "out/cross_alignment",
        "prefix": "cross_",
        "desc": "Case iii: Stim-C (vert) vs Sacc-S (horiz) (correct-only)"
    },
    "case_iv_correctonly": {
        "base": "out/cross_alignment",
        "prefix": "cross_",
        "desc": "Case iv: Stim-C (pooled) vs Sacc-S (pooled) (correct-only)"
    },
    # All trials
    "case_i_alltrials": {
        "base": "out_nofilter/sacc/alignment",
        "prefix": "axis_cov_",
        "desc": "Case i: Sacc C&S, horizontal (all-trials)"
    },
    "case_ii_alltrials": {
        "base": "out_nofilter/sacc/alignment",
        "prefix": "axis_cov_",
        "desc": "Case ii: Sacc C&S, pooled (all-trials)"
    },
    "case_iii_alltrials": {
        "base": "out_nofilter/cross_alignment",
        "prefix": "cross_",
        "desc": "Case iii: Stim-C (vert) vs Sacc-S (horiz) (all-trials)"
    },
    "case_iv_alltrials": {
        "base": "out_nofilter/cross_alignment",
        "prefix": "cross_",
        "desc": "Case iv: Stim-C (pooled) vs Sacc-S (pooled) (all-trials)"
    },
}

def load_case_results(case_tag, case_info):
    """Load all session results for a case."""
    results_dir = PROJECT / case_info["base"] / case_tag
    if not results_dir.exists():
        return []
    
    results = []
    for json_file in results_dir.glob(f"{case_info['prefix']}*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    
    return results

def summarize_case(case_tag, case_info):
    """Compute summary statistics for a case."""
    results = load_case_results(case_tag, case_info)
    
    if not results:
        return None
    
    # Extract key metrics
    a_obs = [r["a_obs"] for r in results if "a_obs" in r]
    p_orth = [r["p_orth"] for r in results if "p_orth" in r]
    theta_obs = [r["theta_obs_deg"] for r in results if "theta_obs_deg" in r]
    null_mean = [r["null_mean"] for r in results if "null_mean" in r]
    D_eff = [r["D_eff"] for r in results if "D_eff" in r]
    
    n_sessions = len(results)
    n_sig = sum(1 for p in p_orth if p < 0.05)
    
    return {
        "case": case_tag,
        "description": case_info["desc"],
        "n_sessions": n_sessions,
        "n_significant": n_sig,
        "pct_significant": 100.0 * n_sig / n_sessions if n_sessions > 0 else 0,
        "a_obs_mean": np.mean(a_obs) if a_obs else np.nan,
        "a_obs_std": np.std(a_obs) if a_obs else np.nan,
        "theta_obs_mean_deg": np.mean(theta_obs) if theta_obs else np.nan,
        "theta_obs_std_deg": np.std(theta_obs) if theta_obs else np.nan,
        "p_orth_median": np.median(p_orth) if p_orth else np.nan,
        "null_mean_mean": np.mean(null_mean) if null_mean else np.nan,
        "D_eff_mean": np.mean(D_eff) if D_eff else np.nan,
    }

print("\n" + "="*80)
print("SUMMARY STATISTICS BY CASE")
print("="*80)

all_summaries = []
for case_tag, case_info in CASES.items():
    summary = summarize_case(case_tag, case_info)
    if summary:
        all_summaries.append(summary)
        print(f"\n{summary['description']}")
        print(f"  Sessions: {summary['n_sessions']}")
        print(f"  Significant (p<0.05): {summary['n_significant']} ({summary['pct_significant']:.1f}%)")
        print(f"  |cos(θ)| obs: {summary['a_obs_mean']:.4f} ± {summary['a_obs_std']:.4f}")
        print(f"  θ obs (deg): {summary['theta_obs_mean_deg']:.1f} ± {summary['theta_obs_std_deg']:.1f}")
        print(f"  |cos(θ)| null: {summary['null_mean_mean']:.4f}")
        print(f"  D_eff: {summary['D_eff_mean']:.1f}")
        print(f"  p_orth median: {summary['p_orth_median']:.4f}")
    else:
        print(f"\n[WARN] No results found for {case_tag}")

# Save summary to CSV
if all_summaries:
    df = pd.DataFrame(all_summaries)
    out_csv = PROJECT / "out" / "axis_alignment_all_cases_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    
    # Also save to out_nofilter/
    out_csv_nf = PROJECT / "out_nofilter" / "axis_alignment_all_cases_summary.csv"
    out_csv_nf.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_nf, index=False)
    print(f"[saved] {out_csv_nf}")

print("\n" + "="*80)
print("GENERATING COMPREHENSIVE FIGURES")
print("="*80)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_axis_covariance_summary(results, out_dir, tag, case_label):
    """Create comprehensive summary plots for axis-covariance analysis."""
    if not results:
        print(f"[warn] No results to plot for {case_label}")
        return
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate by monkey
    M_results = [r for r in results if r.get("monkey") == "M"]
    S_results = [r for r in results if r.get("monkey") == "S"]
    
    for monkey, monkey_results in [("M", M_results), ("S", S_results), ("all", results)]:
        if not monkey_results:
            continue
        
        a_obs_arr = np.array([r["a_obs"] for r in monkey_results])
        null_mean_arr = np.array([r["null_mean"] for r in monkey_results])
        theta_obs_arr = np.array([r["theta_obs_deg"] for r in monkey_results])
        theta_null_arr = np.array([r.get("null_angle_mean_deg", 90 - np.degrees(np.arcsin(r["null_mean"]))) for r in monkey_results])
        delta_arr = np.array([r.get("delta", r["a_obs"] - r["null_mean"]) for r in monkey_results])
        z_scores = np.array([r.get("z_score", 0) for r in monkey_results])
        
        # Figure 1: Alignment scatter (obs vs null mean)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(null_mean_arr, a_obs_arr, c='blue', s=80, alpha=0.7, edgecolors='k')
        lim = [0, max(max(null_mean_arr), max(a_obs_arr)) * 1.1]
        ax.plot(lim, lim, 'k--', alpha=0.5, label='y=x')
        mean_obs = np.mean(a_obs_arr)
        mean_null = np.mean(null_mean_arr)
        ax.scatter([mean_null], [mean_obs], c='red', s=200, marker='*', 
                   edgecolors='darkred', linewidths=2, zorder=10, label='Mean')
        ax.set_xlabel('Null mean |cos(θ)|', fontsize=14)
        ax.set_ylabel('Observed |cos(θ)|', fontsize=14)
        ax.set_title(f'{case_label} - Monkey {monkey}\n'
                     f'N={len(monkey_results)}, Δ={np.mean(delta_arr):.3f}±{np.std(delta_arr)/np.sqrt(len(delta_arr)):.3f}',
                     fontsize=11)
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
        bp['boxes'][1].set_facecolor('lightgreen')
        rng = np.random.default_rng(42)
        for i, (pos, data) in enumerate([(1, theta_null_arr), (2, theta_obs_arr)]):
            x = pos + rng.uniform(-0.1, 0.1, len(data))
            ax.scatter(x, data, c='k', s=30, alpha=0.5, zorder=5)
        for i in range(len(theta_null_arr)):
            ax.plot([1, 2], [theta_null_arr[i], theta_obs_arr[i]], 'k-', alpha=0.2)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Null', 'Observed'], fontsize=12)
        ax.set_ylabel('Angle θ (degrees)', fontsize=14)
        ax.set_title(f'{case_label} - Monkey {monkey}\n'
                     f'Δθ = {np.mean(theta_obs_arr - theta_null_arr):.1f}°', fontsize=11)
        ax.set_ylim(0, 95)
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
        ax.set_xlabel('z-score', fontsize=14)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{case_label} - Monkey {monkey}', fontsize=11)
        ax.legend()
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
        ax.set_title(f'{case_label} - Monkey {monkey}', fontsize=11)
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / f"axis_cov_delta_hist_{monkey}_{tag}.pdf")
        fig.savefig(out_dir / f"axis_cov_delta_hist_{monkey}_{tag}.png", dpi=300)
        plt.close(fig)
    
    print(f"[ok] Saved plots to {out_dir}")

# Generate comprehensive figures for each case
CASE_CONFIGS = [
    ("case_i_correctonly", "out/sacc/alignment/case_i_correctonly", "axis_cov_", "Case i: Sacc C&S (horiz, correct)"),
    ("case_ii_correctonly", "out/sacc/alignment/case_ii_correctonly", "axis_cov_", "Case ii: Sacc C&S (pooled, correct)"),
    ("case_iii_correctonly", "out/cross_alignment/case_iii_correctonly", "cross_case_iii_correctonly_", "Case iii: Stim-C vs Sacc-S (correct)"),
    ("case_iv_correctonly", "out/cross_alignment/case_iv_correctonly", "cross_case_iv_correctonly_", "Case iv: Stim-C vs Sacc-S pooled (correct)"),
    ("case_i_alltrials", "out_nofilter/sacc/alignment/case_i_alltrials", "axis_cov_", "Case i: Sacc C&S (horiz, all)"),
    ("case_ii_alltrials", "out_nofilter/sacc/alignment/case_ii_alltrials", "axis_cov_", "Case ii: Sacc C&S (pooled, all)"),
    ("case_iii_alltrials", "out_nofilter/cross_alignment/case_iii_alltrials", "cross_case_iii_alltrials_", "Case iii: Stim-C vs Sacc-S (all)"),
    ("case_iv_alltrials", "out_nofilter/cross_alignment/case_iv_alltrials", "cross_case_iv_alltrials_", "Case iv: Stim-C vs Sacc-S pooled (all)"),
]

for case_tag, rel_path, prefix, case_label in CASE_CONFIGS:
    results_dir = PROJECT / rel_path
    if not results_dir.exists():
        print(f"[skip] {rel_path} not found")
        continue
    
    results = []
    for jf in sorted(results_dir.glob(f"{prefix}*.json")):
        with open(jf) as f:
            results.append(json.load(f))
    
    if results:
        # Determine output directory
        if "cross_alignment" in rel_path:
            if "nofilter" in rel_path:
                figs_dir = PROJECT / "out_nofilter" / "cross_alignment" / "summary" / case_tag / "figs"
            else:
                figs_dir = PROJECT / "out" / "cross_alignment" / "summary" / case_tag / "figs"
        else:
            if "nofilter" in rel_path:
                figs_dir = PROJECT / "out_nofilter" / "sacc" / "summary" / f"alignment_{case_tag}" / "figs"
            else:
                figs_dir = PROJECT / "out" / "sacc" / "summary" / f"alignment_{case_tag}" / "figs"
        
        plot_axis_covariance_summary(results, figs_dir, case_tag, case_label)

# Create combined comparison figure
print("\n[generating combined comparison figures...]")

def plot_four_cases_combined(all_results_dict, out_path, title):
    """Create 2x2 panel comparing all four cases."""
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
        
        matching = [k for k in all_results_dict.keys() if case_key in k]
        if not matching:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center')
            ax.set_title(case_title)
            continue
        
        results = all_results_dict[matching[0]]
        theta_obs = [r["theta_obs_deg"] for r in results]
        theta_null = [r.get("null_angle_mean_deg", 80) for r in results]
        p_orths = [r.get("p_orth", 0.5) for r in results]
        
        ax.hist(theta_obs, bins=12, alpha=0.7, color=color, edgecolor='white', label='Observed')
        ax.axvline(np.mean(theta_obs), color=color, linestyle='-', linewidth=2)
        ax.axvline(np.mean(theta_null), color='black', linestyle='--', linewidth=2, label='Null mean')
        ax.axvline(90, color='gray', linestyle=':', alpha=0.5)
        
        n_sig = sum(1 for p in p_orths if p < 0.05)
        stats = f"θ_obs={np.mean(theta_obs):.1f}°±{np.std(theta_obs):.1f}°\n"
        stats += f"θ_null={np.mean(theta_null):.1f}°\n"
        stats += f"p_orth={np.median(p_orths):.3f}\n"
        stats += f"Sig: {n_sig}/{len(p_orths)}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        
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

# Load all results for combined figures
all_correct = {}
all_trials = {}
for case_tag, rel_path, prefix, case_label in CASE_CONFIGS:
    results_dir = PROJECT / rel_path
    if not results_dir.exists():
        continue
    results = []
    for jf in sorted(results_dir.glob(f"{prefix}*.json")):
        with open(jf) as f:
            results.append(json.load(f))
    if results:
        if 'correctonly' in case_tag:
            all_correct[case_tag] = results
        else:
            all_trials[case_tag] = results

combined_figs_dir = PROJECT / "out" / "axis_alignment_all_cases_figs"
combined_figs_dir.mkdir(parents=True, exist_ok=True)

if all_correct:
    plot_four_cases_combined(all_correct, combined_figs_dir / "four_cases_summary_correctonly.pdf", 
                             "Four Cases Summary (Correct Trials Only)")
if all_trials:
    plot_four_cases_combined(all_trials, combined_figs_dir / "four_cases_summary_alltrials.pdf",
                             "Four Cases Summary (All Trials)")

print("\n" + "="*80)
print("DONE")
print("="*80)
PYTHON_SCRIPT

echo "[done] Summary complete"
