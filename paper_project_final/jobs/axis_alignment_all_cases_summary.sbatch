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
# AXIS ALIGNMENT SUMMARY - ALL SIX CASES
#
# Aggregates results from axis_alignment_all_cases.sbatch across all sessions.
# Run this AFTER all per-session jobs complete.
#
# THE SIX CASES (REORDERED FOR LOGICAL GROUPING):
#
#   SAME-ALIGNMENT (both C and S trained on sacc-aligned data):
#     Case i:   Sacc C & S, horizontal trials
#     Case ii:  Sacc C & S, pooled trials  
#     Case iii: Sacc C & S, vertical trials
#
#   CROSS-ALIGNMENT (C on stim-aligned, S on sacc-aligned):
#     Case iv:  Stim-C (horizontal) vs Sacc-S (horizontal)
#     Case v:   Stim-C (pooled) vs Sacc-S (pooled)
#     Case vi:  Stim-C (vertical) vs Sacc-S (horizontal)
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
echo "AXIS ALIGNMENT SUMMARY - ALL SIX CASES"
echo "=============================================================="

# Create summary script inline
python3 << 'PYTHON_SCRIPT'
import json
import numpy as np
from pathlib import Path
import pandas as pd

PROJECT = Path("/project/bdoiron/dracoxu/rct-fsflow/paper_project_final")

# =============================================================================
# CASE DEFINITIONS - ALL SIX CASES (REORDERED FOR LOGICAL GROUPING)
#
# i-iii: Same alignment (both C and S on sacc)
# iv-vi: Cross alignment (C on stim, S on sacc)
# =============================================================================
CASE_ORDER = ['case_i', 'case_ii', 'case_iii', 'case_iv', 'case_v', 'case_vi']
CASE_LABELS = {
    # Same alignment (sacc C & S)
    'case_i': 'Case i: Sacc C&S (horiz)',
    'case_ii': 'Case ii: Sacc C&S (pooled)',
    'case_iii': 'Case iii: Sacc C&S (vert)',
    # Cross alignment (stim-C vs sacc-S)
    'case_iv': 'Case iv: Stim-C(horiz) vs Sacc-S(horiz)',
    'case_v': 'Case v: Stim-C(pool) vs Sacc-S(pool)',
    'case_vi': 'Case vi: Stim-C(vert) vs Sacc-S(horiz)',
}
# Colors: blue shades for same-alignment, red/orange shades for cross-alignment
CASE_COLORS = {
    'case_i': '#3498db',   # blue
    'case_ii': '#2ecc71',  # green
    'case_iii': '#1abc9c', # teal
    'case_iv': '#e74c3c',  # red
    'case_v': '#9b59b6',   # purple
    'case_vi': '#f39c12',  # orange
}

# Define all cases to summarize
# NOTE: The file paths use the OLD naming (case_iii_old = new case_vi, etc.)
# We map to the NEW logical ordering here
CASES = {
    # Correct-only - SAME ALIGNMENT
    "case_i_correctonly": {"base": "out/sacc/alignment", "prefix": "axis_cov_", "desc": "Case i: Sacc C&S, horizontal (correct-only)",
                          "old_tag": "case_i_correctonly"},
    "case_ii_correctonly": {"base": "out/sacc/alignment", "prefix": "axis_cov_", "desc": "Case ii: Sacc C&S, pooled (correct-only)",
                           "old_tag": "case_ii_correctonly"},
    "case_iii_correctonly": {"base": "out/sacc/alignment", "prefix": "axis_cov_", "desc": "Case iii: Sacc C&S, vertical (correct-only)",
                            "old_tag": "case_vi_correctonly"},  # was case_vi
    # Correct-only - CROSS ALIGNMENT
    "case_iv_correctonly": {"base": "out/cross_alignment", "prefix": "cross_", "desc": "Case iv: Stim-C(horiz) vs Sacc-S(horiz) (correct-only)",
                           "old_tag": "case_v_correctonly"},   # was case_v
    "case_v_correctonly": {"base": "out/cross_alignment", "prefix": "cross_", "desc": "Case v: Stim-C(pooled) vs Sacc-S(pooled) (correct-only)",
                          "old_tag": "case_iv_correctonly"},   # was case_iv
    "case_vi_correctonly": {"base": "out/cross_alignment", "prefix": "cross_", "desc": "Case vi: Stim-C(vert) vs Sacc-S(horiz) (correct-only)",
                           "old_tag": "case_iii_correctonly"}, # was case_iii
    # All trials - SAME ALIGNMENT
    "case_i_alltrials": {"base": "out_nofilter/sacc/alignment", "prefix": "axis_cov_", "desc": "Case i: Sacc C&S, horizontal (all-trials)",
                        "old_tag": "case_i_alltrials"},
    "case_ii_alltrials": {"base": "out_nofilter/sacc/alignment", "prefix": "axis_cov_", "desc": "Case ii: Sacc C&S, pooled (all-trials)",
                         "old_tag": "case_ii_alltrials"},
    "case_iii_alltrials": {"base": "out_nofilter/sacc/alignment", "prefix": "axis_cov_", "desc": "Case iii: Sacc C&S, vertical (all-trials)",
                          "old_tag": "case_vi_alltrials"},     # was case_vi
    # All trials - CROSS ALIGNMENT
    "case_iv_alltrials": {"base": "out_nofilter/cross_alignment", "prefix": "cross_", "desc": "Case iv: Stim-C(horiz) vs Sacc-S(horiz) (all-trials)",
                         "old_tag": "case_v_alltrials"},       # was case_v
    "case_v_alltrials": {"base": "out_nofilter/cross_alignment", "prefix": "cross_", "desc": "Case v: Stim-C(pooled) vs Sacc-S(pooled) (all-trials)",
                        "old_tag": "case_iv_alltrials"},       # was case_iv
    "case_vi_alltrials": {"base": "out_nofilter/cross_alignment", "prefix": "cross_", "desc": "Case vi: Stim-C(vert) vs Sacc-S(horiz) (all-trials)",
                         "old_tag": "case_iii_alltrials"},     # was case_iii
}

# Config for figure generation: (new_tag, path_with_old_tag, prefix, label)
CASE_CONFIGS = [
    # Correct-only - SAME ALIGNMENT
    ("case_i_correctonly", "out/sacc/alignment/case_i_correctonly", "axis_cov_", "Case i: Sacc C&S (horiz)"),
    ("case_ii_correctonly", "out/sacc/alignment/case_ii_correctonly", "axis_cov_", "Case ii: Sacc C&S (pooled)"),
    ("case_iii_correctonly", "out/sacc/alignment/case_vi_correctonly", "axis_cov_", "Case iii: Sacc C&S (vert)"),  # OLD path: case_vi
    # Correct-only - CROSS ALIGNMENT
    ("case_iv_correctonly", "out/cross_alignment/case_v_correctonly", "cross_case_v_correctonly_", "Case iv: Stim-C(horiz) vs Sacc-S(horiz)"),  # OLD: case_v
    ("case_v_correctonly", "out/cross_alignment/case_iv_correctonly", "cross_case_iv_correctonly_", "Case v: Stim-C(pool) vs Sacc-S(pool)"),   # OLD: case_iv
    ("case_vi_correctonly", "out/cross_alignment/case_iii_correctonly", "cross_case_iii_correctonly_", "Case vi: Stim-C(vert) vs Sacc-S(horiz)"), # OLD: case_iii
    # All trials - SAME ALIGNMENT
    ("case_i_alltrials", "out_nofilter/sacc/alignment/case_i_alltrials", "axis_cov_", "Case i: Sacc C&S (horiz)"),
    ("case_ii_alltrials", "out_nofilter/sacc/alignment/case_ii_alltrials", "axis_cov_", "Case ii: Sacc C&S (pooled)"),
    ("case_iii_alltrials", "out_nofilter/sacc/alignment/case_vi_alltrials", "axis_cov_", "Case iii: Sacc C&S (vert)"),  # OLD: case_vi
    # All trials - CROSS ALIGNMENT
    ("case_iv_alltrials", "out_nofilter/cross_alignment/case_v_alltrials", "cross_case_v_alltrials_", "Case iv: Stim-C(horiz) vs Sacc-S(horiz)"),  # OLD: case_v
    ("case_v_alltrials", "out_nofilter/cross_alignment/case_iv_alltrials", "cross_case_iv_alltrials_", "Case v: Stim-C(pool) vs Sacc-S(pool)"),   # OLD: case_iv
    ("case_vi_alltrials", "out_nofilter/cross_alignment/case_iii_alltrials", "cross_case_iii_alltrials_", "Case vi: Stim-C(vert) vs Sacc-S(horiz)"), # OLD: case_iii
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_case_results(case_tag, case_info):
    """Load all session results for a case."""
    # Use old_tag for file path if available (for remapped cases)
    old_tag = case_info.get("old_tag", case_tag)
    results_dir = PROJECT / case_info["base"] / old_tag
    if not results_dir.exists():
        return []
    results = []
    for json_file in results_dir.glob(f"{case_info['prefix']}*.json"):
        with open(json_file) as f:
            results.append(json.load(f))
    return results

def count_total_sessions():
    """Count total sessions available (for QC reporting)."""
    # Count unique session IDs from the session list
    sessions_file = PROJECT / "sessions.txt"
    if sessions_file.exists():
        with open(sessions_file) as f:
            return len([line.strip() for line in f if line.strip() and not line.startswith("#")])
    return 23  # fallback

def summarize_case(case_tag, case_info):
    """Compute summary statistics for a case."""
    results = load_case_results(case_tag, case_info)
    if not results:
        return None
    a_obs = [r["a_obs"] for r in results if "a_obs" in r]
    p_orth = [r["p_orth"] for r in results if "p_orth" in r]
    theta_obs = [r["theta_obs_deg"] for r in results if "theta_obs_deg" in r]
    null_mean = [r["null_mean"] for r in results if "null_mean" in r]
    D_eff = [r["D_eff"] for r in results if "D_eff" in r]
    n_sessions = len(results)
    n_sig = sum(1 for p in p_orth if p < 0.05)
    
    # Compute delta (obs - null) statistics
    deltas = [r["a_obs"] - r["null_mean"] for r in results if "a_obs" in r and "null_mean" in r]
    
    return {
        "case": case_tag, "description": case_info["desc"],
        "n_sessions": n_sessions, "n_significant": n_sig,
        "pct_significant": 100.0 * n_sig / n_sessions if n_sessions > 0 else 0,
        "a_obs_mean": np.mean(a_obs) if a_obs else np.nan,
        "a_obs_std": np.std(a_obs) if a_obs else np.nan,
        "theta_obs_mean_deg": np.mean(theta_obs) if theta_obs else np.nan,
        "theta_obs_std_deg": np.std(theta_obs) if theta_obs else np.nan,
        "p_orth_median": np.median(p_orth) if p_orth else np.nan,
        "null_mean_mean": np.mean(null_mean) if null_mean else np.nan,
        "D_eff_mean": np.mean(D_eff) if D_eff else np.nan,
        "delta_mean": np.mean(deltas) if deltas else np.nan,
        "delta_std": np.std(deltas) if deltas else np.nan,
    }

def get_case_key(case_tag):
    """Extract case key (e.g., 'case_i') from full tag (e.g., 'case_i_correctonly')."""
    for ck in CASE_ORDER:
        if case_tag.startswith(ck):
            return ck
    return None

# =============================================================================
# PRINT SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS BY CASE")
print("="*80)

# Get total session count for QC reporting
total_sessions = count_total_sessions()
print(f"\nTotal sessions in dataset: {total_sessions}")
print("(Sessions passing QC with AUC >= 0.65 for both C and S axes are included)")

# Group cases for clearer presentation
print("\n" + "-"*80)
print("SAME-ALIGNMENT CASES (i-iii): Both C and S trained on sacc-aligned data")
print("-"*80)

all_summaries = []
for case_tag, case_info in CASES.items():
    summary = summarize_case(case_tag, case_info)
    if summary:
        all_summaries.append(summary)
        
        # Print group headers
        case_key = get_case_key(case_tag)
        if case_key == 'case_iv' and 'correctonly' in case_tag:
            print("\n" + "-"*80)
            print("CROSS-ALIGNMENT CASES (iv-vi): C on stim-aligned, S on sacc-aligned")
            print("-"*80)
        
        print(f"\n{summary['description']}")
        print(f"  Sessions passing QC: {summary['n_sessions']} / {total_sessions} ({100*summary['n_sessions']/total_sessions:.0f}%)")
        print(f"  |cos(θ)| obs:  {summary['a_obs_mean']:.4f} ± {summary['a_obs_std']:.4f}")
        print(f"  |cos(θ)| null: {summary['null_mean_mean']:.4f}")
        print(f"  Δ (obs-null):  {summary['delta_mean']:.4f} ± {summary['delta_std']:.4f}")
        print(f"  D_eff (manifold dim): {summary['D_eff_mean']:.1f}")
        print(f"  p_orth median: {summary['p_orth_median']:.4f}")
        print(f"  Sessions MORE ORTHOGONAL than chance (p<0.05): {summary['n_significant']} ({summary['pct_significant']:.1f}%)")
        if summary['delta_mean'] < 0:
            print(f"  --> Δ < 0: Observed is MORE ORTHOGONAL than null")
        else:
            print(f"  --> Δ > 0: Observed is MORE ALIGNED than null")
    else:
        print(f"\n[WARN] No results found for {case_tag}")

# Save summary to CSV
if all_summaries:
    df = pd.DataFrame(all_summaries)
    out_csv = PROJECT / "out" / "axis_alignment_all_cases_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    out_csv_nf = PROJECT / "out_nofilter" / "axis_alignment_all_cases_summary.csv"
    out_csv_nf.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_nf, index=False)
    print(f"[saved] {out_csv_nf}")

# =============================================================================
# FIGURE GENERATION
# =============================================================================
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE FIGURES FOR ALL SIX CASES")
print("="*80)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load all results
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

# =============================================================================
# FIGURE 1: Six Cases Histogram Summary (2x3 grid) - Using |cos(θ)|
# =============================================================================
def plot_six_cases_histogram(all_results_dict, out_path, title):
    """Create 2x3 panel showing |cos(θ)| histograms for all six cases."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, case_key in enumerate(CASE_ORDER):
        ax = axes[idx // 3, idx % 3]
        matching = [k for k in all_results_dict.keys() if k.startswith(case_key)]
        
        if not matching:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.set_title(CASE_LABELS[case_key], fontsize=10)
            ax.set_xlim(0, 1)
            continue
        
        results = all_results_dict[matching[0]]
        a_obs = np.array([r["a_obs"] for r in results])
        null_mean = np.array([r["null_mean"] for r in results])
        p_orths = [r.get("p_orth", 0.5) for r in results]
        color = CASE_COLORS[case_key]
        
        ax.hist(a_obs, bins=12, alpha=0.7, color=color, edgecolor='white', label='Observed')
        ax.axvline(np.mean(a_obs), color=color, linestyle='-', linewidth=2.5)
        ax.axvline(np.mean(null_mean), color='black', linestyle='--', linewidth=2, label='Null mean')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)  # 0 = orthogonal
        
        n_sig = sum(1 for p in p_orths if p < 0.05)
        delta = np.mean(a_obs) - np.mean(null_mean)
        stats = f"|cos|_obs={np.mean(a_obs):.3f}±{np.std(a_obs):.3f}\n"
        stats += f"|cos|_null={np.mean(null_mean):.3f}\n"
        stats += f"Δ={delta:.3f}\n"
        stats += f"Sig: {n_sig}/{len(p_orths)} (p<0.05)"
        ax.text(0.98, 0.98, stats, transform=ax.transAxes, va='top', ha='right', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.set_xlabel("|cos(θ)|", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(CASE_LABELS[case_key], fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, max(0.6, np.max(a_obs) * 1.1))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")

# =============================================================================
# FIGURE 2: Observed vs Null Comparison (scatter plot for all 6 cases)
# =============================================================================
def plot_obs_vs_null_scatter(all_results_dict, out_path, title):
    """Create scatter plot comparing observed vs null for all six cases."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_obs = []
    all_null = []
    
    for case_key in CASE_ORDER:
        matching = [k for k in all_results_dict.keys() if k.startswith(case_key)]
        if not matching:
            continue
        
        results = all_results_dict[matching[0]]
        a_obs = np.array([r["a_obs"] for r in results])
        null_mean = np.array([r["null_mean"] for r in results])
        color = CASE_COLORS[case_key]
        
        ax.scatter(null_mean, a_obs, c=color, s=60, alpha=0.7, edgecolors='k', 
                   linewidths=0.5, label=CASE_LABELS[case_key])
        
        # Plot mean as star
        ax.scatter([np.mean(null_mean)], [np.mean(a_obs)], c=color, s=200, 
                   marker='*', edgecolors='k', linewidths=1.5, zorder=10)
        
        all_obs.extend(a_obs)
        all_null.extend(null_mean)
    
    # Add diagonal line
    if all_obs and all_null:
        lim = [0, max(max(all_null), max(all_obs)) * 1.1]
        ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=2, label='y=x (no difference)')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    
    ax.set_xlabel('Null mean |cos(θ)|', fontsize=14)
    ax.set_ylabel('Observed |cos(θ)|', fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")

# =============================================================================
# FIGURE 3: |cos(θ)| Comparison Box Plot (all 6 cases side by side)
# =============================================================================
def plot_cos_comparison_boxplot(all_results_dict, out_path, title):
    """Create box plot comparing |cos(θ)| across all six cases."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    positions_obs = []
    positions_null = []
    data_obs = []
    data_null = []
    colors_list = []
    labels = []
    
    pos = 1
    for case_key in CASE_ORDER:
        matching = [k for k in all_results_dict.keys() if k.startswith(case_key)]
        if not matching:
            pos += 3
            continue
        
        results = all_results_dict[matching[0]]
        a_obs = [r["a_obs"] for r in results]
        null_mean = [r["null_mean"] for r in results]
        
        data_null.append(null_mean)
        data_obs.append(a_obs)
        positions_null.append(pos)
        positions_obs.append(pos + 1)
        colors_list.append(CASE_COLORS[case_key])
        labels.append(CASE_LABELS[case_key])
        pos += 3
    
    # Plot null boxes (gray)
    bp_null = ax.boxplot(data_null, positions=positions_null, widths=0.6, 
                         patch_artist=True, showfliers=False)
    for patch in bp_null['boxes']:
        patch.set_facecolor('lightgray')
        patch.set_alpha(0.7)
    
    # Plot observed boxes (colored)
    bp_obs = ax.boxplot(data_obs, positions=positions_obs, widths=0.6, 
                        patch_artist=True, showfliers=False)
    for patch, color in zip(bp_obs['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add scatter points
    rng = np.random.default_rng(42)
    for i, (pn, po, dn, do) in enumerate(zip(positions_null, positions_obs, data_null, data_obs)):
        jitter_n = rng.uniform(-0.15, 0.15, len(dn))
        jitter_o = rng.uniform(-0.15, 0.15, len(do))
        ax.scatter(pn + jitter_n, dn, c='gray', s=20, alpha=0.5, zorder=5)
        ax.scatter(po + jitter_o, do, c=colors_list[i], s=20, alpha=0.7, zorder=5, edgecolors='k', linewidths=0.3)
    
    # Connect paired observations
    for i, (pn, po, dn, do) in enumerate(zip(positions_null, positions_obs, data_null, data_obs)):
        for j in range(len(dn)):
            ax.plot([pn, po], [dn[j], do[j]], 'k-', alpha=0.15, linewidth=0.5)
    
    # Set x-axis labels
    tick_positions = [(pn + po) / 2 for pn, po in zip(positions_null, positions_obs)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([l.replace('Case ', '').split(':')[0] for l in labels], fontsize=11)
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, label='0 (orthogonal)')
    ax.set_ylabel('|cos(θ)|', fontsize=14)
    ax.set_xlabel('Case', fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max([max(d) for d in data_obs + data_null]) * 1.1)
    ax.legend(['Null', 'Observed'], loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")

# =============================================================================
# FIGURE 4: Summary Bar Chart (mean |cos(θ)| and significance)
# =============================================================================
def plot_summary_bar_chart(all_results_dict, out_path, title):
    """Create bar chart summarizing mean |cos(θ)| and significance for all six cases."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    case_names = []
    a_obs_means = []
    a_obs_sems = []
    null_means = []
    delta_means = []
    pct_sig = []
    colors_list = []
    
    for case_key in CASE_ORDER:
        matching = [k for k in all_results_dict.keys() if k.startswith(case_key)]
        if not matching:
            continue
        
        results = all_results_dict[matching[0]]
        a_obs = np.array([r["a_obs"] for r in results])
        null_mean = np.array([r["null_mean"] for r in results])
        p_orths = [r.get("p_orth", 0.5) for r in results]
        
        case_names.append(CASE_LABELS[case_key].split(':')[0])
        a_obs_means.append(np.mean(a_obs))
        a_obs_sems.append(np.std(a_obs) / np.sqrt(len(a_obs)))
        null_means.append(np.mean(null_mean))
        delta_means.append(np.mean(a_obs - null_mean))
        pct_sig.append(100 * sum(1 for p in p_orths if p < 0.05) / len(p_orths))
        colors_list.append(CASE_COLORS[case_key])
    
    x = np.arange(len(case_names))
    width = 0.35
    
    # Left panel: Mean |cos(θ)|
    bars1 = ax1.bar(x - width/2, null_means, width, label='Null', color='lightgray', edgecolor='k')
    bars2 = ax1.bar(x + width/2, a_obs_means, width, label='Observed', color=colors_list, 
                    edgecolor='k', yerr=a_obs_sems, capsize=3)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Mean |cos(θ)|', fontsize=12)
    ax1.set_xlabel('Case', fontsize=12)
    ax1.set_title('Mean Observed vs Null |cos(θ)|', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_names, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, max(max(a_obs_means), max(null_means)) * 1.3)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add delta labels on observed bars
    for bar, obs, null, delta in zip(bars2.patches, a_obs_means, null_means, delta_means):
        sign = '+' if delta < 0 else ''  # negative delta means more orthogonal
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'Δ={delta:.3f}', ha='center', va='bottom', fontsize=8, color='darkred')
    
    # Right panel: Percentage significant
    bars3 = ax2.bar(x, pct_sig, width=0.6, color=colors_list, edgecolor='k')
    ax2.axhline(5, color='red', linestyle='--', alpha=0.7, label='5% (chance level)')
    ax2.set_ylabel('% Sessions with p_orth < 0.05', fontsize=12)
    ax2.set_xlabel('Case', fontsize=12)
    ax2.set_title('Sessions More Orthogonal Than Chance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(case_names, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars3, pct_sig):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{pct:.0f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")

# =============================================================================
# GENERATE ALL COMBINED FIGURES
# =============================================================================
print("\n[Generating combined figures for CORRECT-ONLY trials...]")
if all_correct:
    plot_six_cases_histogram(all_correct, combined_figs_dir / "six_cases_summary_correctonly.pdf",
                             "Six Cases: |cos(θ)| Distributions (Correct Trials Only)")
    plot_obs_vs_null_scatter(all_correct, combined_figs_dir / "obs_vs_null_correctonly.pdf",
                             "Observed vs Null |cos(θ)| (Correct Trials Only)")
    plot_cos_comparison_boxplot(all_correct, combined_figs_dir / "cos_comparison_correctonly.pdf",
                                "|cos(θ)| Comparison Across Six Cases (Correct Trials Only)")
    plot_summary_bar_chart(all_correct, combined_figs_dir / "summary_bar_chart_correctonly.pdf",
                           "Summary Statistics (Correct Trials Only)")

print("\n[Generating combined figures for ALL TRIALS...]")
if all_trials:
    plot_six_cases_histogram(all_trials, combined_figs_dir / "six_cases_summary_alltrials.pdf",
                             "Six Cases: |cos(θ)| Distributions (All Trials)")
    plot_obs_vs_null_scatter(all_trials, combined_figs_dir / "obs_vs_null_alltrials.pdf",
                             "Observed vs Null |cos(θ)| (All Trials)")
    plot_cos_comparison_boxplot(all_trials, combined_figs_dir / "cos_comparison_alltrials.pdf",
                                "|cos(θ)| Comparison Across Six Cases (All Trials)")
    plot_summary_bar_chart(all_trials, combined_figs_dir / "summary_bar_chart_alltrials.pdf",
                           "Summary Statistics (All Trials)")

# =============================================================================
# GENERATE PER-CASE FIGURES
# =============================================================================
print("\n[Generating per-case figures...]")

def plot_per_case_summary(results, out_dir, tag, case_label):
    """Create detailed summary plots for a single case using |cos(θ)|."""
    if not results:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    M_results = [r for r in results if r.get("monkey") == "M"]
    S_results = [r for r in results if r.get("monkey") == "S"]
    
    for monkey, monkey_results in [("M", M_results), ("S", S_results), ("all", results)]:
        if not monkey_results:
            continue
        
        a_obs_arr = np.array([r["a_obs"] for r in monkey_results])
        null_mean_arr = np.array([r["null_mean"] for r in monkey_results])
        delta_arr = a_obs_arr - null_mean_arr
        z_scores = np.array([r.get("z_score", 0) for r in monkey_results])
        p_orths = np.array([r.get("p_orth", 0.5) for r in monkey_results])
        
        # Figure 1: Scatter plot (obs vs null)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(null_mean_arr, a_obs_arr, c='blue', s=80, alpha=0.7, edgecolors='k')
        lim = [0, max(max(null_mean_arr), max(a_obs_arr)) * 1.1]
        ax.plot(lim, lim, 'k--', alpha=0.5, label='y=x')
        ax.scatter([np.mean(null_mean_arr)], [np.mean(a_obs_arr)], c='red', s=200, marker='*', 
                   edgecolors='darkred', linewidths=2, zorder=10, label='Mean')
        ax.set_xlabel('Null mean |cos(θ)|', fontsize=14)
        ax.set_ylabel('Observed |cos(θ)|', fontsize=14)
        n_sig = sum(p_orths < 0.05)
        ax.set_title(f'{case_label} - Monkey {monkey}\nN={len(monkey_results)}, Δ={np.mean(delta_arr):.3f}, Sig={n_sig}/{len(monkey_results)}', fontsize=10)
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        plt.tight_layout()
        fig.savefig(out_dir / f"cos_scatter_{monkey}_{tag}.png", dpi=300)
        fig.savefig(out_dir / f"cos_scatter_{monkey}_{tag}.pdf")
        plt.close(fig)
        
        # Figure 2: |cos(θ)| boxplot (null vs observed)
        fig, ax = plt.subplots(figsize=(5, 6))
        bp = ax.boxplot([null_mean_arr, a_obs_arr], positions=[1, 2], widths=0.5, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgray')
        bp['boxes'][1].set_facecolor('lightgreen')
        rng = np.random.default_rng(42)
        for pos, data in [(1, null_mean_arr), (2, a_obs_arr)]:
            x = pos + rng.uniform(-0.1, 0.1, len(data))
            ax.scatter(x, data, c='k', s=30, alpha=0.5, zorder=5)
        for i in range(len(null_mean_arr)):
            ax.plot([1, 2], [null_mean_arr[i], a_obs_arr[i]], 'k-', alpha=0.2)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Null', 'Observed'], fontsize=12)
        ax.set_ylabel('|cos(θ)|', fontsize=14)
        ax.set_title(f'{case_label} - Monkey {monkey}\nΔ|cos| = {np.mean(delta_arr):.3f}', fontsize=10)
        ax.set_ylim(0, max(max(null_mean_arr), max(a_obs_arr)) * 1.15)
        plt.tight_layout()
        fig.savefig(out_dir / f"cos_boxplot_{monkey}_{tag}.png", dpi=300)
        fig.savefig(out_dir / f"cos_boxplot_{monkey}_{tag}.pdf")
        plt.close(fig)
        
        # Figure 3: Delta histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(delta_arr, bins=15, color='seagreen', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--', linewidth=2, label='Δ=0')
        ax.axvline(np.mean(delta_arr), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean Δ={np.mean(delta_arr):.3f}')
        ax.set_xlabel('Δ = |cos(θ)|_obs - |cos(θ)|_null', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{case_label} - Monkey {monkey}', fontsize=10)
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / f"delta_hist_{monkey}_{tag}.png", dpi=300)
        fig.savefig(out_dir / f"delta_hist_{monkey}_{tag}.pdf")
        plt.close(fig)
        
        # Figure 4: Z-score histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(z_scores, bins=15, color='steelblue', edgecolor='k', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--', linewidth=2, label='z=0')
        ax.axvline(np.mean(z_scores), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean z={np.mean(z_scores):.2f}')
        ax.set_xlabel('z-score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{case_label} - Monkey {monkey}', fontsize=10)
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / f"zscore_hist_{monkey}_{tag}.png", dpi=300)
        fig.savefig(out_dir / f"zscore_hist_{monkey}_{tag}.pdf")
        plt.close(fig)
    
    print(f"[ok] Saved per-case plots to {out_dir}")

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
        plot_per_case_summary(results, figs_dir, case_tag, case_label)

print("\n" + "="*80)
print("DONE - ALL SIX CASES PROCESSED")
print("="*80)
PYTHON_SCRIPT

echo "[done] Summary complete"
