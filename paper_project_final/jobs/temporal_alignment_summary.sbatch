#!/bin/bash
#SBATCH --job-name=temp_align_sum
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/temp_align_sum_%j.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/temp_align_sum_%j.err

# =============================================================================
# TEMPORAL ALIGNMENT SUMMARY
#
# Aggregates temporal alignment results across all sessions and generates
# summary figures showing how C-S alignment changes over time.
#
# Creates 3 figures (for horizontal, pooled, vertical trials):
#   - Each figure shows alignment curves for all sessions
#   - Creates both correctonly and alltrials versions
#
# Run this AFTER temporal_alignment.sbatch completes.
#
# Usage:
#   sbatch jobs/temporal_alignment_summary.sbatch
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
echo "TEMPORAL ALIGNMENT SUMMARY"
echo "=============================================================="

python3 << 'PYTHON_SCRIPT'
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

PROJECT = Path("/project/bdoiron/dracoxu/rct-fsflow/paper_project_final")

# Color schemes
COLORS_BY_SESSION = plt.cm.tab20(np.linspace(0, 1, 24))
ORIENTATION_COLORS = {
    "horizontal": "#3498db",  # blue
    "pooled": "#2ecc71",      # green
    "vertical": "#9b59b6",    # purple
}

def is_fef_area(area: str) -> bool:
    """Check if area is FEF (not LIP or other)."""
    return area in ["MFEF", "SFEF", "FEF"]


def load_results(out_root: Path, orientation: str, filter_tag: str, fef_only: bool = True) -> list:
    """Load all temporal alignment results for an orientation.
    
    Parameters:
    -----------
    out_root : Path
        Root directory for results
    orientation : str
        Trial orientation (horizontal, pooled, vertical)
    filter_tag : str
        Filter tag (correctonly, alltrials)
    fef_only : bool
        If True, only include FEF results (exclude LIP, SC, etc.)
    """
    results_dir = out_root / "temporal_alignment" / f"temporal_alignment_{orientation}_{filter_tag}"
    results = []
    
    if not results_dir.exists():
        print(f"  [warn] Results dir not found: {results_dir}")
        return results
    
    for json_file in sorted(results_dir.glob("temporal_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Filter to FEF only if requested
            if fef_only and not is_fef_area(data.get("area", "")):
                continue
                
            results.append(data)
        except Exception as e:
            print(f"  [warn] Failed to load {json_file}: {e}")
    
    return results


def plot_temporal_alignment_curves(results: list, out_path: Path, title: str,
                                   orientation: str, filter_tag: str):
    """
    Create figure showing temporal alignment curves for all sessions.
    
    Each session gets its own curve showing |cos(θ)| vs time (ms from saccade).
    """
    if not results:
        print(f"  [skip] No results for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group results by session
    session_results = defaultdict(list)
    for r in results:
        session_results[r["sid"]].append(r)
    
    # Plot each session and collect null values
    all_alignments = []
    all_times = []
    null_means = []
    null_stds = []
    D_effs = []
    
    for i, (sid, sess_results) in enumerate(sorted(session_results.items())):
        color = COLORS_BY_SESSION[i % len(COLORS_BY_SESSION)]
        
        for r in sess_results:
            times = np.array(r["window_centers_ms"])
            alignments = np.array(r["alignments"])
            
            ax.plot(times, alignments, color=color, alpha=0.7, linewidth=1.5,
                    label=f"{sid}" if r == sess_results[0] else None)
            
            all_alignments.extend(alignments)
            all_times.extend(times)
            
            # Collect null statistics if available
            if "null_mean" in r and r["null_mean"] is not None:
                null_means.append(r["null_mean"])
            if "null_std" in r and r["null_std"] is not None:
                null_stds.append(r["null_std"])
            if "D_eff" in r and r["D_eff"] is not None:
                D_effs.append(r["D_eff"])
    
    # Add mean curve
    if all_times:
        # Bin by time and compute mean
        time_bins = np.arange(-350, 201, 20)
        mean_alignments = []
        mean_times = []
        
        for t in time_bins:
            mask = (np.array(all_times) >= t - 10) & (np.array(all_times) < t + 10)
            if np.any(mask):
                mean_alignments.append(np.mean(np.array(all_alignments)[mask]))
                mean_times.append(t)
        
        if mean_times:
            ax.plot(mean_times, mean_alignments, 'k-', linewidth=3, alpha=0.9,
                    label='Mean', zorder=100)
    
    # Add null reference line (mean across sessions)
    if null_means:
        null_mean_avg = np.mean(null_means)
        null_std_avg = np.mean(null_stds) if null_stds else 0
        D_eff_avg = np.mean(D_effs) if D_effs else 0
        
        # Plot null mean as horizontal line
        ax.axhline(null_mean_avg, color='red', linestyle='-', linewidth=2, alpha=0.8,
                   label=f'Null mean ({null_mean_avg:.3f})')
        # Plot null ± std as shaded band
        if null_std_avg > 0:
            ax.axhspan(null_mean_avg - null_std_avg, null_mean_avg + null_std_avg,
                       color='red', alpha=0.15, label=f'Null ± std')
    
    # Reference lines
    ax.axvline(0, color='purple', linestyle='--', linewidth=2, alpha=0.7,
               label='Saccade onset')
    
    # Labels
    ax.set_xlabel('Time from Saccade Onset (ms)', fontsize=14)
    ax.set_ylabel('|cos(θ)| (C-S Alignment)', fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend (outside plot)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8,
              ncol=2, framealpha=0.9)
    
    ax.set_xlim(-360, 210)
    ax.set_ylim(0, min(0.8, max(all_alignments) * 1.1) if all_alignments else 0.5)
    ax.grid(True, alpha=0.3)
    
    # Stats text box
    n_sessions = len(session_results)
    if all_alignments:
        stats_text = f"N sessions: {n_sessions}\n"
        stats_text += f"Mean |cos(θ)|: {np.mean(all_alignments):.3f}\n"
        stats_text += f"Max |cos(θ)|: {np.max(all_alignments):.3f}\n"
        if null_means:
            stats_text += f"Null mean: {np.mean(null_means):.3f}\n"
            stats_text += f"D_eff: {np.mean(D_effs):.1f}" if D_effs else ""
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_comparison_figure(all_results: dict, out_path: Path, filter_tag: str):
    """
    Create a 3-panel figure comparing all three orientations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    orientations = ["horizontal", "pooled", "vertical"]
    case_labels = {
        "horizontal": "Case i/iv: Horizontal Trials",
        "pooled": "Case ii/v: Pooled Trials",
        "vertical": "Case iii/vi: Vertical Trials",
    }
    
    for ax, orient in zip(axes, orientations):
        results = all_results.get(orient, [])
        if not results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(case_labels[orient])
            continue
        
        # Group by session
        session_results = defaultdict(list)
        for r in results:
            session_results[r["sid"]].append(r)
        
        all_alignments = []
        all_times = []
        null_means = []
        D_effs = []
        
        for i, (sid, sess_results) in enumerate(sorted(session_results.items())):
            color = COLORS_BY_SESSION[i % len(COLORS_BY_SESSION)]
            
            for r in sess_results:
                times = np.array(r["window_centers_ms"])
                alignments = np.array(r["alignments"])
                ax.plot(times, alignments, color=color, alpha=0.5, linewidth=1)
                all_alignments.extend(alignments)
                all_times.extend(times)
                
                # Collect null statistics
                if "null_mean" in r and r["null_mean"] is not None:
                    null_means.append(r["null_mean"])
                if "D_eff" in r and r["D_eff"] is not None:
                    D_effs.append(r["D_eff"])
        
        # Plot null line
        if null_means:
            null_mean_avg = np.mean(null_means)
            ax.axhline(null_mean_avg, color='red', linestyle='-', linewidth=2, alpha=0.7,
                       label=f'Null ({null_mean_avg:.3f})')
        
        # Mean curve
        if all_times:
            time_bins = np.arange(-350, 201, 20)
            mean_alignments = []
            mean_times = []
            
            for t in time_bins:
                mask = (np.array(all_times) >= t - 10) & (np.array(all_times) < t + 10)
                if np.any(mask):
                    mean_alignments.append(np.mean(np.array(all_alignments)[mask]))
                    mean_times.append(t)
            
            if mean_times:
                ax.plot(mean_times, mean_alignments, 
                        color=ORIENTATION_COLORS[orient], linewidth=3,
                        label='Mean', zorder=100)
        
        ax.axvline(0, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time from Saccade (ms)', fontsize=12)
        ax.set_ylabel('|cos(θ)|', fontsize=12)
        ax.set_title(case_labels[orient], fontsize=12, fontweight='bold')
        ax.set_xlim(-360, 210)
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3)
        
        n_sess = len(session_results)
        mean_align = np.mean(all_alignments) if all_alignments else 0
        null_avg = np.mean(null_means) if null_means else 0
        stats_txt = f"N={n_sess}, mean={mean_align:.3f}"
        if null_means:
            stats_txt += f"\nnull={null_avg:.3f}"
        ax.text(0.02, 0.98, stats_txt,
                transform=ax.transAxes, fontsize=9, va='top')
    
    filter_label = "Correct Trials Only" if filter_tag == "correctonly" else "All Trials"
    plt.suptitle(f"Temporal C-S Alignment (FEF) Across Trial Types ({filter_label})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING TEMPORAL ALIGNMENT FIGURES")
print("=" * 70)

# Use temporal subfolder for temporal alignment figures
figs_dir = PROJECT / "out" / "axis_alignment_all_cases_figs" / "temporal"
figs_dir.mkdir(parents=True, exist_ok=True)

for filter_tag, out_root in [("correctonly", PROJECT / "out"), 
                             ("alltrials", PROJECT / "out_nofilter")]:
    print(f"\n--- {filter_tag.upper()} ---")
    
    all_results = {}
    
    for orient in ["horizontal", "pooled", "vertical"]:
        results = load_results(out_root, orient, filter_tag)
        all_results[orient] = results
        
        if results:
            print(f"  {orient}: {len(results)} results from {len(set(r['sid'] for r in results))} sessions")
            
            # Individual figure for this orientation (FEF only)
            out_path = figs_dir / f"temporal_alignment_{orient}_{filter_tag}.pdf"
            title = f"Temporal C-S Alignment (FEF): {orient.capitalize()} Trials ({filter_tag})"
            plot_temporal_alignment_curves(results, out_path, title, orient, filter_tag)
        else:
            print(f"  {orient}: No results")
    
    # Combined comparison figure
    if any(all_results.values()):
        comparison_path = figs_dir / f"temporal_alignment_comparison_{filter_tag}.pdf"
        plot_comparison_figure(all_results, comparison_path, filter_tag)

print("\n" + "=" * 70)
print("[done] Temporal alignment summary complete")
print(f"Figures saved to: {figs_dir}")
print("=" * 70)
PYTHON_SCRIPT

echo "[done] Summary complete"
