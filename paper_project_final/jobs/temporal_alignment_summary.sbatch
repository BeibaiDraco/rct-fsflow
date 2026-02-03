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

def load_results(out_root: Path, orientation: str, filter_tag: str) -> list:
    """Load all temporal alignment results for an orientation."""
    results_dir = out_root / "temporal_alignment" / f"temporal_alignment_{orientation}_{filter_tag}"
    results = []
    
    if not results_dir.exists():
        print(f"  [warn] Results dir not found: {results_dir}")
        return results
    
    for json_file in sorted(results_dir.glob("temporal_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
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
    
    # Plot each session
    all_alignments = []
    all_times = []
    
    for i, (sid, sess_results) in enumerate(sorted(session_results.items())):
        color = COLORS_BY_SESSION[i % len(COLORS_BY_SESSION)]
        
        for r in sess_results:
            times = np.array(r["window_centers_ms"])
            alignments = np.array(r["alignments"])
            
            ax.plot(times, alignments, color=color, alpha=0.7, linewidth=1.5,
                    label=f"{sid} ({r['area']})" if r == sess_results[0] else None)
            
            all_alignments.extend(alignments)
            all_times.extend(times)
    
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
    
    # Reference lines
    ax.axvline(0, color='purple', linestyle='--', linewidth=2, alpha=0.7,
               label='Saccade onset')
    ax.axhline(0.1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.3, color='gray', linestyle=':', alpha=0.5)
    
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
        stats_text += f"Max |cos(θ)|: {np.max(all_alignments):.3f}"
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
        
        for i, (sid, sess_results) in enumerate(sorted(session_results.items())):
            color = COLORS_BY_SESSION[i % len(COLORS_BY_SESSION)]
            
            for r in sess_results:
                times = np.array(r["window_centers_ms"])
                alignments = np.array(r["alignments"])
                ax.plot(times, alignments, color=color, alpha=0.5, linewidth=1)
                all_alignments.extend(alignments)
                all_times.extend(times)
        
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
        ax.text(0.02, 0.98, f"N={n_sess}, mean={mean_align:.3f}",
                transform=ax.transAxes, fontsize=9, va='top')
    
    filter_label = "Correct Trials Only" if filter_tag == "correctonly" else "All Trials"
    plt.suptitle(f"Temporal C-S Alignment Across Trial Types ({filter_label})", 
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

figs_dir = PROJECT / "out" / "axis_alignment_all_cases_figs"
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
            
            # Individual figure for this orientation
            out_path = figs_dir / f"temporal_alignment_{orient}_{filter_tag}.pdf"
            title = f"Temporal C-S Alignment: {orient.capitalize()} Trials ({filter_tag})"
            plot_temporal_alignment_curves(results, out_path, title, orient, filter_tag)
        else:
            print(f"  {orient}: No results")
    
    # Combined comparison figure
    if any(all_results.values()):
        comparison_path = figs_dir / f"temporal_alignment_comparison_{filter_tag}.pdf"
        plot_comparison_figure(all_results, comparison_path, filter_tag)

print("\n" + "=" * 70)
print("[done] Temporal alignment summary complete")
print("=" * 70)
PYTHON_SCRIPT

echo "[done] Summary complete"
