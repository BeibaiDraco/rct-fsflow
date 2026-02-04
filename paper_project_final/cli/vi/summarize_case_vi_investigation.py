#!/usr/bin/env python3
"""
Summarize Case vi Investigation Results

This script aggregates results from the three analyses:
1. Case vi driver analysis
2. Reverse case (Stim-C(horiz) vs Sacc-S(vert))
3. Temporal cross-alignment

And creates a comprehensive comparison figure and report.
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
from scipy import stats


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_monkey(sid: str) -> str:
    return "M" if sid.startswith("2020") else "S"


def get_fef_area(sid: str) -> str:
    return "MFEF" if get_monkey(sid) == "M" else "SFEF"


def load_case_results(results_dir: Path, prefix: str) -> List[Dict]:
    """Load all session results from a directory."""
    results = []
    for json_file in sorted(results_dir.glob(f"{prefix}*.json")):
        data = load_json(json_file)
        if data:
            results.append(data)
    return results


def main():
    parser = argparse.ArgumentParser(description="Summarize Case vi investigation")
    parser.add_argument("--project", default="/project/bdoiron/dracoxu/rct-fsflow/paper_project_final")
    parser.add_argument("--trial_mode", default="correctonly", choices=["correctonly", "alltrials"],
                        help="Trial filter mode: correctonly or alltrials")
    args = parser.parse_args()
    
    project = Path(args.project)
    trial_mode = args.trial_mode
    
    # Determine paths based on trial mode
    suffix = "correctonly" if trial_mode == "correctonly" else "alltrials"
    out_root = "out" if trial_mode == "correctonly" else "out_nofilter"
    
    print("=" * 70)
    print(f"CASE VI INVESTIGATION SUMMARY ({trial_mode})")
    print("=" * 70)
    
    # Load case vi driver analysis
    driver_analysis = load_json(project / "out" / "vi" / f"case_vi_analysis_{trial_mode}" / "case_vi_analysis.json")
    
    # Load reverse case results
    reverse_results = load_case_results(
        project / "out" / "vi" / f"reverse_case_{suffix}",
        "reverse_case_"
    )
    
    # Load case vi results (original)
    case_vi_results = load_case_results(
        project / out_root / "cross_alignment" / f"case_iii_{suffix}",  # file system name
        f"cross_case_iii_{suffix}_"
    )
    
    # Load temporal cross-alignment results
    temporal_results = load_case_results(
        project / "out" / "vi" / f"temporal_cross_alignment_{trial_mode}" / "case_vi_temporal",
        "case_vi_temporal_"
    )
    
    print(f"\nLoaded results:")
    print(f"  Case vi driver analysis: {'Yes' if driver_analysis else 'No'}")
    print(f"  Case vi original: {len(case_vi_results)} sessions")
    print(f"  Reverse case: {len(reverse_results)} sessions")
    print(f"  Temporal cross-alignment: {len(temporal_results)} sessions")
    
    if not case_vi_results:
        print("[ERROR] No Case vi results found!")
        return
    
    # Build session-matched comparison
    print("\n" + "=" * 70)
    print("CASE VI vs REVERSE CASE COMPARISON")
    print("=" * 70)
    
    case_vi_by_sid = {r["sid"]: r for r in case_vi_results}
    reverse_by_sid = {r["sid"]: r for r in reverse_results}
    temporal_by_sid = {r["sid"]: r for r in temporal_results}
    
    matched_sids = set(case_vi_by_sid.keys()) & set(reverse_by_sid.keys())
    
    if matched_sids:
        print(f"\nMatched sessions: {len(matched_sids)}")
        
        comparison_data = []
        
        print("\n{:12s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
            "Session", "VI_a_obs", "Rev_a_obs", "VI_null", "Rev_null", "VI-Rev"
        ))
        print("-" * 65)
        
        for sid in sorted(matched_sids):
            vi = case_vi_by_sid[sid]
            rev = reverse_by_sid[sid]
            
            diff = vi["a_obs"] - rev["a_obs"]
            
            print("{:12s} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:+10.3f}".format(
                sid,
                vi["a_obs"],
                rev["a_obs"],
                vi["null_mean"],
                rev["null_mean"],
                diff
            ))
            
            comparison_data.append({
                "sid": sid,
                "vi_a_obs": vi["a_obs"],
                "rev_a_obs": rev["a_obs"],
                "vi_null": vi["null_mean"],
                "rev_null": rev["null_mean"],
                "vi_D_eff": vi.get("D_eff", np.nan),
                "rev_D_eff": rev.get("D_eff", np.nan),
            })
        
        # Statistics
        vi_a_obs = np.array([d["vi_a_obs"] for d in comparison_data])
        rev_a_obs = np.array([d["rev_a_obs"] for d in comparison_data])
        
        print("-" * 65)
        print("{:12s} {:10.3f} {:10.3f}".format(
            "Mean",
            np.nanmean(vi_a_obs),
            np.nanmean(rev_a_obs),
        ))
        print("{:12s} {:10.3f} {:10.3f}".format(
            "Std",
            np.nanstd(vi_a_obs),
            np.nanstd(rev_a_obs),
        ))
        
        # Paired t-test
        t_stat, t_p = stats.ttest_rel(vi_a_obs, rev_a_obs)
        print(f"\nPaired t-test (VI vs Reverse): t={t_stat:.3f}, p={t_p:.4f}")
        
        # Wilcoxon signed-rank test
        w_stat, w_p = stats.wilcoxon(vi_a_obs, rev_a_obs)
        print(f"Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")
        
        # Correlation
        r, p = stats.pearsonr(vi_a_obs, rev_a_obs)
        print(f"Correlation (VI vs Reverse): r={r:.3f}, p={p:.4f}")
    
    # Create comprehensive figure
    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE FIGURE")
    print("=" * 70)
    
    out_dir = project / "out" / "vi" / f"case_vi_analysis_{trial_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Case vi vs Reverse scatter
    ax1 = fig.add_subplot(2, 3, 1)
    if matched_sids:
        vi_a = np.array([case_vi_by_sid[sid]["a_obs"] for sid in matched_sids])
        rev_a = np.array([reverse_by_sid[sid]["a_obs"] for sid in matched_sids])
        
        ax1.scatter(vi_a, rev_a, c='purple', alpha=0.7, s=60)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
        ax1.set_xlabel('Case vi |cos(θ)| (C:vert-stim, S:horiz-sacc)')
        ax1.set_ylabel('Reverse |cos(θ)| (C:horiz-stim, S:vert-sacc)')
        ax1.set_title(f'Case vi vs Reverse Case\nr={r:.3f}, paired-t p={t_p:.4f}')
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    else:
        ax1.text(0.5, 0.5, 'No matched data', ha='center', va='center')
    
    # Panel B: Box plot comparison
    ax2 = fig.add_subplot(2, 3, 2)
    
    box_data = []
    box_labels = []
    box_colors = []
    
    if case_vi_results:
        box_data.append([r["a_obs"] for r in case_vi_results])
        box_labels.append('Case vi\n(C:vert, S:horiz)')
        box_colors.append('#f39c12')
    
    if reverse_results:
        box_data.append([r["a_obs"] for r in reverse_results])
        box_labels.append('Reverse\n(C:horiz, S:vert)')
        box_colors.append('#9b59b6')
    
    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('|cos(θ)| observed')
        ax2.set_title('Observed Alignment: Case vi vs Reverse')
    
    # Panel C: Null distributions
    ax3 = fig.add_subplot(2, 3, 3)
    if case_vi_results:
        vi_null = [r["null_mean"] for r in case_vi_results]
        ax3.hist(vi_null, bins=15, alpha=0.6, color='#f39c12', label='Case vi null')
    if reverse_results:
        rev_null = [r["null_mean"] for r in reverse_results]
        ax3.hist(rev_null, bins=15, alpha=0.6, color='#9b59b6', label='Reverse null')
    ax3.set_xlabel('Null mean')
    ax3.set_ylabel('Count')
    ax3.set_title('Null Distribution Comparison')
    ax3.legend()
    
    # Panel D: Temporal cross-alignment (if available)
    ax4 = fig.add_subplot(2, 3, 4)
    if temporal_results:
        for result in temporal_results:
            t = np.array(result["window_centers_ms"])
            a = np.array(result["alignments"])
            ax4.plot(t, a, alpha=0.3, linewidth=1)
        
        # Mean
        all_alignments = np.array([r["alignments"] for r in temporal_results])
        mean_align = np.nanmean(all_alignments, axis=0)
        common_t = np.array(temporal_results[0]["window_centers_ms"])
        ax4.plot(common_t, mean_align, 'k-', linewidth=3, label='Mean')
        
        # Null
        null_means = [r["null_mean"] for r in temporal_results]
        ax4.axhline(np.mean(null_means), color='red', linestyle='--', 
                    linewidth=2, label=f'Null ({np.mean(null_means):.3f})')
        
        ax4.set_xlabel('Time from Stimulus Onset (ms)')
        ax4.set_ylabel('|cos(θ)|')
        ax4.set_title('Temporal Cross-Alignment (Case vi config)')
        ax4.legend()
        ax4.axvline(0, color='gray', linestyle=':', alpha=0.5)
    else:
        ax4.text(0.5, 0.5, 'No temporal data', ha='center', va='center')
    
    # Panel E: a_obs vs D_eff for both cases
    ax5 = fig.add_subplot(2, 3, 5)
    if case_vi_results:
        vi_a = [r["a_obs"] for r in case_vi_results]
        vi_D = [r.get("D_eff", np.nan) for r in case_vi_results]
        ax5.scatter(vi_D, vi_a, c='#f39c12', alpha=0.7, s=50, label='Case vi')
    if reverse_results:
        rev_a = [r["a_obs"] for r in reverse_results]
        rev_D = [r.get("D_eff", np.nan) for r in reverse_results]
        ax5.scatter(rev_D, rev_a, c='#9b59b6', alpha=0.7, s=50, label='Reverse')
    ax5.set_xlabel('D_eff')
    ax5.set_ylabel('|cos(θ)| observed')
    ax5.set_title('Alignment vs Effective Dimensionality')
    ax5.legend()
    
    # Panel F: Delta (obs - null) comparison
    ax6 = fig.add_subplot(2, 3, 6)
    delta_data = []
    delta_labels = []
    delta_colors = []
    
    if case_vi_results:
        vi_delta = [r["a_obs"] - r["null_mean"] for r in case_vi_results]
        delta_data.append(vi_delta)
        delta_labels.append('Case vi')
        delta_colors.append('#f39c12')
    
    if reverse_results:
        rev_delta = [r["a_obs"] - r["null_mean"] for r in reverse_results]
        delta_data.append(rev_delta)
        delta_labels.append('Reverse')
        delta_colors.append('#9b59b6')
    
    if delta_data:
        bp = ax6.boxplot(delta_data, tick_labels=delta_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], delta_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax6.axhline(0, color='gray', linestyle='--')
        ax6.set_ylabel('Δ = a_obs - null_mean')
        ax6.set_title('Alignment Relative to Null')
    
    plt.tight_layout()
    fig.savefig(out_dir / "case_vi_comprehensive_summary.png", dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / "case_vi_comprehensive_summary.pdf", bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {out_dir / 'case_vi_comprehensive_summary.png'}")
    
    # Save summary JSON
    summary = {
        "n_case_vi": len(case_vi_results),
        "n_reverse": len(reverse_results),
        "n_temporal": len(temporal_results),
        "n_matched": len(matched_sids) if matched_sids else 0,
        "case_vi_stats": {
            "a_obs_mean": float(np.nanmean([r["a_obs"] for r in case_vi_results])) if case_vi_results else None,
            "a_obs_std": float(np.nanstd([r["a_obs"] for r in case_vi_results])) if case_vi_results else None,
            "null_mean_mean": float(np.nanmean([r["null_mean"] for r in case_vi_results])) if case_vi_results else None,
        },
        "reverse_stats": {
            "a_obs_mean": float(np.nanmean([r["a_obs"] for r in reverse_results])) if reverse_results else None,
            "a_obs_std": float(np.nanstd([r["a_obs"] for r in reverse_results])) if reverse_results else None,
            "null_mean_mean": float(np.nanmean([r["null_mean"] for r in reverse_results])) if reverse_results else None,
        },
        "comparison": {
            "paired_t_stat": float(t_stat) if matched_sids else None,
            "paired_t_p": float(t_p) if matched_sids else None,
            "correlation_r": float(r) if matched_sids else None,
            "correlation_p": float(p) if matched_sids else None,
        } if matched_sids else None,
    }
    
    with open(out_dir / "case_vi_investigation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved: {out_dir / 'case_vi_investigation_summary.json'}")
    
    print("\n" + "=" * 70)
    print("SUMMARY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
