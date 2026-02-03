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
# WINDOW TIMING ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("OPTIMAL WINDOW TIMING ANALYSIS")
print("="*80)
print("Comparing when C and S information peaks across cases")
print("(allows understanding temporal evolution of category coding)")

def load_axes_metadata(axes_path):
    """Load metadata from axes npz file."""
    try:
        d = np.load(axes_path, allow_pickle=True)
        meta_str = str(d.get('meta', '{}'))
        return json.loads(meta_str)
    except Exception as e:
        print(f"  [warn] Failed to load {axes_path}: {e}")
        return {}

def get_session_list_from_dirs():
    """Get list of session IDs by scanning axes directories."""
    # Scan sacc directories to get session list
    sacc_base = PROJECT / "out" / "sacc"
    sessions = set()
    if sacc_base.exists():
        for d in sacc_base.iterdir():
            if d.is_dir() and d.name.isdigit():
                sessions.add(d.name)
    return sorted(sessions)

def get_fef_area(sid):
    """Get FEF area name for a session."""
    return "MFEF" if sid.startswith("202") and int(sid[:4]) <= 2021 else "SFEF"

# Define axes paths for each case type
# Maps: case_key -> {feature: (align, axes_tag)}
AXES_INFO = {
    # Same-alignment cases (sacc C&S)
    'case_i': {'C': ('sacc', 'axes_peakbin_saccCS-sacc-horizontal-20mssw'),
               'S': ('sacc', 'axes_peakbin_saccCS-sacc-horizontal-20mssw')},
    'case_ii': {'C': ('sacc', 'axes_peakbin_saccCS-sacc-pooled-20mssw'),
                'S': ('sacc', 'axes_peakbin_saccCS-sacc-pooled-20mssw')},
    'case_iii': {'C': ('sacc', 'axes_peakbin_saccCS-sacc-vertical-20mssw'),
                 'S': ('sacc', 'axes_peakbin_saccCS-sacc-vertical-20mssw')},
    # Cross-alignment cases (stim-C vs sacc-S)
    'case_iv': {'C': ('stim', 'axes_peakbin_stimC-stim-horizontal-20mssw'),
                'S': ('sacc', 'axes_peakbin_saccCS-sacc-horizontal-20mssw')},
    'case_v': {'C': ('stim', 'axes_peakbin_stimC-stim-pooled-20mssw'),
               'S': ('sacc', 'axes_peakbin_saccCS-sacc-pooled-20mssw')},
    'case_vi': {'C': ('stim', 'axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain'),
                'S': ('sacc', 'axes_peakbin_saccCS-sacc-horizontal-20mssw')},
}

def get_session_reaction_time(sid, area):
    """Get mean reaction time (stim-to-sacc delay) for a session from stim-aligned cache."""
    cache_path = PROJECT / "out" / "stim" / sid / "caches" / f"area_{area}.npz"
    if not cache_path.exists():
        return None
    try:
        d = np.load(cache_path, allow_pickle=True)
        pt_ms = d['lab_PT_ms']
        # Filter valid values (not NaN, positive, reasonable range)
        valid_pt = pt_ms[np.isfinite(pt_ms) & (pt_ms > 0) & (pt_ms < 1000)]
        if len(valid_pt) > 0:
            return float(np.median(valid_pt))
    except:
        pass
    return None

window_data = []
session_rt = {}  # Store reaction times per session
sessions = get_session_list_from_dirs()
print(f"Found {len(sessions)} sessions: {sessions[:5]}..." if len(sessions) > 5 else f"Found {len(sessions)} sessions: {sessions}")

# First, collect reaction times for all sessions
for sid in sessions:
    area = get_fef_area(sid)
    rt = get_session_reaction_time(sid, area)
    if rt is not None:
        session_rt[sid] = rt

print(f"Got reaction times for {len(session_rt)} sessions")
if session_rt:
    rts = list(session_rt.values())
    print(f"  RT stats: mean={np.mean(rts):.0f}ms, median={np.median(rts):.0f}ms, range=[{np.min(rts):.0f}, {np.max(rts):.0f}]ms")

for sid in sessions:
    area = get_fef_area(sid)
    rt = session_rt.get(sid)  # May be None if not found
    
    for case_key, axes_info in AXES_INFO.items():
        for feature, (align, axes_tag) in axes_info.items():
            axes_path = PROJECT / "out" / align / sid / "axes" / axes_tag / f"axes_{area}.npz"
            if not axes_path.exists():
                continue
            
            meta = load_axes_metadata(axes_path)
            if not meta:
                continue
            
            # Extract window info based on feature
            if feature == 'C':
                win_selected = meta.get('winC_selected')
                peak_time = meta.get('winC_peak_time_ms')
            else:  # feature == 'S'
                win_selected = meta.get('winS_selected')
                peak_time = meta.get('winS_peak_time_ms')
            
            if win_selected and peak_time is not None:
                # Convert to stim-aligned timeline
                # stim-aligned: peak_time is already relative to stim onset
                # sacc-aligned: peak_time is relative to sacc onset
                #   stim_aligned_time = sacc_aligned_time + RT
                win_start_ms = win_selected[0] * 1000
                win_end_ms = win_selected[1] * 1000
                win_center_ms = (win_start_ms + win_end_ms) / 2  # center of window (relative to align event)
                
                if align == 'sacc' and rt is not None:
                    peak_time_stim_aligned = peak_time + rt
                    win_center_stim_aligned = win_center_ms + rt
                elif align == 'stim':
                    peak_time_stim_aligned = peak_time
                    win_center_stim_aligned = win_center_ms
                else:
                    peak_time_stim_aligned = np.nan
                    win_center_stim_aligned = np.nan
                
                window_data.append({
                    'sid': sid,
                    'case': case_key,
                    'feature': feature,
                    'align': align,
                    'win_start': win_start_ms,
                    'win_end': win_end_ms,
                    'win_center_ms': win_center_ms,  # center of window (relative to align event)
                    'win_center_stim_aligned_ms': win_center_stim_aligned,  # converted to stim timeline
                    'peak_time_ms': peak_time,  # original (relative to align event)
                    'peak_time_stim_aligned_ms': peak_time_stim_aligned,  # converted to stim timeline
                    'win_len_ms': win_end_ms - win_start_ms,
                    'reaction_time_ms': rt if rt else np.nan,
                })

print(f"Collected {len(window_data)} window timing entries")

if window_data:
    win_df = pd.DataFrame(window_data)
    
    print("\n--- Optimal Window Summary by Case and Feature ---")
    print("(peak_time_ms is relative to alignment event: stim onset or sacc onset)")
    
    for case_key in CASE_ORDER:
        case_data = win_df[win_df['case'] == case_key]
        if case_data.empty:
            continue
        
        print(f"\n{CASE_LABELS[case_key]}:")
        for feature in ['C', 'S']:
            feat_data = case_data[case_data['feature'] == feature]
            if feat_data.empty:
                continue
            align = feat_data['align'].iloc[0]
            peak_mean = feat_data['peak_time_ms'].mean()
            peak_std = feat_data['peak_time_ms'].std()
            win_len_mean = feat_data['win_len_ms'].mean()
            n = len(feat_data)
            print(f"  {feature} ({align}-aligned): peak = {peak_mean:.0f} ± {peak_std:.0f} ms, window_len = {win_len_mean:.0f} ms (n={n})")
    
    # Comparison: Same-align vs Cross-align for C
    print("\n--- C Axis Timing Comparison: Sacc-aligned vs Stim-aligned ---")
    print("(Negative Δ means stim-C peaks earlier relative to saccade)")
    
    comparisons = [
        ('case_i', 'case_iv', 'horizontal'),
        ('case_ii', 'case_v', 'pooled'),
        ('case_iii', 'case_vi', 'vertical'),
    ]
    
    for sacc_case, stim_case, orientation in comparisons:
        sacc_C = win_df[(win_df['case'] == sacc_case) & (win_df['feature'] == 'C')]
        stim_C = win_df[(win_df['case'] == stim_case) & (win_df['feature'] == 'C')]
        
        if not sacc_C.empty and not stim_C.empty:
            sacc_peak_orig = sacc_C['peak_time_ms'].mean()
            sacc_peak_stim = sacc_C['peak_time_stim_aligned_ms'].mean()
            stim_peak = stim_C['peak_time_stim_aligned_ms'].mean()
            mean_rt = sacc_C['reaction_time_ms'].mean()
            print(f"\n  {orientation.capitalize()} trials (mean RT={mean_rt:.0f}ms):")
            print(f"    Sacc-aligned C (Case {sacc_case[-1]}): {sacc_peak_orig:.0f} ms from saccade = {sacc_peak_stim:.0f} ms from stimulus")
            print(f"    Stim-aligned C (Case {stim_case[-2:]}): {stim_peak:.0f} ms from stimulus")
            print(f"    Δ = {stim_peak - sacc_peak_stim:.0f} ms (stim-C peaks {'earlier' if stim_peak < sacc_peak_stim else 'later'} than sacc-C)")
    
    # Save window data
    win_csv = PROJECT / "out" / "axis_alignment_window_timing.csv"
    win_df.to_csv(win_csv, index=False)
    print(f"\n[saved] Window timing data: {win_csv}")
    
    # --- Generate Window Timing Figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate mean reaction time for reference line
    mean_rt = win_df['reaction_time_ms'].dropna().mean()
    
    # Left panel: All peak times on STIM-ALIGNED timeline
    ax1 = axes[0]
    
    case_order_plot = ['case_i', 'case_ii', 'case_iii', 'case_iv', 'case_v', 'case_vi']
    x_positions = []
    x_labels = []
    
    pos = 0
    for case_key in case_order_plot:
        case_data = win_df[win_df['case'] == case_key]
        if case_data.empty:
            continue
        
        c_data = case_data[case_data['feature'] == 'C']
        s_data = case_data[case_data['feature'] == 'S']
        
        # Use stim-aligned times for everything
        if not c_data.empty:
            c_align = c_data['align'].iloc[0]
            c_peaks = c_data['peak_time_stim_aligned_ms'].dropna().values
            if len(c_peaks) > 0:
                jitter = np.random.uniform(-0.1, 0.1, len(c_peaks))
                color = '#e74c3c' if c_align == 'stim' else '#3498db'
                marker = 'o'
                ax1.scatter(pos + jitter - 0.15, c_peaks, c=color, s=40, alpha=0.6, marker=marker)
                ax1.errorbar(pos - 0.15, np.mean(c_peaks), yerr=np.std(c_peaks), fmt=marker, color=color, 
                            markersize=10, capsize=5, capthick=2, linewidth=2)
        
        if not s_data.empty:
            s_peaks = s_data['peak_time_stim_aligned_ms'].dropna().values
            if len(s_peaks) > 0:
                jitter = np.random.uniform(-0.1, 0.1, len(s_peaks))
                ax1.scatter(pos + jitter + 0.15, s_peaks, c='#2ecc71', s=40, alpha=0.6, marker='s')
                ax1.errorbar(pos + 0.15, np.mean(s_peaks), yerr=np.std(s_peaks), fmt='s', color='#2ecc71',
                            markersize=10, capsize=5, capthick=2, linewidth=2)
        
        x_positions.append(pos)
        x_labels.append(case_key.replace('case_', '').upper())
        pos += 1
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, fontsize=12)
    ax1.set_xlabel('Case', fontsize=14)
    ax1.set_ylabel('Peak Time (ms from STIMULUS onset)', fontsize=12)
    ax1.set_title('All Peak Times on Stimulus-Aligned Timeline\n(●=C axis, ■=S axis; blue=trained sacc-aligned, red=trained stim-aligned)', fontsize=10)
    ax1.axhline(0, color='orange', linestyle='-', linewidth=2, alpha=0.7, label='Stimulus onset')
    ax1.axhline(mean_rt, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Saccade onset (~{mean_rt:.0f}ms)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add separator between same-align and cross-align
    ax1.axvline(2.5, color='black', linestyle=':', alpha=0.5)
    ax1.text(1, ax1.get_ylim()[1] * 0.95, 'Same-align', ha='center', fontsize=9, style='italic')
    ax1.text(4, ax1.get_ylim()[1] * 0.95, 'Cross-align', ha='center', fontsize=9, style='italic')
    
    # Right panel: C axis comparison on STIM-ALIGNED timeline
    ax2 = axes[1]
    
    orientations = ['horizontal', 'pooled', 'vertical']
    sacc_cases = ['case_i', 'case_ii', 'case_iii']
    stim_cases = ['case_iv', 'case_v', 'case_vi']
    
    for i, (sacc_case, stim_case, ori) in enumerate(zip(sacc_cases, stim_cases, orientations)):
        # Get stim-aligned times
        sacc_C = win_df[(win_df['case'] == sacc_case) & (win_df['feature'] == 'C')]['peak_time_stim_aligned_ms'].dropna()
        stim_C = win_df[(win_df['case'] == stim_case) & (win_df['feature'] == 'C')]['peak_time_stim_aligned_ms'].dropna()
        
        if len(sacc_C) > 0:
            jitter = np.random.uniform(-0.1, 0.1, len(sacc_C))
            ax2.scatter(i + jitter - 0.15, sacc_C.values, c='#3498db', s=40, alpha=0.5)
            ax2.errorbar(i - 0.15, sacc_C.mean(), yerr=sacc_C.std(), fmt='o', color='#3498db',
                        markersize=12, capsize=5, capthick=2, linewidth=2, label='C trained sacc-aligned' if i==0 else '')
        
        if len(stim_C) > 0:
            jitter = np.random.uniform(-0.1, 0.1, len(stim_C))
            ax2.scatter(i + jitter + 0.15, stim_C.values, c='#e74c3c', s=40, alpha=0.5)
            ax2.errorbar(i + 0.15, stim_C.mean(), yerr=stim_C.std(), fmt='o', color='#e74c3c',
                        markersize=12, capsize=5, capthick=2, linewidth=2, label='C trained stim-aligned' if i==0 else '')
    
    ax2.set_xticks(range(len(orientations)))
    ax2.set_xticklabels([o.capitalize() for o in orientations], fontsize=12)
    ax2.set_xlabel('Trial Orientation', fontsize=14)
    ax2.set_ylabel('C Axis Peak Time (ms from STIMULUS onset)', fontsize=12)
    ax2.set_title('C Axis Peak Timing Comparison\n(all converted to stimulus-aligned timeline)', fontsize=10)
    ax2.axhline(0, color='orange', linestyle='-', linewidth=2, alpha=0.7, label='Stimulus onset')
    ax2.axhline(mean_rt, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Saccade onset (~{mean_rt:.0f}ms)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Renamed from window_timing_comparison to peak_time_comparison
    timing_fig_path = PROJECT / "out" / "axis_alignment_all_cases_figs" / "peak_time_comparison.pdf"
    timing_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(timing_fig_path, dpi=150, bbox_inches='tight')
    fig.savefig(timing_fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] Peak time comparison figure: {timing_fig_path}")
    
    # =========================================================================
    # NEW: Window Center Comparison Figure
    # Same format as peak time comparison, but showing window CENTER instead
    # =========================================================================
    fig_wc, axes_wc = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: All window centers on STIM-ALIGNED timeline
    ax_wc1 = axes_wc[0]
    
    pos = 0
    x_positions_wc = []
    x_labels_wc = []
    for case_key in case_order_plot:
        case_data = win_df[win_df['case'] == case_key]
        if case_data.empty:
            continue
        
        c_data = case_data[case_data['feature'] == 'C']
        s_data = case_data[case_data['feature'] == 'S']
        
        # Use stim-aligned window centers
        if not c_data.empty:
            c_align = c_data['align'].iloc[0]
            c_centers = c_data['win_center_stim_aligned_ms'].dropna().values
            if len(c_centers) > 0:
                jitter = np.random.uniform(-0.1, 0.1, len(c_centers))
                color = '#e74c3c' if c_align == 'stim' else '#3498db'
                marker = 'o'
                ax_wc1.scatter(pos + jitter - 0.15, c_centers, c=color, s=40, alpha=0.6, marker=marker)
                ax_wc1.errorbar(pos - 0.15, np.mean(c_centers), yerr=np.std(c_centers), fmt=marker, color=color, 
                            markersize=10, capsize=5, capthick=2, linewidth=2)
        
        if not s_data.empty:
            s_centers = s_data['win_center_stim_aligned_ms'].dropna().values
            if len(s_centers) > 0:
                jitter = np.random.uniform(-0.1, 0.1, len(s_centers))
                ax_wc1.scatter(pos + jitter + 0.15, s_centers, c='#2ecc71', s=40, alpha=0.6, marker='s')
                ax_wc1.errorbar(pos + 0.15, np.mean(s_centers), yerr=np.std(s_centers), fmt='s', color='#2ecc71',
                            markersize=10, capsize=5, capthick=2, linewidth=2)
        
        x_positions_wc.append(pos)
        x_labels_wc.append(case_key.replace('case_', '').upper())
        pos += 1
    
    ax_wc1.set_xticks(x_positions_wc)
    ax_wc1.set_xticklabels(x_labels_wc, fontsize=12)
    ax_wc1.set_xlabel('Case', fontsize=14)
    ax_wc1.set_ylabel('Window Center (ms from STIMULUS onset)', fontsize=12)
    ax_wc1.set_title('All Window Centers on Stimulus-Aligned Timeline\n(●=C axis, ■=S axis; blue=trained sacc-aligned, red=trained stim-aligned)', fontsize=10)
    ax_wc1.axhline(0, color='orange', linestyle='-', linewidth=2, alpha=0.7, label='Stimulus onset')
    ax_wc1.axhline(mean_rt, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Saccade onset (~{mean_rt:.0f}ms)')
    ax_wc1.legend(loc='upper left', fontsize=8)
    ax_wc1.grid(True, axis='y', alpha=0.3)
    
    # Add separator between same-align and cross-align
    ax_wc1.axvline(2.5, color='black', linestyle=':', alpha=0.5)
    ax_wc1.text(1, ax_wc1.get_ylim()[1] * 0.95, 'Same-align', ha='center', fontsize=9, style='italic')
    ax_wc1.text(4, ax_wc1.get_ylim()[1] * 0.95, 'Cross-align', ha='center', fontsize=9, style='italic')
    
    # Right panel: C axis window center comparison on STIM-ALIGNED timeline
    ax_wc2 = axes_wc[1]
    
    for i, (sacc_case, stim_case, ori) in enumerate(zip(sacc_cases, stim_cases, orientations)):
        # Get stim-aligned window centers
        sacc_C = win_df[(win_df['case'] == sacc_case) & (win_df['feature'] == 'C')]['win_center_stim_aligned_ms'].dropna()
        stim_C = win_df[(win_df['case'] == stim_case) & (win_df['feature'] == 'C')]['win_center_stim_aligned_ms'].dropna()
        
        if len(sacc_C) > 0:
            jitter = np.random.uniform(-0.1, 0.1, len(sacc_C))
            ax_wc2.scatter(i + jitter - 0.15, sacc_C.values, c='#3498db', s=40, alpha=0.5)
            ax_wc2.errorbar(i - 0.15, sacc_C.mean(), yerr=sacc_C.std(), fmt='o', color='#3498db',
                        markersize=12, capsize=5, capthick=2, linewidth=2, label='C trained sacc-aligned' if i==0 else '')
        
        if len(stim_C) > 0:
            jitter = np.random.uniform(-0.1, 0.1, len(stim_C))
            ax_wc2.scatter(i + jitter + 0.15, stim_C.values, c='#e74c3c', s=40, alpha=0.5)
            ax_wc2.errorbar(i + 0.15, stim_C.mean(), yerr=stim_C.std(), fmt='o', color='#e74c3c',
                        markersize=12, capsize=5, capthick=2, linewidth=2, label='C trained stim-aligned' if i==0 else '')
    
    ax_wc2.set_xticks(range(len(orientations)))
    ax_wc2.set_xticklabels([o.capitalize() for o in orientations], fontsize=12)
    ax_wc2.set_xlabel('Trial Orientation', fontsize=14)
    ax_wc2.set_ylabel('C Axis Window Center (ms from STIMULUS onset)', fontsize=12)
    ax_wc2.set_title('C Axis Window Center Comparison\n(all converted to stimulus-aligned timeline)', fontsize=10)
    ax_wc2.axhline(0, color='orange', linestyle='-', linewidth=2, alpha=0.7, label='Stimulus onset')
    ax_wc2.axhline(mean_rt, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Saccade onset (~{mean_rt:.0f}ms)')
    ax_wc2.legend(loc='upper left', fontsize=9)
    ax_wc2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    wc_fig_path = PROJECT / "out" / "axis_alignment_all_cases_figs" / "window_center_comparison.pdf"
    fig_wc.savefig(wc_fig_path, dpi=150, bbox_inches='tight')
    fig_wc.savefig(wc_fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_wc)
    print(f"[saved] Window center comparison figure: {wc_fig_path}")
    
    # =========================================================================
    # NEW: Window comparison in ABSOLUTE TIME (why sacc vs stim axes differ)
    # =========================================================================
    # Load per-trial RT from one session to show distribution
    example_sid = sessions[0] if sessions else None
    example_area = get_fef_area(example_sid) if example_sid else None
    cache_path = PROJECT / "out" / "stim" / example_sid / "caches" / f"area_{example_area}.npz" if example_sid else None
    
    if cache_path and cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=True)
            pt_ms = d['lab_PT_ms']
            valid_rt = pt_ms[np.isfinite(pt_ms) & (pt_ms > 100) & (pt_ms < 500)]
            
            # Use typical window from our results (pooled case)
            sacc_peak_rel = 1   # ms from saccade (Case ii)
            sacc_len = 50
            stim_peak_rel = 264  # ms from stimulus (Case v)
            stim_len = 57
            
            sacc_start_abs = valid_rt + (sacc_peak_rel - sacc_len)
            sacc_end_abs = valid_rt + sacc_peak_rel
            stim_start = stim_peak_rel - stim_len
            stim_end = stim_peak_rel
            
            # Overlap per trial
            overlap = np.maximum(0, np.minimum(sacc_end_abs, stim_end) - np.maximum(sacc_start_abs, stim_start))
            pct_no_overlap = 100 * np.sum(overlap == 0) / len(valid_rt)
            
            fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.2, 1])
            
            # Panel A: Distribution of sacc-aligned window in absolute time
            ax_a = axes2[0]
            ax_a.fill_between([stim_start, stim_end], [0, 0], [1.1, 1.1], color='red', alpha=0.4, 
                             label=f'Stim-aligned window (FIXED)\n[{stim_start:.0f}, {stim_end:.0f}] ms from stim')
            ax_a.hist(sacc_start_abs, bins=40, alpha=0.5, color='blue', density=True, 
                     label='Sacc-aligned window START\n(varies with RT per trial)', edgecolor='white')
            ax_a.hist(sacc_end_abs, bins=40, alpha=0.3, color='navy', density=True, 
                     label='Sacc-aligned window END', edgecolor='white', histtype='step', linewidth=2)
            ax_a.set_xlabel('Time from stimulus onset (ms)', fontsize=12)
            ax_a.set_ylabel('Density (trials)', fontsize=12)
            ax_a.set_title(f'Panel A: Where does training data come from? (Session {example_sid}, N={len(valid_rt)} trials)\n'
                          f'Red = same absolute time for ALL trials (stim-aligned). '
                          f'Blue = different absolute time per trial (sacc-aligned, moves with RT).', fontsize=10)
            ax_a.legend(loc='upper right', fontsize=9)
            ax_a.set_xlim(0, 450)
            ax_a.set_ylim(0, None)
            ax_a.axvline(0, color='orange', linestyle='-', linewidth=1.5, alpha=0.8)
            ax_a.text(5, ax_a.get_ylim()[1]*0.9, 'Stimulus', fontsize=9, color='orange')
            ax_a.grid(True, alpha=0.3)
            
            # Panel B: Schematic - 5 example trials (fast to slow RT)
            ax_b = axes2[1]
            sorted_rt_idx = np.argsort(valid_rt)
            idxs = sorted_rt_idx[np.linspace(0, len(valid_rt)-1, 5).astype(int)]
            
            y_centers = []
            for i, idx in enumerate(idxs):
                rt = valid_rt[idx]
                s1, e1 = sacc_start_abs[idx], sacc_end_abs[idx]
                y_row = i * 1.2
                ax_b.barh(y_row, e1 - s1, left=s1, height=0.35, color='blue', alpha=0.7, edgecolor='darkblue')
                ax_b.barh(y_row + 0.5, stim_end - stim_start, left=stim_start, height=0.35, color='red', alpha=0.7, edgecolor='darkred')
                ax_b.text(-25, y_row + 0.25, f'RT={rt:.0f}ms', fontsize=8, ha='right', va='center')
                y_centers.append(y_row + 0.25)
            
            ax_b.axvspan(stim_start, stim_end, alpha=0.1, color='red')
            ax_b.set_xlabel('Time from stimulus onset (ms)', fontsize=12)
            ax_b.set_ylabel('Example trials (sorted by RT)', fontsize=10)
            ax_b.set_yticks(y_centers, [f'RT={valid_rt[k]:.0f}ms' for k in idxs])
            ax_b.set_title(f'Panel B: Same trial, different data! Sacc window (blue) moves with RT; stim window (red) fixed. '
                          f'~{pct_no_overlap:.0f}% of trials have NO overlap.', fontsize=10)
            ax_b.set_xlim(-30, 450)
            ax_b.axvline(0, color='orange', linestyle='-', linewidth=1.5, alpha=0.8)
            ax_b.legend(['Sacc-aligned window', 'Stim-aligned window'], loc='lower right', fontsize=9)
            ax_b.grid(True, axis='x', alpha=0.3)
            
            # Takeaway box
            fig2.text(0.5, 0.02, 
                      'Takeaway: Sacc-aligned and stim-aligned training use DIFFERENT neural data per trial '
                      '(different absolute times). Same trials, but different slices of each trial → different axes.',
                      ha='center', fontsize=10, wrap=True,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.06, 1, 1])
            abs_fig_path = PROJECT / "out" / "axis_alignment_all_cases_figs" / "window_comparison_absolute_time.pdf"
            fig2.savefig(abs_fig_path, dpi=150, bbox_inches='tight')
            fig2.savefig(abs_fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"[saved] Window comparison (absolute time): {abs_fig_path}")
        except Exception as e:
            print(f"[warn] Could not create absolute-time window figure: {e}")
    
    # =========================================================================
    # NEW: All-sessions window comparison (absolute time)
    # Shows window positions for ALL sessions as horizontal bars
    # =========================================================================
    print("\n[Generating all-sessions window comparison figure...]")
    
    def load_axes_summary_json(path):
        """Load axes_summary.json if it exists."""
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    
    def load_rt_from_cache(cache_path, correct_only=True):
        """Load RT (PT_ms) from cache file."""
        if not cache_path.exists():
            return None
        try:
            cache = np.load(cache_path, allow_pickle=True)
            if "lab_PT_ms" not in cache:
                return None
            rt = cache["lab_PT_ms"].astype(float)
            if correct_only and "lab_is_correct" in cache:
                is_correct = cache["lab_is_correct"].astype(bool)
                rt = rt[is_correct]
            valid = np.isfinite(rt) & (rt > 100) & (rt < 1000)
            return rt[valid]
        except:
            return None
    
    # Gather window data for all sessions
    all_session_windows = []
    
    # Define axes tags for cross-alignment case (stim-C vs sacc-S)
    stim_tag = "axes_peakbin_stimCR-stim-vertical-20mssw"
    sacc_tag = "axes_peakbin_saccCS-sacc-horizontal-20mssw"
    
    for sid in sessions:
        # Load stim-aligned axes summary
        stim_summary_path = PROJECT / "out" / "stim" / sid / "axes" / stim_tag / "axes_summary.json"
        stim_summary = load_axes_summary_json(stim_summary_path)
        
        # Load sacc-aligned axes summary  
        sacc_summary_path = PROJECT / "out" / "sacc" / sid / "axes" / sacc_tag / "axes_summary.json"
        sacc_summary = load_axes_summary_json(sacc_summary_path)
        
        if stim_summary is None or sacc_summary is None:
            continue
        
        # Load RT from sacc cache
        area = get_fef_area(sid)
        rt = load_rt_from_cache(PROJECT / "out" / "sacc" / sid / "caches" / f"area_{area}.npz", True)
        if rt is None or len(rt) < 10:
            # Try other areas
            for alt_area in ["MFEF", "MLIP", "SFEF", "SLIP"]:
                rt = load_rt_from_cache(PROJECT / "out" / "sacc" / sid / "caches" / f"area_{alt_area}.npz", True)
                if rt is not None and len(rt) >= 10:
                    break
        
        if rt is None or len(rt) < 10:
            continue
        
        # Get stim-aligned C window
        stim_win_C = None
        per_area = stim_summary.get("per_area_results") or {}
        for area_name, area_data in per_area.items():
            if area_data and "winC_selected" in area_data:
                stim_win_C = area_data["winC_selected"]
                break
        if stim_win_C is None:
            stim_win_C = stim_summary.get("winC")
        
        # Get sacc-aligned S window
        sacc_win_S = None
        per_area = sacc_summary.get("per_area_results") or {}
        for area_name, area_data in per_area.items():
            if area_data and "winS_selected" in area_data:
                sacc_win_S = area_data["winS_selected"]
                break
        if sacc_win_S is None:
            sacc_win_S = sacc_summary.get("winS")
        
        if stim_win_C is None or sacc_win_S is None:
            continue
        
        # Convert to ms
        stim_win_C_ms = [stim_win_C[0] * 1000, stim_win_C[1] * 1000]
        sacc_win_S_ms = [sacc_win_S[0] * 1000, sacc_win_S[1] * 1000]
        
        # Compute RT statistics
        rt_median = np.median(rt)
        rt_q10, rt_q90 = np.percentile(rt, [10, 90])
        
        # Convert sacc window to absolute time
        sacc_win_abs_median = [rt_median + sacc_win_S_ms[0], rt_median + sacc_win_S_ms[1]]
        sacc_win_abs_q10 = [rt_q10 + sacc_win_S_ms[0], rt_q10 + sacc_win_S_ms[1]]
        sacc_win_abs_q90 = [rt_q90 + sacc_win_S_ms[0], rt_q90 + sacc_win_S_ms[1]]
        
        # Compute overlap fraction
        n_overlap = 0
        for rt_i in rt:
            sacc_abs_start = rt_i + sacc_win_S_ms[0]
            sacc_abs_end = rt_i + sacc_win_S_ms[1]
            if sacc_abs_start < stim_win_C_ms[1] and sacc_abs_end > stim_win_C_ms[0]:
                n_overlap += 1
        frac_overlap = n_overlap / len(rt)
        
        all_session_windows.append({
            "sid": sid,
            "stim_win_C_ms": stim_win_C_ms,
            "sacc_win_abs_median": sacc_win_abs_median,
            "sacc_win_abs_q10": sacc_win_abs_q10,
            "sacc_win_abs_q90": sacc_win_abs_q90,
            "rt_median": rt_median,
            "frac_overlap": frac_overlap,
        })
    
    if all_session_windows:
        # Sort by RT
        all_session_windows.sort(key=lambda x: x["rt_median"])
        n = len(all_session_windows)
        
        # Create detailed timeline figure
        fig3, ax3 = plt.subplots(figsize=(12, max(8, n * 0.4)))
        y_positions = np.arange(n) * 2
        
        for i, r in enumerate(all_session_windows):
            y = y_positions[i]
            
            # Stim window (red)
            stim_start, stim_end = r["stim_win_C_ms"]
            rect_stim = plt.Rectangle((stim_start, y + 0.2), stim_end - stim_start, 0.6,
                                       facecolor='red', edgecolor='darkred', alpha=0.7, linewidth=1)
            ax3.add_patch(rect_stim)
            
            # Sacc window RT range (light blue)
            sacc_q10_start = r["sacc_win_abs_q10"][0]
            sacc_q90_end = r["sacc_win_abs_q90"][1]
            rect_bg = plt.Rectangle((sacc_q10_start, y - 0.8), sacc_q90_end - sacc_q10_start, 0.6,
                                     facecolor='blue', alpha=0.15, linewidth=0)
            ax3.add_patch(rect_bg)
            
            # Sacc window at median RT (blue)
            sacc_med_start, sacc_med_end = r["sacc_win_abs_median"]
            rect_sacc = plt.Rectangle((sacc_med_start, y - 0.8), sacc_med_end - sacc_med_start, 0.6,
                                       facecolor='blue', edgecolor='darkblue', alpha=0.7, linewidth=1)
            ax3.add_patch(rect_sacc)
            
            # Overlap indicator
            overlap_pct = r["frac_overlap"] * 100
            color = 'green' if overlap_pct > 50 else ('orange' if overlap_pct > 10 else 'red')
            ax3.text(max(stim_end, sacc_med_end) + 10, y, f'{overlap_pct:.0f}%',
                     fontsize=7, color=color, va='center')
            ax3.text(-40, y, r["sid"], fontsize=7, ha='right', va='center')
        
        ax3.axvline(0, color='orange', linewidth=2, label='Stimulus onset', zorder=0)
        mean_rt = np.mean([r["rt_median"] for r in all_session_windows])
        ax3.axvline(mean_rt, color='purple', linestyle='--', linewidth=2, 
                    label=f'Mean saccade onset ({mean_rt:.0f}ms)', zorder=0)
        
        import matplotlib.patches as mpatches
        stim_patch = mpatches.Patch(color='red', alpha=0.7, label='Stim-aligned window (C)')
        sacc_patch = mpatches.Patch(color='blue', alpha=0.7, label='Sacc-aligned window (S, median RT)')
        sacc_range_patch = mpatches.Patch(color='blue', alpha=0.15, label='Sacc window RT range (10-90%ile)')
        ax3.legend(handles=[stim_patch, sacc_patch, sacc_range_patch], loc='upper right', fontsize=9)
        
        ax3.set_xlim(-60, 500)
        ax3.set_ylim(-1.5, n * 2)
        ax3.set_yticks([])
        ax3.set_xlabel("Time from Stimulus Onset (ms)")
        ax3.set_title("Window Positions on Absolute Timeline\n(% = fraction of trials with overlap)")
        ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        timeline_path = PROJECT / "out" / "axis_alignment_all_cases_figs" / "window_absolute_timeline_detailed.pdf"
        fig3.savefig(timeline_path, dpi=150, bbox_inches='tight')
        fig3.savefig(timeline_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"[saved] {timeline_path}")
        
        # Create summary figure with histogram
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: All sessions
        ax4a = axes4[0]
        for i, r in enumerate(all_session_windows):
            stim_start, stim_end = r["stim_win_C_ms"]
            ax4a.barh(i, stim_end - stim_start, left=stim_start, height=0.35,
                      color='red', alpha=0.6, edgecolor='darkred', linewidth=0.5)
            sacc_start, sacc_end = r["sacc_win_abs_median"]
            ax4a.barh(i - 0.35, sacc_end - sacc_start, left=sacc_start, height=0.35,
                      color='blue', alpha=0.6, edgecolor='darkblue', linewidth=0.5)
            # RT variability
            sacc_q10_start = r["sacc_win_abs_q10"][0]
            sacc_q90_end = r["sacc_win_abs_q90"][1]
            ax4a.plot([sacc_q10_start, sacc_q90_end], [i - 0.35, i - 0.35],
                      color='blue', alpha=0.3, linewidth=6, solid_capstyle='butt')
        
        ax4a.axvline(mean_rt, color='purple', linestyle='--', alpha=0.7, linewidth=1.5,
                     label=f'Mean saccade onset (~{mean_rt:.0f}ms)')
        ax4a.axvline(0, color='orange', linewidth=2, label='Stimulus onset')
        ax4a.set_yticks(range(n))
        ax4a.set_yticklabels([r["sid"] for r in all_session_windows], fontsize=7)
        ax4a.set_xlabel("Time from Stimulus Onset (ms)")
        ax4a.set_ylabel("Session (sorted by median RT)")
        ax4a.set_title("Window Positions in Absolute Time\n(red=stim-aligned C, blue=sacc-aligned S)")
        ax4a.legend(loc='upper right', fontsize=8)
        ax4a.grid(axis='x', alpha=0.3)
        ax4a.set_xlim(-50, 450)
        
        # Right: Overlap histogram
        ax4b = axes4[1]
        overlaps = [r["frac_overlap"] for r in all_session_windows]
        ax4b.hist(overlaps, bins=20, range=(0, 1), color='green', alpha=0.7, edgecolor='darkgreen')
        ax4b.axvline(np.mean(overlaps), color='black', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(overlaps):.1%}')
        ax4b.axvline(np.median(overlaps), color='gray', linestyle=':', linewidth=2,
                     label=f'Median: {np.median(overlaps):.1%}')
        ax4b.set_xlabel("Fraction of trials with window overlap")
        ax4b.set_ylabel("Number of sessions")
        ax4b.set_title("How often do stim & sacc windows overlap?")
        ax4b.legend()
        ax4b.set_xlim(0, 1)
        
        text = f"N sessions: {n}\n"
        text += f"Mean overlap: {np.mean(overlaps):.1%}\n"
        text += f"Sessions with >50% overlap: {sum(1 for o in overlaps if o > 0.5)}/{n}\n"
        text += f"Sessions with <10% overlap: {sum(1 for o in overlaps if o < 0.1)}/{n}"
        ax4b.text(0.95, 0.95, text, transform=ax4b.transAxes, fontsize=9,
                  va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle("Training Window Comparison Across All Sessions", fontsize=14, fontweight='bold')
        plt.tight_layout()
        summary_path = PROJECT / "out" / "axis_alignment_all_cases_figs" / "window_comparison_all_sessions.pdf"
        fig4.savefig(summary_path, dpi=150, bbox_inches='tight')
        fig4.savefig(summary_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"[saved] {summary_path}")
        
        # Print summary
        print(f"\nWindow overlap statistics:")
        print(f"  Mean:   {np.mean(overlaps):.1%}")
        print(f"  Median: {np.median(overlaps):.1%}")
        print(f"  Sessions with >50% overlap: {sum(1 for o in overlaps if o > 0.5)}/{n}")
        print(f"  Sessions with <10% overlap: {sum(1 for o in overlaps if o < 0.1)}/{n}")
    else:
        print("[warn] Could not gather session window data for all-sessions figure")
    
    # =========================================================================
    # C AXIS WINDOW COMPARISONS: Sacc-aligned C vs Stim-aligned C
    # 
    # Three comparisons to understand why same trials → different alignment:
    #   1. Case i (sacc-C horizontal) vs Case iv (stim-C horizontal)
    #   2. Case ii (sacc-C pooled) vs Case v (stim-C pooled)
    #   3. Case iii (sacc-C vertical) vs Case vi (stim-C vertical)
    # =========================================================================
    print("\n[Generating C axis window comparison figures...]")
    
    # Define the three C axis comparisons
    C_COMPARISONS = [
        {
            'name': 'horizontal',
            'sacc_case': 'case_i',
            'stim_case': 'case_iv',
            'sacc_tag': 'axes_peakbin_saccCS-sacc-horizontal-20mssw',
            'stim_tag': 'axes_peakbin_stimC-stim-horizontal-20mssw',
            'orientation': 'horizontal',
            'title': 'Case i vs Case iv: Horizontal Trials\nSacc-aligned C (blue) vs Stim-aligned C (red)',
        },
        {
            'name': 'pooled',
            'sacc_case': 'case_ii',
            'stim_case': 'case_v',
            'sacc_tag': 'axes_peakbin_saccCS-sacc-pooled-20mssw',
            'stim_tag': 'axes_peakbin_stimC-stim-pooled-20mssw',
            'orientation': 'pooled',
            'title': 'Case ii vs Case v: Pooled Trials\nSacc-aligned C (blue) vs Stim-aligned C (red)',
        },
        {
            'name': 'vertical',
            'sacc_case': 'case_iii',
            'stim_case': 'case_vi',
            'sacc_tag': 'axes_peakbin_saccCS-sacc-vertical-20mssw',
            'stim_tag': 'axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain',
            'orientation': 'vertical',
            'title': 'Case iii vs Case vi: Vertical Trials\nSacc-aligned C (blue) vs Stim-aligned C (red)',
        },
    ]
    
    def load_axes_summary_for_C(path):
        """Load axes_summary.json and extract C window info."""
        if not path.exists():
            return None
        with open(path) as f:
            summary = json.load(f)
        # Get C window from per_area_results or fallback
        win_C = None
        peak_C = None
        per_area = summary.get("per_area_results") or {}
        for area_name, area_data in per_area.items():
            if area_data and "winC_selected" in area_data:
                win_C = area_data["winC_selected"]
                peak_C = area_data.get("winC_peak_time_ms")
                break
        if win_C is None:
            win_C = summary.get("winC")
        return {"win_C": win_C, "peak_C": peak_C}
    
    def load_rt_distribution(cache_path, correct_only=True):
        """Load RT distribution from cache."""
        if not cache_path.exists():
            return None
        try:
            cache = np.load(cache_path, allow_pickle=True)
            if "lab_PT_ms" not in cache:
                return None
            rt = cache["lab_PT_ms"].astype(float)
            if correct_only and "lab_is_correct" in cache:
                is_correct = cache["lab_is_correct"].astype(bool)
                rt = rt[is_correct]
            valid = np.isfinite(rt) & (rt > 100) & (rt < 1000)
            return rt[valid]
        except:
            return None
    
    for comp in C_COMPARISONS:
        print(f"\n  Generating {comp['name']} comparison...")
        
        session_data = []
        for sid in sessions:
            area = get_fef_area(sid)
            
            # Load sacc-aligned C info
            sacc_path = PROJECT / "out" / "sacc" / sid / "axes" / comp['sacc_tag'] / "axes_summary.json"
            sacc_info = load_axes_summary_for_C(sacc_path)
            
            # Load stim-aligned C info
            stim_path = PROJECT / "out" / "stim" / sid / "axes" / comp['stim_tag'] / "axes_summary.json"
            stim_info = load_axes_summary_for_C(stim_path)
            
            if sacc_info is None or stim_info is None:
                continue
            if sacc_info["win_C"] is None or stim_info["win_C"] is None:
                continue
            
            # Load RT
            rt = None
            for alt_area in [area, "MFEF", "MLIP", "SFEF", "SLIP"]:
                cache_path = PROJECT / "out" / "sacc" / sid / "caches" / f"area_{alt_area}.npz"
                rt = load_rt_distribution(cache_path, True)
                if rt is not None and len(rt) >= 10:
                    break
            
            if rt is None or len(rt) < 10:
                continue
            
            # Convert windows to ms
            sacc_win_ms = [sacc_info["win_C"][0] * 1000, sacc_info["win_C"][1] * 1000]
            stim_win_ms = [stim_info["win_C"][0] * 1000, stim_info["win_C"][1] * 1000]
            
            # RT stats
            rt_median = np.median(rt)
            rt_q10, rt_q90 = np.percentile(rt, [10, 90])
            
            # Convert sacc C window to absolute time
            sacc_win_abs_median = [rt_median + sacc_win_ms[0], rt_median + sacc_win_ms[1]]
            sacc_win_abs_q10 = [rt_q10 + sacc_win_ms[0], rt_q10 + sacc_win_ms[1]]
            sacc_win_abs_q90 = [rt_q90 + sacc_win_ms[0], rt_q90 + sacc_win_ms[1]]
            
            # Compute overlap (C windows)
            n_overlap = 0
            for rt_i in rt:
                sacc_abs_start = rt_i + sacc_win_ms[0]
                sacc_abs_end = rt_i + sacc_win_ms[1]
                if sacc_abs_start < stim_win_ms[1] and sacc_abs_end > stim_win_ms[0]:
                    n_overlap += 1
            frac_overlap = n_overlap / len(rt)
            
            session_data.append({
                "sid": sid,
                "sacc_win_ms": sacc_win_ms,
                "stim_win_ms": stim_win_ms,
                "sacc_win_abs_median": sacc_win_abs_median,
                "sacc_win_abs_q10": sacc_win_abs_q10,
                "sacc_win_abs_q90": sacc_win_abs_q90,
                "rt_median": rt_median,
                "frac_overlap": frac_overlap,
            })
        
        if not session_data:
            print(f"    [warn] No data for {comp['name']} comparison")
            continue
        
        # Sort by RT
        session_data.sort(key=lambda x: x["rt_median"])
        n = len(session_data)
        
        # Create figure similar to window_absolute_timeline_detailed.png
        fig, ax = plt.subplots(figsize=(12, max(8, n * 0.4)))
        y_positions = np.arange(n) * 2
        
        for i, r in enumerate(session_data):
            y = y_positions[i]
            
            # Stim-aligned C window (red) - fixed position
            stim_start, stim_end = r["stim_win_ms"]
            rect_stim = plt.Rectangle((stim_start, y + 0.2), stim_end - stim_start, 0.6,
                                       facecolor='red', edgecolor='darkred', alpha=0.7, linewidth=1)
            ax.add_patch(rect_stim)
            
            # Sacc-aligned C window RT range (light blue)
            sacc_q10_start = r["sacc_win_abs_q10"][0]
            sacc_q90_end = r["sacc_win_abs_q90"][1]
            rect_bg = plt.Rectangle((sacc_q10_start, y - 0.8), sacc_q90_end - sacc_q10_start, 0.6,
                                     facecolor='blue', alpha=0.15, linewidth=0)
            ax.add_patch(rect_bg)
            
            # Sacc-aligned C window at median RT (blue)
            sacc_med_start, sacc_med_end = r["sacc_win_abs_median"]
            rect_sacc = plt.Rectangle((sacc_med_start, y - 0.8), sacc_med_end - sacc_med_start, 0.6,
                                       facecolor='blue', edgecolor='darkblue', alpha=0.7, linewidth=1)
            ax.add_patch(rect_sacc)
            
            # Overlap indicator
            overlap_pct = r["frac_overlap"] * 100
            color = 'green' if overlap_pct > 50 else ('orange' if overlap_pct > 10 else 'red')
            ax.text(max(stim_end, sacc_med_end) + 10, y, f'{overlap_pct:.0f}%',
                    fontsize=7, color=color, va='center')
            ax.text(-40, y, r["sid"], fontsize=7, ha='right', va='center')
        
        ax.axvline(0, color='orange', linewidth=2, label='Stimulus onset', zorder=0)
        mean_rt = np.mean([r["rt_median"] for r in session_data])
        ax.axvline(mean_rt, color='purple', linestyle='--', linewidth=2,
                   label=f'Mean saccade onset ({mean_rt:.0f}ms)', zorder=0)
        
        import matplotlib.patches as mpatches
        stim_patch = mpatches.Patch(color='red', alpha=0.7, label=f'Stim-aligned C ({comp["stim_case"]})')
        sacc_patch = mpatches.Patch(color='blue', alpha=0.7, label=f'Sacc-aligned C ({comp["sacc_case"]}, median RT)')
        sacc_range_patch = mpatches.Patch(color='blue', alpha=0.15, label='Sacc-C RT range (10-90%ile)')
        ax.legend(handles=[stim_patch, sacc_patch, sacc_range_patch], loc='upper right', fontsize=9)
        
        ax.set_xlim(-60, 500)
        ax.set_ylim(-1.5, n * 2)
        ax.set_yticks([])
        ax.set_xlabel("Time from Stimulus Onset (ms)")
        ax.set_title(comp['title'] + "\n(% = fraction of trials with C window overlap)")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        out_path = PROJECT / "out" / "axis_alignment_all_cases_figs" / f"C_window_comparison_{comp['name']}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    [saved] {out_path}")
        
        # Print stats
        overlaps = [r["frac_overlap"] for r in session_data]
        print(f"    {comp['name']} C window overlap: mean={np.mean(overlaps):.1%}, median={np.median(overlaps):.1%}")
        print(f"    Sessions with >50% overlap: {sum(1 for o in overlaps if o > 0.5)}/{n}")

else:
    print("[warn] No window timing data found - skipping timing analysis figure")

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
    ax.legend(loc='lower right', fontsize=9)
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

# =============================================================================
# REGENERATE WINDOW COMPARISON FIGURES FOR BOTH CORRECTONLY AND ALLTRIALS
# =============================================================================
print("\n" + "="*80)
print("GENERATING WINDOW COMPARISON FIGURES FOR BOTH FILTER MODES")
print("="*80)

def generate_window_comparison_figures_for_filter(out_root, filter_tag, figs_dir):
    """Generate all window comparison figures for a specific filter mode."""
    print(f"\n--- Generating window figures for {filter_tag} ({out_root}) ---")
    
    # Define axes tags based on filter mode
    nofilter_suffix = "_nofilter" if "nofilter" in str(out_root) else ""
    
    # C AXIS WINDOW COMPARISONS for this filter mode
    C_COMPARISONS_FILTER = [
        {
            'name': 'horizontal',
            'sacc_tag': f'axes_peakbin_saccCS-sacc-horizontal-20mssw{nofilter_suffix}',
            'stim_tag': f'axes_peakbin_stimC-stim-horizontal-20mssw{nofilter_suffix}',
            'title': f'Case i vs iv: Horizontal ({filter_tag})\nSacc-C (blue) vs Stim-C (red)',
        },
        {
            'name': 'pooled',
            'sacc_tag': f'axes_peakbin_saccCS-sacc-pooled-20mssw{nofilter_suffix}',
            'stim_tag': f'axes_peakbin_stimC-stim-pooled-20mssw{nofilter_suffix}',
            'title': f'Case ii vs v: Pooled ({filter_tag})\nSacc-C (blue) vs Stim-C (red)',
        },
        {
            'name': 'vertical',
            'sacc_tag': f'axes_peakbin_saccCS-sacc-vertical-20mssw{nofilter_suffix}',
            'stim_tag': f'axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain{nofilter_suffix}' if nofilter_suffix else 'axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain',
            'title': f'Case iii vs vi: Vertical ({filter_tag})\nSacc-C (blue) vs Stim-C (red)',
        },
    ]
    
    def load_axes_summary_for_C_filt(path):
        if not path.exists():
            return None
        with open(path) as f:
            summary = json.load(f)
        win_C = None
        per_area = summary.get("per_area_results") or {}
        for area_name, area_data in per_area.items():
            if area_data and "winC_selected" in area_data:
                win_C = area_data["winC_selected"]
                break
        if win_C is None:
            win_C = summary.get("winC")
        return {"win_C": win_C}
    
    def load_rt_filt(cache_path):
        if not cache_path.exists():
            return None
        try:
            cache = np.load(cache_path, allow_pickle=True)
            if "lab_PT_ms" not in cache:
                return None
            rt = cache["lab_PT_ms"].astype(float)
            valid = np.isfinite(rt) & (rt > 100) & (rt < 1000)
            return rt[valid]
        except:
            return None
    
    for comp in C_COMPARISONS_FILTER:
        print(f"  Generating {comp['name']} C comparison ({filter_tag})...")
        
        session_data = []
        for sid in sessions:
            area = get_fef_area(sid)
            
            sacc_path = out_root / "sacc" / sid / "axes" / comp['sacc_tag'] / "axes_summary.json"
            sacc_info = load_axes_summary_for_C_filt(sacc_path)
            
            stim_path = out_root / "stim" / sid / "axes" / comp['stim_tag'] / "axes_summary.json"
            stim_info = load_axes_summary_for_C_filt(stim_path)
            
            if sacc_info is None or stim_info is None:
                continue
            if sacc_info["win_C"] is None or stim_info["win_C"] is None:
                continue
            
            rt = None
            for alt_area in [area, "MFEF", "MLIP", "SFEF", "SLIP"]:
                cache_path = out_root / "sacc" / sid / "caches" / f"area_{alt_area}.npz"
                rt = load_rt_filt(cache_path)
                if rt is not None and len(rt) >= 10:
                    break
            
            if rt is None or len(rt) < 10:
                continue
            
            sacc_win_ms = [sacc_info["win_C"][0] * 1000, sacc_info["win_C"][1] * 1000]
            stim_win_ms = [stim_info["win_C"][0] * 1000, stim_info["win_C"][1] * 1000]
            
            rt_median = np.median(rt)
            rt_q10, rt_q90 = np.percentile(rt, [10, 90])
            
            sacc_win_abs_median = [rt_median + sacc_win_ms[0], rt_median + sacc_win_ms[1]]
            sacc_win_abs_q10 = [rt_q10 + sacc_win_ms[0], rt_q10 + sacc_win_ms[1]]
            sacc_win_abs_q90 = [rt_q90 + sacc_win_ms[0], rt_q90 + sacc_win_ms[1]]
            
            n_overlap = 0
            for rt_i in rt:
                sacc_abs_start = rt_i + sacc_win_ms[0]
                sacc_abs_end = rt_i + sacc_win_ms[1]
                if sacc_abs_start < stim_win_ms[1] and sacc_abs_end > stim_win_ms[0]:
                    n_overlap += 1
            frac_overlap = n_overlap / len(rt)
            
            session_data.append({
                "sid": sid,
                "sacc_win_ms": sacc_win_ms,
                "stim_win_ms": stim_win_ms,
                "sacc_win_abs_median": sacc_win_abs_median,
                "sacc_win_abs_q10": sacc_win_abs_q10,
                "sacc_win_abs_q90": sacc_win_abs_q90,
                "rt_median": rt_median,
                "frac_overlap": frac_overlap,
            })
        
        if not session_data:
            print(f"    [warn] No data for {comp['name']} ({filter_tag})")
            continue
        
        session_data.sort(key=lambda x: x["rt_median"])
        n = len(session_data)
        
        fig, ax = plt.subplots(figsize=(12, max(8, n * 0.4)))
        y_positions = np.arange(n) * 2
        
        for i, r in enumerate(session_data):
            y = y_positions[i]
            
            stim_start, stim_end = r["stim_win_ms"]
            rect_stim = plt.Rectangle((stim_start, y + 0.2), stim_end - stim_start, 0.6,
                                       facecolor='red', edgecolor='darkred', alpha=0.7, linewidth=1)
            ax.add_patch(rect_stim)
            
            sacc_q10_start = r["sacc_win_abs_q10"][0]
            sacc_q90_end = r["sacc_win_abs_q90"][1]
            rect_bg = plt.Rectangle((sacc_q10_start, y - 0.8), sacc_q90_end - sacc_q10_start, 0.6,
                                     facecolor='blue', alpha=0.15, linewidth=0)
            ax.add_patch(rect_bg)
            
            sacc_med_start, sacc_med_end = r["sacc_win_abs_median"]
            rect_sacc = plt.Rectangle((sacc_med_start, y - 0.8), sacc_med_end - sacc_med_start, 0.6,
                                       facecolor='blue', edgecolor='darkblue', alpha=0.7, linewidth=1)
            ax.add_patch(rect_sacc)
            
            overlap_pct = r["frac_overlap"] * 100
            color = 'green' if overlap_pct > 50 else ('orange' if overlap_pct > 10 else 'red')
            ax.text(max(stim_end, sacc_med_end) + 10, y, f'{overlap_pct:.0f}%',
                    fontsize=7, color=color, va='center')
            ax.text(-40, y, r["sid"], fontsize=7, ha='right', va='center')
        
        ax.axvline(0, color='orange', linewidth=2, label='Stimulus onset', zorder=0)
        mean_rt = np.mean([r["rt_median"] for r in session_data])
        ax.axvline(mean_rt, color='purple', linestyle='--', linewidth=2,
                   label=f'Mean saccade ({mean_rt:.0f}ms)', zorder=0)
        
        import matplotlib.patches as mpatches
        stim_patch = mpatches.Patch(color='red', alpha=0.7, label='Stim-aligned C')
        sacc_patch = mpatches.Patch(color='blue', alpha=0.7, label='Sacc-aligned C (median RT)')
        sacc_range_patch = mpatches.Patch(color='blue', alpha=0.15, label='Sacc-C RT range (10-90%ile)')
        ax.legend(handles=[stim_patch, sacc_patch, sacc_range_patch], loc='upper right', fontsize=9)
        
        ax.set_xlim(-60, 500)
        ax.set_ylim(-1.5, n * 2)
        ax.set_yticks([])
        ax.set_xlabel("Time from Stimulus Onset (ms)")
        ax.set_title(comp['title'] + "\n(% = fraction of trials with C window overlap)")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        out_path = figs_dir / f"C_window_comparison_{comp['name']}_{filter_tag}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    [saved] {out_path}")
    
    # Generate window_absolute_timeline_detailed with filter suffix
    stim_tag = f"axes_peakbin_stimCR-stim-vertical-20mssw{nofilter_suffix if nofilter_suffix else '_mixedtrain'}"
    if nofilter_suffix:
        stim_tag = "axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain_nofilter"
    else:
        stim_tag = "axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain"
    sacc_tag = f"axes_peakbin_saccCS-sacc-horizontal-20mssw{nofilter_suffix}"
    
    all_session_windows_filt = []
    for sid in sessions:
        stim_summary_path = out_root / "stim" / sid / "axes" / stim_tag / "axes_summary.json"
        sacc_summary_path = out_root / "sacc" / sid / "axes" / sacc_tag / "axes_summary.json"
        
        if not stim_summary_path.exists() or not sacc_summary_path.exists():
            continue
        
        with open(stim_summary_path) as f:
            stim_summary = json.load(f)
        with open(sacc_summary_path) as f:
            sacc_summary = json.load(f)
        
        area = get_fef_area(sid)
        rt = load_rt_filt(out_root / "sacc" / sid / "caches" / f"area_{area}.npz")
        if rt is None or len(rt) < 10:
            for alt_area in ["MFEF", "MLIP", "SFEF", "SLIP"]:
                rt = load_rt_filt(out_root / "sacc" / sid / "caches" / f"area_{alt_area}.npz")
                if rt is not None and len(rt) >= 10:
                    break
        
        if rt is None or len(rt) < 10:
            continue
        
        stim_win_C = None
        per_area = stim_summary.get("per_area_results") or {}
        for area_name, area_data in per_area.items():
            if area_data and "winC_selected" in area_data:
                stim_win_C = area_data["winC_selected"]
                break
        if stim_win_C is None:
            stim_win_C = stim_summary.get("winC")
        
        sacc_win_S = None
        per_area = sacc_summary.get("per_area_results") or {}
        for area_name, area_data in per_area.items():
            if area_data and "winS_selected" in area_data:
                sacc_win_S = area_data["winS_selected"]
                break
        if sacc_win_S is None:
            sacc_win_S = sacc_summary.get("winS")
        
        if stim_win_C is None or sacc_win_S is None:
            continue
        
        stim_win_C_ms = [stim_win_C[0] * 1000, stim_win_C[1] * 1000]
        sacc_win_S_ms = [sacc_win_S[0] * 1000, sacc_win_S[1] * 1000]
        
        rt_median = np.median(rt)
        rt_q10, rt_q90 = np.percentile(rt, [10, 90])
        
        sacc_win_abs_median = [rt_median + sacc_win_S_ms[0], rt_median + sacc_win_S_ms[1]]
        sacc_win_abs_q10 = [rt_q10 + sacc_win_S_ms[0], rt_q10 + sacc_win_S_ms[1]]
        sacc_win_abs_q90 = [rt_q90 + sacc_win_S_ms[0], rt_q90 + sacc_win_S_ms[1]]
        
        n_overlap = 0
        for rt_i in rt:
            sacc_abs_start = rt_i + sacc_win_S_ms[0]
            sacc_abs_end = rt_i + sacc_win_S_ms[1]
            if sacc_abs_start < stim_win_C_ms[1] and sacc_abs_end > stim_win_C_ms[0]:
                n_overlap += 1
        frac_overlap = n_overlap / len(rt)
        
        all_session_windows_filt.append({
            "sid": sid,
            "stim_win_C_ms": stim_win_C_ms,
            "sacc_win_abs_median": sacc_win_abs_median,
            "sacc_win_abs_q10": sacc_win_abs_q10,
            "sacc_win_abs_q90": sacc_win_abs_q90,
            "rt_median": rt_median,
            "frac_overlap": frac_overlap,
        })
    
    if all_session_windows_filt:
        all_session_windows_filt.sort(key=lambda x: x["rt_median"])
        n = len(all_session_windows_filt)
        
        # Detailed timeline figure
        fig3, ax3 = plt.subplots(figsize=(12, max(8, n * 0.4)))
        y_positions = np.arange(n) * 2
        
        for i, r in enumerate(all_session_windows_filt):
            y = y_positions[i]
            
            stim_start, stim_end = r["stim_win_C_ms"]
            rect_stim = plt.Rectangle((stim_start, y + 0.2), stim_end - stim_start, 0.6,
                                       facecolor='red', edgecolor='darkred', alpha=0.7, linewidth=1)
            ax3.add_patch(rect_stim)
            
            sacc_q10_start = r["sacc_win_abs_q10"][0]
            sacc_q90_end = r["sacc_win_abs_q90"][1]
            rect_bg = plt.Rectangle((sacc_q10_start, y - 0.8), sacc_q90_end - sacc_q10_start, 0.6,
                                     facecolor='blue', alpha=0.15, linewidth=0)
            ax3.add_patch(rect_bg)
            
            sacc_med_start, sacc_med_end = r["sacc_win_abs_median"]
            rect_sacc = plt.Rectangle((sacc_med_start, y - 0.8), sacc_med_end - sacc_med_start, 0.6,
                                       facecolor='blue', edgecolor='darkblue', alpha=0.7, linewidth=1)
            ax3.add_patch(rect_sacc)
            
            overlap_pct = r["frac_overlap"] * 100
            color = 'green' if overlap_pct > 50 else ('orange' if overlap_pct > 10 else 'red')
            ax3.text(max(stim_end, sacc_med_end) + 10, y, f'{overlap_pct:.0f}%',
                     fontsize=7, color=color, va='center')
            ax3.text(-40, y, r["sid"], fontsize=7, ha='right', va='center')
        
        ax3.axvline(0, color='orange', linewidth=2, label='Stimulus onset', zorder=0)
        mean_rt = np.mean([r["rt_median"] for r in all_session_windows_filt])
        ax3.axvline(mean_rt, color='purple', linestyle='--', linewidth=2, 
                    label=f'Mean saccade ({mean_rt:.0f}ms)', zorder=0)
        
        import matplotlib.patches as mpatches
        stim_patch = mpatches.Patch(color='red', alpha=0.7, label='Stim-aligned C')
        sacc_patch = mpatches.Patch(color='blue', alpha=0.7, label='Sacc-aligned S (median RT)')
        sacc_range_patch = mpatches.Patch(color='blue', alpha=0.15, label='Sacc-S RT range (10-90%ile)')
        ax3.legend(handles=[stim_patch, sacc_patch, sacc_range_patch], loc='upper right', fontsize=9)
        
        ax3.set_xlim(-60, 500)
        ax3.set_ylim(-1.5, n * 2)
        ax3.set_yticks([])
        ax3.set_xlabel("Time from Stimulus Onset (ms)")
        ax3.set_title(f"Window Positions ({filter_tag})\n(% = fraction of trials with overlap)")
        ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        timeline_path = figs_dir / f"window_absolute_timeline_detailed_{filter_tag}.pdf"
        fig3.savefig(timeline_path, dpi=150, bbox_inches='tight')
        fig3.savefig(timeline_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  [saved] {timeline_path}")
        
        # Summary figure
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
        
        ax4a = axes4[0]
        for i, r in enumerate(all_session_windows_filt):
            stim_start, stim_end = r["stim_win_C_ms"]
            ax4a.barh(i, stim_end - stim_start, left=stim_start, height=0.35,
                      color='red', alpha=0.6, edgecolor='darkred', linewidth=0.5)
            sacc_start, sacc_end = r["sacc_win_abs_median"]
            ax4a.barh(i - 0.35, sacc_end - sacc_start, left=sacc_start, height=0.35,
                      color='blue', alpha=0.6, edgecolor='darkblue', linewidth=0.5)
            sacc_q10_start = r["sacc_win_abs_q10"][0]
            sacc_q90_end = r["sacc_win_abs_q90"][1]
            ax4a.plot([sacc_q10_start, sacc_q90_end], [i - 0.35, i - 0.35],
                      color='blue', alpha=0.3, linewidth=6, solid_capstyle='butt')
        
        ax4a.axvline(mean_rt, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
        ax4a.axvline(0, color='orange', linewidth=2)
        ax4a.set_yticks(range(n))
        ax4a.set_yticklabels([r["sid"] for r in all_session_windows_filt], fontsize=7)
        ax4a.set_xlabel("Time from Stimulus Onset (ms)")
        ax4a.set_ylabel("Session (sorted by median RT)")
        ax4a.set_title(f"Window Positions ({filter_tag})")
        ax4a.grid(axis='x', alpha=0.3)
        ax4a.set_xlim(-50, 450)
        
        ax4b = axes4[1]
        overlaps = [r["frac_overlap"] for r in all_session_windows_filt]
        ax4b.hist(overlaps, bins=20, range=(0, 1), color='green', alpha=0.7, edgecolor='darkgreen')
        ax4b.axvline(np.mean(overlaps), color='black', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(overlaps):.1%}')
        ax4b.set_xlabel("Fraction of trials with window overlap")
        ax4b.set_ylabel("Number of sessions")
        ax4b.set_title(f"Window Overlap ({filter_tag})")
        ax4b.legend()
        ax4b.set_xlim(0, 1)
        
        plt.suptitle(f"Training Window Comparison ({filter_tag})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        summary_path = figs_dir / f"window_comparison_all_sessions_{filter_tag}.pdf"
        fig4.savefig(summary_path, dpi=150, bbox_inches='tight')
        fig4.savefig(summary_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  [saved] {summary_path}")
    else:
        print(f"  [warn] No session window data for {filter_tag}")

# Generate for both filter modes
figs_dir_combined = PROJECT / "out" / "axis_alignment_all_cases_figs"
figs_dir_combined.mkdir(parents=True, exist_ok=True)

generate_window_comparison_figures_for_filter(
    PROJECT / "out", "correctonly", figs_dir_combined
)
generate_window_comparison_figures_for_filter(
    PROJECT / "out_nofilter", "alltrials", figs_dir_combined
)

print("\n" + "="*80)
print("DONE - ALL FIGURES GENERATED WITH FILTER SUFFIXES")
print("="*80)
PYTHON_SCRIPT

echo "[done] Summary complete"
