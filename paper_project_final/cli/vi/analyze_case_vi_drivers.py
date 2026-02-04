#!/usr/bin/env python3
"""
Analyze drivers of high alignment in Case vi.

This script examines what factors correlate with high observed alignment
in Case vi (Stim-C(vert) vs Sacc-S(horiz)) to understand why this case
shows unusually high C-S alignment.

Analyses:
1. Cross-orientation matrix: computes |cos(θ)| for all 4 C×S combinations
   - C(horiz) vs S(horiz), C(horiz) vs S(vert), C(vert) vs S(horiz), C(vert) vs S(vert)
2. Session-level correlations: a_obs vs decoding performance, D_eff, n_trials, RT
3. Comparison with reverse case (if available): Stim-C(horiz) vs Sacc-S(vert)
4. Axis quality metrics from QC data
5. Manifold geometry comparison across cases

Output:
- Summary statistics and correlations
- Cross-orientation matrix results
- Figures showing relationships
- JSON with all computed metrics
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


def load_npz(path: Path) -> Optional[Dict]:
    """Load NPZ file."""
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def get_monkey(sid: str) -> str:
    """Return 'M' for 2020 sessions, 'S' for 2023."""
    return "M" if sid.startswith("2020") else "S"


def get_fef_area(sid: str) -> str:
    """Return FEF area name based on monkey."""
    return "MFEF" if get_monkey(sid) == "M" else "SFEF"


def load_axes(axes_path: Path):
    """Load C and S axes from npz file."""
    if not axes_path.exists():
        return None, None
    d = np.load(axes_path, allow_pickle=True)
    
    w_C = d.get("sC", None)
    if w_C is not None and w_C.size == 0:
        w_C = None
    if w_C is not None:
        w_C = w_C.flatten()
    
    w_S = d.get("sS_raw", d.get("sS_inv", None))
    if w_S is not None and w_S.size == 0:
        w_S = None
    if w_S is not None:
        w_S = w_S.flatten()
    
    return w_C, w_S


def compute_axis_alignment(w1: np.ndarray, w2: np.ndarray) -> float:
    """Compute |cos(θ)| between two axes."""
    if w1 is None or w2 is None:
        return np.nan
    w1 = w1.flatten() / (np.linalg.norm(w1) + 1e-10)
    w2 = w2.flatten() / (np.linalg.norm(w2) + 1e-10)
    cos_theta = np.abs(np.dot(w1, w2))
    return float(np.clip(cos_theta, 0, 1))


def compute_cross_orientation_matrix(project: Path, sids: List[str], out_root: str = "out") -> Dict:
    """
    Compute the full cross-orientation matrix: C×S for all 4 orientation combinations.
    
    Returns alignment values for:
    - C(horiz) vs S(horiz)
    - C(horiz) vs S(vert)
    - C(vert) vs S(horiz)  <- Case vi
    - C(vert) vs S(vert)
    """
    print("\n" + "=" * 70)
    print("CROSS-ORIENTATION MATRIX")
    print("=" * 70)
    
    cross_data = {
        "C(h)-S(h)": [], "C(h)-S(v)": [], 
        "C(v)-S(h)": [], "C(v)-S(v)": []
    }
    
    per_session = []
    
    for sid in sids:
        area = get_fef_area(sid)
        out_path = project / out_root
        
        # Load all axes
        axes = {}
        
        # Stim-aligned C axes
        for ori in ["horizontal", "vertical"]:
            tag = f"axes_peakbin_stimC-stim-{ori}-20mssw"
            path = out_path / "stim" / sid / "axes" / tag / f"axes_{area}.npz"
            w_C, _ = load_axes(path)
            if w_C is not None:
                axes[f"C_stim_{ori}"] = w_C
        
        # Sacc-aligned S axes
        for ori in ["horizontal", "vertical"]:
            tag = f"axes_peakbin_saccCS-sacc-{ori}-20mssw"
            path = out_path / "sacc" / sid / "axes" / tag / f"axes_{area}.npz"
            _, w_S = load_axes(path)
            if w_S is not None:
                axes[f"S_sacc_{ori}"] = w_S
        
        if not axes:
            continue
        
        session_result = {"sid": sid, "area": area}
        
        # Compute all 4 combinations
        for c_ori in ["horizontal", "vertical"]:
            for s_ori in ["horizontal", "vertical"]:
                c_key = f"C_stim_{c_ori}"
                s_key = f"S_sacc_{s_ori}"
                
                matrix_key = f"C({c_ori[0]})-S({s_ori[0]})"
                
                if c_key in axes and s_key in axes:
                    cos_val = compute_axis_alignment(axes[c_key], axes[s_key])
                    cross_data[matrix_key].append(cos_val)
                    session_result[matrix_key] = cos_val
        
        per_session.append(session_result)
    
    # Print matrix
    print("\nCross-Orientation Matrix (Stim-C vs Sacc-S):")
    print("                    S(horizontal)    S(vertical)")
    
    for c_ori, c_label in [("h", "horizontal"), ("v", "vertical")]:
        s_h_key = f"C({c_ori})-S(h)"
        s_v_key = f"C({c_ori})-S(v)"
        
        s_h_vals = cross_data.get(s_h_key, [])
        s_v_vals = cross_data.get(s_v_key, [])
        
        s_h_str = f"{np.mean(s_h_vals):.3f} ± {np.std(s_h_vals):.3f}" if s_h_vals else "N/A"
        s_v_str = f"{np.mean(s_v_vals):.3f} ± {np.std(s_v_vals):.3f}" if s_v_vals else "N/A"
        
        print(f"C({c_label:10s})    {s_h_str:16s}    {s_v_str:16s}")
    
    # Check asymmetry
    cv_sh = cross_data.get("C(v)-S(h)", [])
    ch_sv = cross_data.get("C(h)-S(v)", [])
    
    if cv_sh and ch_sv:
        diff = np.mean(cv_sh) - np.mean(ch_sv)
        print(f"\n*** ASYMMETRY: C(v)-S(h) - C(h)-S(v) = {diff:.3f} ***")
        if abs(diff) > 0.1:
            print("    Significant asymmetry detected!")
    
    return {
        "matrix": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} 
                   for k, v in cross_data.items() if v},
        "per_session": per_session
    }


def load_case_results(project: Path, case_dir: str, prefix: str) -> List[Dict]:
    """Load all session results for a case."""
    results_dir = project / case_dir
    if not results_dir.exists():
        print(f"[WARN] Directory not found: {results_dir}")
        return []
    
    results = []
    for json_file in sorted(results_dir.glob(f"{prefix}*.json")):
        data = load_json(json_file)
        if data:
            # Extract session ID from filename
            sid = json_file.stem.replace(prefix, "")
            data["sid"] = sid
            results.append(data)
    
    return results


def load_axes_summary(project: Path, align: str, sid: str, axes_tag: str) -> Optional[Dict]:
    """Load axes summary for a session."""
    path = project / "out" / align / sid / "axes" / axes_tag / "axes_summary.json"
    return load_json(path)


def load_qc_data(project: Path, align: str, sid: str, axes_tag: str) -> Optional[Dict]:
    """Load QC data for axes."""
    area = get_fef_area(sid)
    path = project / "out" / align / sid / "qc" / axes_tag / f"qc_axes_{area}.json"
    return load_json(path)


def compute_correlations(x: np.ndarray, y: np.ndarray, label_x: str, label_y: str) -> Dict:
    """Compute correlation between two variables."""
    # Remove NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return {"r": np.nan, "p": np.nan, "n": mask.sum(), "x_label": label_x, "y_label": label_y}
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    r, p = stats.pearsonr(x_clean, y_clean)
    return {
        "r": float(r),
        "p": float(p),
        "n": int(mask.sum()),
        "x_label": label_x,
        "y_label": label_y,
        "x_mean": float(np.mean(x_clean)),
        "y_mean": float(np.mean(y_clean)),
    }


def analyze_case_vi_drivers(project: Path, out_dir: Path, trial_mode: str = "correctonly"):
    """Main analysis of Case vi drivers.
    
    Parameters:
    -----------
    project : Path
        Project root directory
    out_dir : Path
        Output directory for results
    trial_mode : str
        "correctonly" or "alltrials" - determines which results to load
    """
    print("=" * 70)
    print(f"ANALYZING CASE VI DRIVERS ({trial_mode})")
    print("=" * 70)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine path suffix based on trial mode
    suffix = "correctonly" if trial_mode == "correctonly" else "alltrials"
    out_root = "out" if trial_mode == "correctonly" else "out_nofilter"
    
    # Load Case vi results (file: case_iii_{suffix}, which maps to case vi)
    # Case vi = Stim-C(vert) vs Sacc-S(horiz)
    case_vi_results = load_case_results(
        project, 
        f"{out_root}/cross_alignment/case_iii_{suffix}",
        f"cross_case_iii_{suffix}_"
    )
    
    # Get all session IDs for cross-orientation matrix
    sids_all = [r["sid"] for r in case_vi_results] if case_vi_results else []
    
    # Compute cross-orientation matrix (uses out_root)
    cross_matrix_results = compute_cross_orientation_matrix(project, sids_all, out_root) if sids_all else None
    print(f"\n[Case vi] Loaded {len(case_vi_results)} sessions")
    
    # Load other cases for comparison
    # Case iv = Stim-C(horiz) vs Sacc-S(horiz) (file: case_v_{suffix})
    case_iv_results = load_case_results(
        project,
        f"{out_root}/cross_alignment/case_v_{suffix}",
        f"cross_case_v_{suffix}_"
    )
    print(f"[Case iv] Loaded {len(case_iv_results)} sessions")
    
    # Case v = Stim-C(pooled) vs Sacc-S(pooled) (file: case_iv_{suffix})
    case_v_results = load_case_results(
        project,
        f"{out_root}/cross_alignment/case_iv_{suffix}",
        f"cross_case_iv_{suffix}_"
    )
    print(f"[Case v] Loaded {len(case_v_results)} sessions")
    
    # Load same-alignment cases for reference
    case_i_results = load_case_results(
        project,
        f"{out_root}/sacc/alignment/case_i_{suffix}",
        "axis_cov_"
    )
    print(f"[Case i] Loaded {len(case_i_results)} sessions")
    
    case_iii_sacc_results = load_case_results(
        project,
        f"{out_root}/sacc/alignment/case_vi_{suffix}",  # case_vi in files = case iii (sacc-vert)
        "axis_cov_"
    )
    print(f"[Case iii] Loaded {len(case_iii_sacc_results)} sessions")
    
    if not case_vi_results:
        print("[ERROR] No Case vi results found!")
        return
    
    # Extract metrics for Case vi
    sids_vi = [r["sid"] for r in case_vi_results]
    a_obs_vi = np.array([r["a_obs"] for r in case_vi_results])
    null_mean_vi = np.array([r["null_mean"] for r in case_vi_results])
    D_eff_vi = np.array([r["D_eff"] for r in case_vi_results])
    manifold_dim_vi = np.array([r["manifold_dim"] for r in case_vi_results])
    n_trials_C_vi = np.array([r.get("n_trials_C", np.nan) for r in case_vi_results])
    n_trials_S_vi = np.array([r.get("n_trials_S", np.nan) for r in case_vi_results])
    delta_vi = np.array([r["delta"] for r in case_vi_results])
    z_score_vi = np.array([r["z_score"] for r in case_vi_results])
    
    # Load axes quality metrics
    auc_C_vi = []
    auc_S_vi = []
    winC_selected = []
    winS_selected = []
    
    for sid in sids_vi:
        # C axis from stim-vertical
        axes_C = load_axes_summary(project, "stim", sid, "axes_peakbin_stimC-stim-vertical-20mssw")
        # S axis from sacc-horizontal
        axes_S = load_axes_summary(project, "sacc", sid, "axes_peakbin_saccCS-sacc-horizontal-20mssw")
        
        area = get_fef_area(sid)
        
        if axes_C and "per_area_results" in axes_C:
            area_results = axes_C["per_area_results"].get(area, {})
            auc_C_vi.append(area_results.get("sC_auc", np.nan))
            winC_selected.append(area_results.get("winC_selected", [np.nan, np.nan]))
        else:
            auc_C_vi.append(np.nan)
            winC_selected.append([np.nan, np.nan])
        
        if axes_S and "per_area_results" in axes_S:
            area_results = axes_S["per_area_results"].get(area, {})
            # Get S axis AUC (try winS_cv_best first)
            s_auc = area_results.get("winS_cv_best", area_results.get("winS_peak_auc", np.nan))
            auc_S_vi.append(s_auc)
            winS_selected.append(area_results.get("winS_selected", [np.nan, np.nan]))
        else:
            auc_S_vi.append(np.nan)
            winS_selected.append([np.nan, np.nan])
    
    auc_C_vi = np.array(auc_C_vi)
    auc_S_vi = np.array(auc_S_vi)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("CASE VI SUMMARY STATISTICS")
    print("=" * 70)
    print(f"N sessions: {len(case_vi_results)}")
    print(f"a_obs:      mean={np.nanmean(a_obs_vi):.3f}, std={np.nanstd(a_obs_vi):.3f}, "
          f"min={np.nanmin(a_obs_vi):.3f}, max={np.nanmax(a_obs_vi):.3f}")
    print(f"null_mean:  mean={np.nanmean(null_mean_vi):.3f}, std={np.nanstd(null_mean_vi):.3f}")
    print(f"delta:      mean={np.nanmean(delta_vi):.3f}, std={np.nanstd(delta_vi):.3f}")
    print(f"D_eff:      mean={np.nanmean(D_eff_vi):.1f}, std={np.nanstd(D_eff_vi):.1f}")
    print(f"auc_C:      mean={np.nanmean(auc_C_vi):.3f}, std={np.nanstd(auc_C_vi):.3f}")
    print(f"auc_S:      mean={np.nanmean(auc_S_vi):.3f}, std={np.nanstd(auc_S_vi):.3f}")
    
    # Identify high and low alignment sessions
    high_threshold = np.nanpercentile(a_obs_vi, 75)
    low_threshold = np.nanpercentile(a_obs_vi, 25)
    
    high_sessions = [sids_vi[i] for i in range(len(sids_vi)) if a_obs_vi[i] >= high_threshold]
    low_sessions = [sids_vi[i] for i in range(len(sids_vi)) if a_obs_vi[i] <= low_threshold]
    
    print(f"\nHigh alignment sessions (a_obs >= {high_threshold:.3f}): {high_sessions}")
    print(f"Low alignment sessions (a_obs <= {low_threshold:.3f}): {low_sessions}")
    
    # Compute correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    correlations = []
    
    # a_obs vs various metrics
    corr_D_eff = compute_correlations(a_obs_vi, D_eff_vi, "a_obs", "D_eff")
    correlations.append(corr_D_eff)
    print(f"a_obs vs D_eff:       r={corr_D_eff['r']:.3f}, p={corr_D_eff['p']:.4f}")
    
    corr_manifold = compute_correlations(a_obs_vi, manifold_dim_vi, "a_obs", "manifold_dim")
    correlations.append(corr_manifold)
    print(f"a_obs vs manifold_dim: r={corr_manifold['r']:.3f}, p={corr_manifold['p']:.4f}")
    
    corr_null = compute_correlations(a_obs_vi, null_mean_vi, "a_obs", "null_mean")
    correlations.append(corr_null)
    print(f"a_obs vs null_mean:   r={corr_null['r']:.3f}, p={corr_null['p']:.4f}")
    
    corr_auc_C = compute_correlations(a_obs_vi, auc_C_vi, "a_obs", "auc_C")
    correlations.append(corr_auc_C)
    print(f"a_obs vs auc_C:       r={corr_auc_C['r']:.3f}, p={corr_auc_C['p']:.4f}")
    
    corr_auc_S = compute_correlations(a_obs_vi, auc_S_vi, "a_obs", "auc_S")
    correlations.append(corr_auc_S)
    print(f"a_obs vs auc_S:       r={corr_auc_S['r']:.3f}, p={corr_auc_S['p']:.4f}")
    
    corr_trials_C = compute_correlations(a_obs_vi, n_trials_C_vi, "a_obs", "n_trials_C")
    correlations.append(corr_trials_C)
    print(f"a_obs vs n_trials_C:  r={corr_trials_C['r']:.3f}, p={corr_trials_C['p']:.4f}")
    
    corr_trials_S = compute_correlations(a_obs_vi, n_trials_S_vi, "a_obs", "n_trials_S")
    correlations.append(corr_trials_S)
    print(f"a_obs vs n_trials_S:  r={corr_trials_S['r']:.3f}, p={corr_trials_S['p']:.4f}")
    
    # Compare with other cases
    print("\n" + "=" * 70)
    print("COMPARISON WITH OTHER CASES")
    print("=" * 70)
    
    case_comparison = {}
    
    for case_name, case_results in [
        ("case_i (sacc-horiz)", case_i_results),
        ("case_iii (sacc-vert)", case_iii_sacc_results),
        ("case_iv (stim-horiz vs sacc-horiz)", case_iv_results),
        ("case_v (stim-pool vs sacc-pool)", case_v_results),
        ("case_vi (stim-vert vs sacc-horiz)", case_vi_results),
    ]:
        if case_results:
            a_obs_case = np.array([r["a_obs"] for r in case_results])
            null_mean_case = np.array([r["null_mean"] for r in case_results])
            D_eff_case = np.array([r.get("D_eff", np.nan) for r in case_results])
            
            case_comparison[case_name] = {
                "n": len(case_results),
                "a_obs_mean": float(np.nanmean(a_obs_case)),
                "a_obs_std": float(np.nanstd(a_obs_case)),
                "null_mean_mean": float(np.nanmean(null_mean_case)),
                "D_eff_mean": float(np.nanmean(D_eff_case)),
                "delta_mean": float(np.nanmean(a_obs_case - null_mean_case)),
            }
            
            print(f"{case_name:40s}: N={len(case_results):2d}, "
                  f"a_obs={np.nanmean(a_obs_case):.3f}±{np.nanstd(a_obs_case):.3f}, "
                  f"null={np.nanmean(null_mean_case):.3f}, "
                  f"delta={np.nanmean(a_obs_case - null_mean_case):.3f}")
    
    # Statistical test: Case vi vs Case iv (both cross-alignment, different orientations)
    if case_iv_results and case_vi_results:
        a_obs_iv = np.array([r["a_obs"] for r in case_iv_results])
        t_stat, t_p = stats.ttest_ind(a_obs_vi, a_obs_iv)
        print(f"\nt-test (case vi vs case iv): t={t_stat:.3f}, p={t_p:.4f}")
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(a_obs_vi, a_obs_iv, alternative='greater')
        print(f"Mann-Whitney U (case vi > case iv): U={u_stat:.1f}, p={u_p:.4f}")
    
    # Create figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    # Figure 1: Scatter plots of correlations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # a_obs vs D_eff
    ax = axes[0, 0]
    ax.scatter(D_eff_vi, a_obs_vi, c='orange', alpha=0.7, s=50)
    ax.set_xlabel('D_eff')
    ax.set_ylabel('|cos(θ)| observed')
    ax.set_title(f'a_obs vs D_eff\nr={corr_D_eff["r"]:.3f}, p={corr_D_eff["p"]:.4f}')
    ax.axhline(np.nanmean(null_mean_vi), color='gray', linestyle='--', label='null mean')
    ax.legend()
    
    # a_obs vs null_mean
    ax = axes[0, 1]
    ax.scatter(null_mean_vi, a_obs_vi, c='orange', alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Null mean')
    ax.set_ylabel('|cos(θ)| observed')
    ax.set_title(f'a_obs vs null_mean\nr={corr_null["r"]:.3f}, p={corr_null["p"]:.4f}')
    ax.legend()
    
    # a_obs vs auc_C
    ax = axes[0, 2]
    mask = np.isfinite(auc_C_vi)
    ax.scatter(auc_C_vi[mask], a_obs_vi[mask], c='orange', alpha=0.7, s=50)
    ax.set_xlabel('C axis AUC (stim-vert)')
    ax.set_ylabel('|cos(θ)| observed')
    ax.set_title(f'a_obs vs auc_C\nr={corr_auc_C["r"]:.3f}, p={corr_auc_C["p"]:.4f}')
    
    # a_obs vs auc_S
    ax = axes[1, 0]
    mask = np.isfinite(auc_S_vi)
    ax.scatter(auc_S_vi[mask], a_obs_vi[mask], c='orange', alpha=0.7, s=50)
    ax.set_xlabel('S axis AUC (sacc-horiz)')
    ax.set_ylabel('|cos(θ)| observed')
    ax.set_title(f'a_obs vs auc_S\nr={corr_auc_S["r"]:.3f}, p={corr_auc_S["p"]:.4f}')
    
    # a_obs vs n_trials_C
    ax = axes[1, 1]
    ax.scatter(n_trials_C_vi, a_obs_vi, c='orange', alpha=0.7, s=50)
    ax.set_xlabel('N trials (C, vertical)')
    ax.set_ylabel('|cos(θ)| observed')
    ax.set_title(f'a_obs vs n_trials_C\nr={corr_trials_C["r"]:.3f}, p={corr_trials_C["p"]:.4f}')
    
    # a_obs vs manifold_dim
    ax = axes[1, 2]
    ax.scatter(manifold_dim_vi, a_obs_vi, c='orange', alpha=0.7, s=50)
    ax.set_xlabel('Manifold dimension')
    ax.set_ylabel('|cos(θ)| observed')
    ax.set_title(f'a_obs vs manifold_dim\nr={corr_manifold["r"]:.3f}, p={corr_manifold["p"]:.4f}')
    
    plt.tight_layout()
    fig.savefig(out_dir / "case_vi_correlations.png", dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / "case_vi_correlations.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir / 'case_vi_correlations.png'}")
    
    # Figure 2: Case comparison box plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot of a_obs across cases
    ax = axes[0]
    case_data = []
    case_labels = []
    case_colors = []
    
    for case_name, case_results, color in [
        ("i\n(sacc-h)", case_i_results, '#3498db'),
        ("iii\n(sacc-v)", case_iii_sacc_results, '#1abc9c'),
        ("iv\n(stim-h)", case_iv_results, '#e74c3c'),
        ("v\n(stim-p)", case_v_results, '#9b59b6'),
        ("vi\n(stim-v)", case_vi_results, '#f39c12'),
    ]:
        if case_results:
            case_data.append([r["a_obs"] for r in case_results])
            case_labels.append(case_name)
            case_colors.append(color)
    
    bp = ax.boxplot(case_data, tick_labels=case_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], case_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('|cos(θ)| observed')
    ax.set_xlabel('Case')
    ax.set_title('Observed Alignment Across Cases')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Box plot of delta (obs - null) across cases
    ax = axes[1]
    case_delta_data = []
    
    for case_results in [case_i_results, case_iii_sacc_results, case_iv_results, case_v_results, case_vi_results]:
        if case_results:
            deltas = [r["a_obs"] - r["null_mean"] for r in case_results]
            case_delta_data.append(deltas)
    
    bp2 = ax.boxplot(case_delta_data, tick_labels=case_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], case_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Δ = a_obs - null_mean')
    ax.set_xlabel('Case')
    ax.set_title('Alignment Relative to Null')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(out_dir / "case_comparison.png", dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / "case_comparison.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir / 'case_comparison.png'}")
    
    # Figure 3: Session-by-session detail for Case vi
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort sessions by a_obs
    sorted_idx = np.argsort(a_obs_vi)[::-1]
    sorted_sids = [sids_vi[i] for i in sorted_idx]
    sorted_a_obs = a_obs_vi[sorted_idx]
    sorted_null = null_mean_vi[sorted_idx]
    
    x = np.arange(len(sorted_sids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sorted_a_obs, width, label='Observed', color='#f39c12', alpha=0.8)
    bars2 = ax.bar(x + width/2, sorted_null, width, label='Null mean', color='gray', alpha=0.6)
    
    ax.set_ylabel('|cos(θ)|')
    ax.set_xlabel('Session')
    ax.set_title('Case vi: Session-by-Session Alignment')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sids, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Add monkey labels
    for i, sid in enumerate(sorted_sids):
        monkey = get_monkey(sid)
        ax.annotate(monkey, (i, -0.05), ha='center', fontsize=7, color='blue' if monkey == 'M' else 'green')
    
    plt.tight_layout()
    fig.savefig(out_dir / "case_vi_sessions.png", dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / "case_vi_sessions.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir / 'case_vi_sessions.png'}")
    
    # Save results to JSON
    results = {
        "analysis": "case_vi_drivers",
        "n_sessions": len(case_vi_results),
        "summary": {
            "a_obs_mean": float(np.nanmean(a_obs_vi)),
            "a_obs_std": float(np.nanstd(a_obs_vi)),
            "a_obs_min": float(np.nanmin(a_obs_vi)),
            "a_obs_max": float(np.nanmax(a_obs_vi)),
            "null_mean_mean": float(np.nanmean(null_mean_vi)),
            "D_eff_mean": float(np.nanmean(D_eff_vi)),
            "auc_C_mean": float(np.nanmean(auc_C_vi)),
            "auc_S_mean": float(np.nanmean(auc_S_vi)),
        },
        "correlations": correlations,
        "case_comparison": case_comparison,
        "high_alignment_sessions": high_sessions,
        "low_alignment_sessions": low_sessions,
        "cross_orientation_matrix": cross_matrix_results["matrix"] if cross_matrix_results else None,
        "session_details": [
            {
                "sid": sids_vi[i],
                "monkey": get_monkey(sids_vi[i]),
                "a_obs": float(a_obs_vi[i]),
                "null_mean": float(null_mean_vi[i]),
                "delta": float(delta_vi[i]),
                "D_eff": float(D_eff_vi[i]),
                "auc_C": float(auc_C_vi[i]) if np.isfinite(auc_C_vi[i]) else None,
                "auc_S": float(auc_S_vi[i]) if np.isfinite(auc_S_vi[i]) else None,
                "n_trials_C": int(n_trials_C_vi[i]) if np.isfinite(n_trials_C_vi[i]) else None,
                "n_trials_S": int(n_trials_S_vi[i]) if np.isfinite(n_trials_S_vi[i]) else None,
            }
            for i in range(len(sids_vi))
        ],
    }
    
    with open(out_dir / "case_vi_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_dir / 'case_vi_analysis.json'}")
    
    # Save cross-orientation matrix separately for easy access
    if cross_matrix_results:
        with open(out_dir.parent / "diagnose_case_vi_matrix.json", "w") as f:
            json.dump(cross_matrix_results, f, indent=2)
        print(f"  Saved: {out_dir.parent / 'diagnose_case_vi_matrix.json'}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Case vi alignment drivers")
    parser.add_argument("--project", default="/project/bdoiron/dracoxu/rct-fsflow/paper_project_final",
                        help="Project root directory")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: project/out/vi/case_vi_analysis_{trial_mode})")
    parser.add_argument("--trial_mode", default="correctonly", choices=["correctonly", "alltrials"],
                        help="Trial filter mode: correctonly or alltrials")
    args = parser.parse_args()
    
    project = Path(args.project)
    out_dir = Path(args.out_dir) if args.out_dir else project / "out" / "vi" / f"case_vi_analysis_{args.trial_mode}"
    
    analyze_case_vi_drivers(project, out_dir, args.trial_mode)


if __name__ == "__main__":
    main()
