#!/usr/bin/env python3
"""
Reverse Case Analysis: Stim-C(horizontal) vs Sacc-S(vertical)

This is the reverse of Case vi (Stim-C(vert) vs Sacc-S(horiz)).
If Case vi's high alignment is due to genuine cross-orientation structure,
the reverse case might show a different or similar pattern.

This analysis helps distinguish between:
1. Genuine cross-orientation information sharing (reverse might be similar)
2. Orientation-specific artifact (reverse would be different)
3. General alignment inflation (reverse would also be high)
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")

# Import from paperflow
from paperflow.norm import sliding_window_cache_data

# Import from the existing cross-alignment script in cli/
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add cli/ to path
from axis_alignment_cross import (
    load_npz, get_monkey, get_fef_area, unit_vec, window_mask,
    get_cache_sigma, convert_axis_to_raw, convert_axis_to_standardized,
    trial_mask, condition_average_activity
)


def analyze_reverse_case(
    out_root: Path,
    sid: str,
    n_perms: int = 1000,
    seed: int = 42,
    pt_min_ms: float = 200.0,
    use_rich_cond: bool = False,
    axes_suffix: str = "20mssw",
) -> Optional[Dict]:
    """
    Reverse case analysis: Stim-C(horizontal) vs Sacc-S(vertical)
    
    This is the opposite orientation pairing from Case vi.
    """
    area = get_fef_area(sid)
    
    # Reverse case configuration
    align_C = "stim"
    axes_tag_C = f"axes_peakbin_stimC-stim-horizontal-{axes_suffix}"  # C from horizontal
    orientation_C = "horizontal"
    win_C = (0.0, 0.5)
    
    align_S = "sacc"
    axes_tag_S = f"axes_peakbin_saccCS-sacc-vertical-{axes_suffix}"  # S from vertical
    orientation_S = "vertical"
    win_S = (-0.2, 0.05)
    
    # Load caches
    cache_path_C = out_root / align_C / sid / "caches" / f"area_{area}.npz"
    cache_path_S = out_root / align_S / sid / "caches" / f"area_{area}.npz"
    
    if not cache_path_C.exists():
        print(f"  [skip] No cache for C: {cache_path_C}")
        return None
    if not cache_path_S.exists():
        print(f"  [skip] No cache for S: {cache_path_S}")
        return None
    
    cache_C = load_npz(cache_path_C)
    cache_S = load_npz(cache_path_S)
    
    # Load axes
    axes_path_C = out_root / align_C / sid / "axes" / axes_tag_C / f"axes_{area}.npz"
    axes_path_S = out_root / align_S / sid / "axes" / axes_tag_S / f"axes_{area}.npz"
    
    if not axes_path_C.exists():
        print(f"  [skip] No axes for C: {axes_path_C}")
        return None
    if not axes_path_S.exists():
        print(f"  [skip] No axes for S: {axes_path_S}")
        return None
    
    axes_C = load_npz(axes_path_C)
    axes_S = load_npz(axes_path_S)
    
    # Get axis vectors
    sC_Z = axes_C.get("sC", np.array([]))
    sS_Z = axes_S.get("sS_raw", np.array([]))
    
    if sC_Z.size == 0 or sS_Z.size == 0:
        print(f"  [skip] Missing sC or sS_raw")
        return None
    
    sC_Z = sC_Z.ravel()
    sS_Z = sS_Z.ravel()
    
    # Check dimension compatibility
    n_units_C = cache_C["Z"].shape[2]
    n_units_S = cache_S["Z"].shape[2]
    
    if n_units_C != n_units_S:
        print(f"  [skip] Unit count mismatch: C has {n_units_C}, S has {n_units_S}")
        return None
    
    if sC_Z.size != n_units_C or sS_Z.size != n_units_S:
        print(f"  [skip] Axis dimension mismatch")
        return None
    
    n_units = n_units_C
    
    # Get meta for normalization
    meta_C = axes_C.get("meta", {})
    meta_S = axes_S.get("meta", {})
    if isinstance(meta_C, str):
        meta_C = json.loads(meta_C)
    if isinstance(meta_S, str):
        meta_S = json.loads(meta_S)
    
    # Convert axes to raw space
    sigma_C = get_cache_sigma(cache_C, meta_C)
    sigma_S = get_cache_sigma(cache_S, meta_S)
    
    sC_raw = convert_axis_to_raw(sC_Z, sigma_C)
    sS_raw = convert_axis_to_raw(sS_Z, sigma_S)
    
    # --- Apply sliding window transformation to BOTH caches for OBSERVED geometry ---
    # Parameters: 20ms window, 10ms step
    # IMPORTANT: Native bin sizes differ by alignment:
    #   - Stim-aligned caches: 10ms native bins
    #   - Sacc-aligned caches: 5ms native bins
    sliding_window_ms = 20.0
    sliding_step_ms = 10.0
    
    # Stim-aligned cache (cache_C): 10ms native bins
    native_bin_ms_stim = 10.0
    sw_bins_C = int(sliding_window_ms / native_bin_ms_stim)  # 2 bins
    sw_step_C = int(sliding_step_ms / native_bin_ms_stim)    # 1 bin
    
    # Sacc-aligned cache (cache_S): 5ms native bins
    native_bin_ms_sacc = 5.0
    sw_bins_S = int(sliding_window_ms / native_bin_ms_sacc)  # 4 bins
    sw_step_S = int(sliding_step_ms / native_bin_ms_sacc)    # 2 bins
    
    print(f"  [preprocess] Applying sliding window to both caches for observed geometry...")
    print(f"               {sliding_window_ms}ms window, {sliding_step_ms}ms step")
    print(f"               C (stim): {sw_bins_C} bins, step {sw_step_C}")
    print(f"               S (sacc): {sw_bins_S} bins, step {sw_step_S}")
    
    # Apply SW with alignment-specific parameters
    if sw_bins_C > 0 and sw_step_C > 0:
        cache_C_sw, _ = sliding_window_cache_data(cache_C, sw_bins_C, sw_step_C)
    else:
        cache_C_sw = cache_C
    
    if sw_bins_S > 0 and sw_step_S > 0:
        cache_S_sw, _ = sliding_window_cache_data(cache_S, sw_bins_S, sw_step_S)
    else:
        cache_S_sw = cache_S
    
    # Build trial masks on SW-transformed caches
    mask_C = trial_mask(cache_C_sw, orientation_C, pt_min_ms)
    mask_S = trial_mask(cache_S_sw, orientation_S, pt_min_ms)
    
    if mask_C.sum() < 40:
        print(f"  [skip] Not enough trials for C: {mask_C.sum()}")
        return None
    if mask_S.sum() < 40:
        print(f"  [skip] Not enough trials for S: {mask_S.sum()}")
        return None
    
    # Get Z-scored data from SW-transformed caches (same as same-alignment)
    time_C = cache_C_sw["time"].astype(float)
    time_S = cache_S_sw["time"].astype(float)
    
    Z_C = cache_C_sw["Z"][mask_C].astype(np.float64)
    Z_S = cache_S_sw["Z"][mask_S].astype(np.float64)
    
    # Labels
    C_labels_C = cache_C_sw.get("lab_C", np.full(cache_C_sw["Z"].shape[0], np.nan))[mask_C].astype(float)
    S_labels_S = cache_S_sw.get("lab_S", np.full(cache_S_sw["Z"].shape[0], np.nan))[mask_S].astype(float)
    
    S_labels_C = cache_C_sw.get("lab_S", np.full(cache_C_sw["Z"].shape[0], np.nan))[mask_C].astype(float)
    R_labels_C = cache_C_sw.get("lab_R", np.full(cache_C_sw["Z"].shape[0], np.nan))[mask_C].astype(float)
    ori_C = cache_C_sw.get("lab_orientation", np.array(["pooled"] * cache_C_sw["Z"].shape[0]))[mask_C]
    
    C_labels_S = cache_S_sw.get("lab_C", np.full(cache_S_sw["Z"].shape[0], np.nan))[mask_S].astype(float)
    R_labels_S = cache_S_sw.get("lab_R", np.full(cache_S_sw["Z"].shape[0], np.nan))[mask_S].astype(float)
    ori_S = cache_S_sw.get("lab_orientation", np.array(["pooled"] * cache_S_sw["Z"].shape[0]))[mask_S]
    
    # Build condition-averaged activity using Z-scored SW-transformed data
    print(f"  [reverse] Building geometry from both epochs (Z-scored, SW-transformed)...")
    print(f"            C epoch: {align_C}, {orientation_C}, window={win_C}, trials={mask_C.sum()}")
    print(f"            S epoch: {align_S}, {orientation_S}, window={win_S}, trials={mask_S.sum()}")
    
    D_C = condition_average_activity(
        Z_C, C_labels_C, time_C, win_C,
        use_rich_cond=use_rich_cond,
        C=C_labels_C, S=S_labels_C, R=R_labels_C, orientation=ori_C
    )
    
    D_S = condition_average_activity(
        Z_S, S_labels_S, time_S, win_S,
        use_rich_cond=use_rich_cond,
        C=C_labels_S, S=S_labels_S, R=R_labels_S, orientation=ori_S
    )
    
    if D_C.size == 0 or D_S.size == 0:
        print(f"  [skip] Empty condition-averaged matrix")
        return None
    
    print(f"            D_C shape: {D_C.shape}, D_S shape: {D_S.shape}")
    
    # Concatenate and center (already Z-scored, no additional standardization needed)
    D_combined = np.hstack([D_C, D_S])
    D_combined_centered = D_combined - D_combined.mean(axis=1, keepdims=True)
    
    # For coordinate conversion, we still need a common reference
    sigma_ref = np.std(D_combined, axis=1)
    sigma_ref[~np.isfinite(sigma_ref) | (sigma_ref < 1e-8)] = 1.0
    
    # Convert axes to standardized space
    sC_std = convert_axis_to_standardized(sC_raw, sigma_ref)
    sS_std = convert_axis_to_standardized(sS_raw, sigma_ref)
    
    # Compute observed alignment
    a_obs = abs(np.dot(sC_std, sS_std))
    a_obs = min(1.0, max(0.0, a_obs))
    theta_obs = float(np.degrees(np.arccos(a_obs)))
    
    print(f"  [reverse] Observed: |cos(θ)|={a_obs:.4f}, θ={theta_obs:.1f}°")
    
    # Build covariance factor using S-epoch geometry with BOTH C and S conditions
    # Use the SAME SW-transformed data as the observed geometry
    print(f"  [null] Building S-epoch geometry with both C and S conditions...")
    print(f"         Using SAME SW-transformed S cache as observed geometry")
    
    # Build D_C from S epoch using Z-scored data (C labels, S time window)
    D_C_from_S_epoch_Z = condition_average_activity(
        Z_S, C_labels_S, time_S, win_S,
        use_rich_cond=use_rich_cond,
        C=C_labels_S, S=S_labels_S, R=R_labels_S, orientation=ori_S
    )
    
    # D_S is already computed above from the same cache
    # Combine: D_C and D_S from S epoch (z-scored)
    if D_C_from_S_epoch_Z.size > 0 and D_S.size > 0:
        D_S_combined_Z = np.hstack([D_C_from_S_epoch_Z, D_S])
        D_S_centered = D_S_combined_Z - D_S_combined_Z.mean(axis=1, keepdims=True)
        print(f"         D_S_combined shape: {D_S_combined_Z.shape}")
    else:
        # Fallback to D_S only
        D_S_centered = D_S - D_S.mean(axis=1, keepdims=True)
        print(f"         [warn] Using D_S only (D_C_from_S_epoch was empty)")
    
    U_cov, s_cov, _ = np.linalg.svd(D_S_centered, full_matrices=False)
    
    cumvar = np.cumsum(s_cov**2) / np.sum(s_cov**2)
    n_keep = min(np.searchsorted(cumvar, 0.99) + 1, len(s_cov))
    n_keep = max(n_keep, 2)
    
    cov_sqrt = U_cov[:, :n_keep] * s_cov[:n_keep]
    
    D_eff = (np.sum(s_cov**2)**2) / np.sum(s_cov**4)
    expected_cos = np.sqrt(2.0 / (np.pi * D_eff))
    expected_angle = np.degrees(np.arccos(expected_cos))
    
    print(f"  [null] Using S-epoch geometry (fixed-axis null)")
    print(f"  [null] Manifold dim: {n_keep}, D_eff={D_eff:.1f}")
    
    # Generate null distribution
    rng = np.random.default_rng(seed)
    null_a = np.empty(n_perms, dtype=float)
    null_angles = np.empty(n_perms, dtype=float)
    
    print(f"  [null] Running {n_perms} covariance-constrained null samples...")
    for b in range(n_perms):
        z = rng.standard_normal(n_keep)
        v = cov_sqrt @ z
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            null_a[b] = np.nan
            null_angles[b] = np.nan
        else:
            s_rand = v / norm
            null_a[b] = abs(np.dot(sC_std, s_rand))
            null_a[b] = min(1.0, max(0.0, null_a[b]))
            null_angles[b] = np.degrees(np.arccos(null_a[b]))
        
        if (b + 1) % 200 == 0:
            print(f"    [{b+1}/{n_perms}] null_mean={np.nanmean(null_a[:b+1]):.4f}")
    
    # Compute statistics
    null_mean = float(np.nanmean(null_a))
    null_std = float(np.nanstd(null_a))
    null_angle_mean = float(np.nanmean(null_angles))
    
    valid_null = null_a[np.isfinite(null_a)]
    n_valid = len(valid_null)
    
    p_orth = float((1 + np.sum(valid_null <= a_obs)) / (1 + n_valid))
    p_align = float((1 + np.sum(valid_null >= a_obs)) / (1 + n_valid))
    
    if null_std > 0:
        z_score = (a_obs - null_mean) / null_std
    else:
        z_score = 0.0
    
    delta = a_obs - null_mean
    delta_theta = theta_obs - null_angle_mean
    
    print(f"  [result] a_obs={a_obs:.4f}, null_mean={null_mean:.4f}±{null_std:.4f}")
    print(f"           Δ={delta:.4f}, z={z_score:.2f}")
    print(f"           p(more orthogonal)={p_orth:.4f}, p(more aligned)={p_align:.4f}")
    
    return {
        "sid": sid,
        "area": area,
        "monkey": get_monkey(sid),
        "analysis": "reverse_cross_alignment",
        "description": "Stim-C(horizontal) vs Sacc-S(vertical) - reverse of Case vi",
        "null_mode": "s_epoch",  # fixed-axis null using S-epoch geometry
        
        "align_C": align_C,
        "axes_tag_C": axes_tag_C,
        "orientation_C": orientation_C,
        "align_S": align_S,
        "axes_tag_S": axes_tag_S,
        "orientation_S": orientation_S,
        
        "a_obs": a_obs,
        "theta_obs_deg": theta_obs,
        
        "null_mean": null_mean,
        "null_std": null_std,
        "null_angle_mean_deg": null_angle_mean,
        
        "p_orth": p_orth,
        "p_align": p_align,
        "n_perms_valid": int(n_valid),
        
        "delta": delta,
        "delta_theta_deg": delta_theta,
        "z_score": z_score,
        
        "manifold_dim": n_keep,
        "D_eff": float(D_eff),
        "expected_cos": float(expected_cos),
        "expected_angle_deg": float(expected_angle),
        
        "D_C_shape": list(D_C.shape),
        "D_S_shape": list(D_S.shape),
        
        "win_C": list(win_C),
        "win_S": list(win_S),
        "pt_min_ms": pt_min_ms,
        "n_trials_C": int(mask_C.sum()),
        "n_trials_S": int(mask_S.sum()),
    }


def _convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(v) for v in obj]
    else:
        return obj


def save_result(result: Dict, out_dir: Path, tag: str):
    """Save result to JSON and NPZ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sid = result["sid"]
    
    json_result = _convert_to_json_serializable(result)
    
    with open(out_dir / f"{tag}_{sid}.json", "w") as f:
        json.dump(json_result, f, indent=2)
    
    print(f"  [saved] {out_dir / f'{tag}_{sid}.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Reverse case analysis: Stim-C(horiz) vs Sacc-S(vert)"
    )
    
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--sid", help="Single session ID")
    session_group.add_argument("--sid_list", help="File with session IDs")
    
    parser.add_argument("--out_root", default="out")
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pt_min_ms", type=float, default=200.0)
    parser.add_argument("--tag", default="reverse_case_correctonly")
    parser.add_argument("--axes_suffix", default="20mssw",
                        help="Axes tag suffix (e.g., 20mssw or 20mssw_nofilter)")
    
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    
    # Determine sessions
    if args.sid:
        sids = [args.sid]
    else:
        sid_list_path = Path(args.sid_list)
        if not sid_list_path.exists():
            raise SystemExit(f"Session list not found: {sid_list_path}")
        sids = []
        with open(sid_list_path) as f:
            for line in f:
                sid = line.strip()
                if sid and not sid.startswith("#"):
                    sids.append(sid)
    
    print("=" * 70)
    print("REVERSE CASE ANALYSIS: Stim-C(horiz) vs Sacc-S(vert)")
    print("=" * 70)
    print(f"[info] Processing {len(sids)} session(s)")
    print(f"[info] Axes suffix: {args.axes_suffix}")
    
    # Output always goes to out/vi/, not out_root (which may be out_nofilter)
    project_root = out_root.parent if out_root.name in ("out", "out_nofilter") else out_root
    results_dir = project_root / "out" / "vi" / args.tag
    
    for sid in sids:
        print(f"\n{'=' * 60}")
        print(f"[{sid}] Processing...")
        
        result = analyze_reverse_case(
            out_root=out_root,
            axes_suffix=args.axes_suffix,
            sid=sid,
            n_perms=args.n_perms,
            seed=args.seed,
            pt_min_ms=args.pt_min_ms,
        )
        
        if result is not None:
            save_result(result, results_dir, args.tag)
    
    print(f"\n{'=' * 70}")
    print(f"[done] Results saved to {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
