#!/usr/bin/env python3
"""
Cross-Alignment Axis Analysis for Cases iii and iv.

Compares C axis from stim-aligned cache with S axis from sacc-aligned cache.
Must handle coordinate system alignment since axes come from different z-scorings.

CASES:
  Case iii: C from stim-aligned (vertical) vs S from sacc-aligned (horizontal)
  Case iv:  C from stim-aligned (pooled) vs S from sacc-aligned (pooled)

MATHEMATICAL FRAMEWORK:
  1. Load axes from different caches (stim-aligned C, sacc-aligned S)
  2. Convert axes to raw-unit space: w_raw = w_Z / σ_cache
  3. Build geometry matrix D from both epochs in raw X space
  4. Apply common standardization: D_std = D / σ_ref
  5. Convert axes to standardized space: w_std = w_raw * σ_ref
  6. Compute observed alignment and covariance-constrained null

COORDINATE SYSTEM ALIGNMENT:
  - Each cache has its own z-scoring: Z_u = (X_u - μ_u) / σ_u
  - A direction w_Z in Z-space corresponds to w_X = w_Z / σ in X-space
  - We use a common standardized space derived from the combined geometry
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Utility functions
# =============================================================================

def load_npz(p: Path) -> Dict:
    """Load NPZ file and parse meta if present."""
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta" in out and not isinstance(out["meta"], dict):
        try:
            out["meta"] = json.loads(out["meta"].item())
        except Exception:
            pass
    return out


def get_monkey(sid: str) -> str:
    """Return 'M' for sessions starting with 2020, 'S' for 2023."""
    if sid.startswith("2020"):
        return "M"
    elif sid.startswith("2023"):
        return "S"
    return "Unknown"


def get_fef_area(sid: str) -> str:
    """Return FEF area name based on monkey."""
    return "MFEF" if get_monkey(sid) == "M" else "SFEF"


def unit_vec(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        return np.zeros_like(v)
    return v / n


def window_mask(time_s: np.ndarray, win: Tuple[float, float]) -> np.ndarray:
    """Return boolean mask for time bins within window."""
    return (time_s >= win[0]) & (time_s <= win[1])


# =============================================================================
# Coordinate conversion
# =============================================================================

def get_cache_sigma(cache: Dict, axes_meta: Dict) -> np.ndarray:
    """
    Get the per-unit standard deviation used for z-scoring in this cache.
    
    If baseline normalization was used, returns norm_sd from axes.
    Otherwise, computes σ from the raw X data.
    """
    norm_mode = axes_meta.get("norm", "global") if axes_meta else "global"
    
    if norm_mode == "baseline":
        # Use stored normalization parameters from axes
        norm_sd = axes_meta.get("norm_sd")
        if norm_sd is not None and len(norm_sd) > 0:
            sd = np.asarray(norm_sd, dtype=np.float64)
            sd[sd < 1e-10] = 1.0
            return sd
    
    # Global normalization: compute σ from cache X or Z
    if "X" in cache:
        X = cache["X"]  # (trials, bins, units)
        Xf = X.reshape(-1, X.shape[-1]).astype(np.float64)
        sd = np.nanstd(Xf, axis=0, ddof=1)
    elif "Z" in cache:
        # If only Z is available, estimate σ from Z (should be ~1, but may vary)
        Z = cache["Z"]
        Zf = Z.reshape(-1, Z.shape[-1]).astype(np.float64)
        sd = np.nanstd(Zf, axis=0, ddof=1)
    else:
        raise KeyError("Cache missing both 'X' and 'Z'")
    
    sd[~np.isfinite(sd) | (sd < 1e-10)] = 1.0
    return sd


def convert_axis_to_raw(
    axis_Z: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Convert an axis from Z-space to raw X-space.
    
    If Z = (X - μ) / σ, and w_Z is a direction in Z-space,
    then w_X = w_Z / σ gives the same direction in X-space.
    (The intercept doesn't matter for directions.)
    """
    return axis_Z / sigma


def convert_axis_to_standardized(
    axis_raw: np.ndarray,
    sigma_ref: np.ndarray,
) -> np.ndarray:
    """
    Convert an axis from raw X-space to standardized space.
    
    If Y = X / σ_ref, then a direction w_X in X-space
    corresponds to w_Y = w_X * σ_ref in Y-space.
    """
    return unit_vec(axis_raw * sigma_ref)


# =============================================================================
# Trial mask and condition averaging
# =============================================================================

def trial_mask(
    cache: Dict, 
    orientation: str, 
    pt_min_ms: Optional[float],
) -> np.ndarray:
    """Build trial mask matching existing pipeline filters."""
    N = cache["Z"].shape[0]
    keep = np.ones(N, dtype=bool)
    
    # Note: We do NOT filter by lab_is_correct here because:
    # - For out/: caches only contain correct trials
    # - For out_nofilter/: all trials are marked correct (intentional design)
    
    # Orientation filter
    if orientation != "pooled" and "lab_orientation" in cache:
        keep &= (cache["lab_orientation"].astype(str) == orientation)
    
    # PT filter
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))
    
    return keep


def build_rich_cond_id(
    C: np.ndarray,
    S: np.ndarray,
    R: np.ndarray,
    orientation: np.ndarray,
) -> np.ndarray:
    """
    Build a rich condition ID for covariance estimation.
    
    Combines: sign(C) × sign(S) × R_binned × orientation
    Gives ~24 conditions vs ~4 for simple C×S.
    
    Returns condition IDs (0, 1, 2, ...) or -1 for invalid trials.
    """
    N = len(C)
    
    # Encode sign(C): 0 or 1
    C_sign = np.zeros(N, dtype=int)
    C_sign[np.sign(C) > 0] = 1
    
    # Encode sign(S): 0 or 2
    S_sign = np.zeros(N, dtype=int)
    S_sign[np.sign(S) > 0] = 2
    
    # Encode R (bin into 3 levels): 0, 4, 8
    R_binned = np.zeros(N, dtype=int)
    R_valid = np.isfinite(R)
    if R_valid.sum() > 10:
        R_percentiles = np.percentile(R[R_valid], [33, 67])
        R_binned[R_valid] = np.digitize(R[R_valid], R_percentiles) * 4
    
    # Encode orientation: 0 or 12
    ori_numeric = np.zeros(N, dtype=int)
    ori_str = np.asarray(orientation).astype(str)
    ori_numeric[ori_str == "vertical"] = 12
    
    cond_id = C_sign + S_sign + R_binned + ori_numeric
    
    # Mark invalid trials
    invalid = ~np.isfinite(C) | ~np.isfinite(S)
    cond_id[invalid] = -1
    
    return cond_id


def condition_average_activity(
    X: np.ndarray,           # (N_trials, B, U) - raw or standardized
    labels: np.ndarray,      # (N_trials,) condition labels
    time_s: np.ndarray,      # (B,) time axis
    window: Tuple[float, float],
    use_rich_cond: bool = False,
    C: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    orientation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute condition-averaged activity matrix for a time window.
    
    Returns:
        D: (U, n_conditions * n_time) condition-averaged activity
    """
    mask_t = window_mask(time_s, window)
    if not np.any(mask_t):
        return np.array([])
    
    # Determine condition IDs
    if use_rich_cond and C is not None and S is not None:
        cond_ids = build_rich_cond_id(C, S, R, orientation)
    else:
        # Simple: use provided labels
        valid = np.isfinite(labels)
        cond_ids = np.full(len(labels), -1, dtype=int)
        unique_labels = np.unique(labels[valid])
        for i, lab in enumerate(unique_labels):
            cond_ids[labels == lab] = i
    
    # Get unique valid conditions
    valid_conds = np.unique(cond_ids[cond_ids >= 0])
    
    if len(valid_conds) == 0:
        return np.array([])
    
    # Build condition-averaged matrix
    n_time = mask_t.sum()
    n_units = X.shape[2]
    D_list = []
    
    for cond in valid_conds:
        trial_mask = (cond_ids == cond)
        if trial_mask.sum() > 0:
            # Mean across trials, shape (n_time, n_units)
            mean_act = np.nanmean(X[trial_mask][:, mask_t, :], axis=0)
            D_list.append(mean_act)  # (n_time, U)
    
    if not D_list:
        return np.array([])
    
    # Stack: (n_conditions * n_time, U) then transpose to (U, n_conditions * n_time)
    D = np.vstack(D_list).T
    return D


# =============================================================================
# Cross-alignment analysis
# =============================================================================

def analyze_cross_alignment(
    out_root: Path,
    sid: str,
    # C axis settings
    axes_tag_C: str,
    align_C: str,           # "stim" or "sacc"
    orientation_C: str,
    win_C: Tuple[float, float],
    # S axis settings
    axes_tag_S: str,
    align_S: str,           # typically "sacc"
    orientation_S: str,
    win_S: Tuple[float, float],
    # Common settings
    pt_min_ms: float,
    n_perms: int,
    seed: int,
    use_rich_cond: bool = True,
    # QC thresholds (optional)
    qc_threshold_C: float = 0.0,
    qc_threshold_S: float = 0.0,
) -> Optional[Dict]:
    """
    Cross-alignment analysis: C from one alignment vs S from another.
    
    This handles the coordinate system alignment needed when axes
    come from different caches with different z-scorings.
    """
    area = get_fef_area(sid)
    
    # --- Load caches ---
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
    
    # --- Load axes ---
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
    
    # --- Get meta for normalization info ---
    meta_C = axes_C.get("meta", {})
    meta_S = axes_S.get("meta", {})
    if isinstance(meta_C, str):
        meta_C = json.loads(meta_C)
    if isinstance(meta_S, str):
        meta_S = json.loads(meta_S)
    
    # --- Convert axes to raw space ---
    sigma_C = get_cache_sigma(cache_C, meta_C)
    sigma_S = get_cache_sigma(cache_S, meta_S)
    
    sC_raw = convert_axis_to_raw(sC_Z, sigma_C)
    sS_raw = convert_axis_to_raw(sS_Z, sigma_S)
    
    # --- Build trial masks ---
    mask_C = trial_mask(cache_C, orientation_C, pt_min_ms)
    mask_S = trial_mask(cache_S, orientation_S, pt_min_ms)
    
    if mask_C.sum() < 40:
        print(f"  [skip] Not enough trials for C: {mask_C.sum()}")
        return None
    if mask_S.sum() < 40:
        print(f"  [skip] Not enough trials for S: {mask_S.sum()}")
        return None
    
    # --- Get raw X data and labels ---
    time_C = cache_C["time"].astype(float)
    time_S = cache_S["time"].astype(float)
    
    # Use X if available, otherwise Z (less ideal for cross-cache comparison)
    if "X" in cache_C and "X" in cache_S:
        X_C = cache_C["X"][mask_C].astype(np.float64)
        X_S = cache_S["X"][mask_S].astype(np.float64)
    else:
        print(f"  [warn] Using Z instead of X for geometry (less ideal)")
        X_C = cache_C["Z"][mask_C].astype(np.float64) * sigma_C
        X_S = cache_S["Z"][mask_S].astype(np.float64) * sigma_S
    
    # Labels for condition averaging
    C_labels_C = cache_C.get("lab_C", np.full(cache_C["Z"].shape[0], np.nan))[mask_C].astype(float)
    S_labels_S = cache_S.get("lab_S", np.full(cache_S["Z"].shape[0], np.nan))[mask_S].astype(float)
    
    # Additional labels for rich condition IDs
    S_labels_C = cache_C.get("lab_S", np.full(cache_C["Z"].shape[0], np.nan))[mask_C].astype(float)
    R_labels_C = cache_C.get("lab_R", np.full(cache_C["Z"].shape[0], np.nan))[mask_C].astype(float)
    ori_C = cache_C.get("lab_orientation", np.array(["pooled"] * cache_C["Z"].shape[0]))[mask_C]
    
    C_labels_S = cache_S.get("lab_C", np.full(cache_S["Z"].shape[0], np.nan))[mask_S].astype(float)
    R_labels_S = cache_S.get("lab_R", np.full(cache_S["Z"].shape[0], np.nan))[mask_S].astype(float)
    ori_S = cache_S.get("lab_orientation", np.array(["pooled"] * cache_S["Z"].shape[0]))[mask_S]
    
    # --- Build condition-averaged activity in raw X space ---
    print(f"  [cross] Building geometry from both epochs...")
    print(f"          C epoch: {align_C}, window={win_C}, trials={mask_C.sum()}")
    print(f"          S epoch: {align_S}, window={win_S}, trials={mask_S.sum()}")
    
    D_C = condition_average_activity(
        X_C, C_labels_C, time_C, win_C,
        use_rich_cond=use_rich_cond,
        C=C_labels_C, S=S_labels_C, R=R_labels_C, orientation=ori_C
    )
    
    D_S = condition_average_activity(
        X_S, S_labels_S, time_S, win_S,
        use_rich_cond=use_rich_cond,
        C=C_labels_S, S=S_labels_S, R=R_labels_S, orientation=ori_S
    )
    
    if D_C.size == 0 or D_S.size == 0:
        print(f"  [skip] Empty condition-averaged matrix")
        return None
    
    print(f"          D_C shape: {D_C.shape}, D_S shape: {D_S.shape}")
    
    # --- Concatenate and compute common standardization ---
    D_combined = np.hstack([D_C, D_S])
    
    # Per-unit std from combined data
    sigma_ref = np.std(D_combined, axis=1)
    sigma_ref[~np.isfinite(sigma_ref) | (sigma_ref < 1e-8)] = 1.0
    
    # Standardize the geometry matrix
    D_std = D_combined / sigma_ref[:, None]
    
    # --- Convert axes to standardized space ---
    sC_std = convert_axis_to_standardized(sC_raw, sigma_ref)
    sS_std = convert_axis_to_standardized(sS_raw, sigma_ref)
    
    # --- Compute observed alignment ---
    a_obs = abs(np.dot(sC_std, sS_std))
    a_obs = min(1.0, max(0.0, a_obs))
    theta_obs = float(np.degrees(np.arccos(a_obs)))
    
    print(f"  [cross] Observed: |cos(θ)|={a_obs:.4f}, θ={theta_obs:.1f}°")
    
    # --- Build covariance factor ---
    D_centered = D_std - D_std.mean(axis=1, keepdims=True)
    U_cov, s_cov, _ = np.linalg.svd(D_centered, full_matrices=False)
    
    # Keep components explaining 99% variance
    cumvar = np.cumsum(s_cov**2) / np.sum(s_cov**2)
    n_keep = min(np.searchsorted(cumvar, 0.99) + 1, len(s_cov))
    n_keep = max(n_keep, 2)
    
    # Covariance factor: L = U @ diag(s), so LL.T ∝ Σ
    cov_sqrt = U_cov[:, :n_keep] * s_cov[:n_keep]  # NOT sqrt!
    
    # Effective dimensionality
    D_eff = (np.sum(s_cov**2)**2) / np.sum(s_cov**4)
    expected_cos = np.sqrt(2.0 / (np.pi * D_eff))
    expected_angle = np.degrees(np.arccos(expected_cos))
    
    print(f"  [null] Manifold dim: {n_keep}, D_eff={D_eff:.1f}")
    print(f"         Expected |cos(θ)| ≈ {expected_cos:.3f} (angle ≈ {expected_angle:.1f}°)")
    
    # --- Generate null distribution ---
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
    
    # --- Compute statistics ---
    null_mean = float(np.nanmean(null_a))
    null_std = float(np.nanstd(null_a))
    null_angle_mean = float(np.nanmean(null_angles))
    
    valid_null = null_a[np.isfinite(null_a)]
    n_valid = len(valid_null)
    
    # P-values
    p_orth = float((1 + np.sum(valid_null <= a_obs)) / (1 + n_valid))
    p_align = float((1 + np.sum(valid_null >= a_obs)) / (1 + n_valid))
    
    # Effect size
    if null_std > 0:
        z_score = (a_obs - null_mean) / null_std
    else:
        z_score = 0.0
    
    delta = a_obs - null_mean
    delta_theta = theta_obs - null_angle_mean
    
    print(f"  [result] a_obs={a_obs:.4f}, null_mean={null_mean:.4f}±{null_std:.4f}")
    print(f"           Δ={delta:.4f}, z={z_score:.2f}")
    print(f"           θ_obs={theta_obs:.1f}°, θ_null={null_angle_mean:.1f}°")
    print(f"           p(more orthogonal)={p_orth:.4f}, p(more aligned)={p_align:.4f}")
    
    if p_orth < 0.05:
        print(f"  [**] SIGNIFICANT: Cross-alignment C-S is LOWER than geometry-constrained chance!")
    
    return {
        "sid": sid,
        "area": area,
        "monkey": get_monkey(sid),
        "analysis": "cross_alignment",
        
        # Sources
        "align_C": align_C,
        "axes_tag_C": axes_tag_C,
        "orientation_C": orientation_C,
        "align_S": align_S,
        "axes_tag_S": axes_tag_S,
        "orientation_S": orientation_S,
        
        # Observed
        "a_obs": a_obs,
        "theta_obs_deg": theta_obs,
        
        # Null distribution
        "null_mean": null_mean,
        "null_std": null_std,
        "null_a": null_a,
        "null_angle_mean_deg": null_angle_mean,
        
        # P-values
        "p_orth": p_orth,
        "p_align": p_align,
        "n_perms_valid": int(n_valid),
        
        # Effect size
        "delta": delta,
        "delta_theta_deg": delta_theta,
        "z_score": z_score,
        
        # Manifold info
        "manifold_dim": n_keep,
        "D_eff": float(D_eff),
        "expected_cos": float(expected_cos),
        "expected_angle_deg": float(expected_angle),
        
        # Geometry info
        "D_C_shape": list(D_C.shape),
        "D_S_shape": list(D_S.shape),
        "use_rich_cond": use_rich_cond,
        
        # Settings
        "win_C": list(win_C),
        "win_S": list(win_S),
        "pt_min_ms": pt_min_ms,
        "n_trials_C": int(mask_C.sum()),
        "n_trials_S": int(mask_S.sum()),
    }


# =============================================================================
# Save results
# =============================================================================

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
    
    # Save JSON (without large arrays)
    json_result = {k: _convert_to_json_serializable(v) 
                   for k, v in result.items() 
                   if k not in ["null_a"]}
    
    with open(out_dir / f"cross_{tag}_{sid}.json", "w") as f:
        json.dump(json_result, f, indent=2)
    
    # Save NPZ (with null distribution)
    npz_data = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            npz_data[k] = v
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            npz_data[k] = np.array(v)
        else:
            npz_data[k] = np.array([v])
    
    np.savez_compressed(out_dir / f"cross_{tag}_{sid}.npz", **npz_data)
    
    print(f"  [saved] {out_dir / f'cross_{tag}_{sid}.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="""
Cross-alignment axis analysis for cases iii and iv.

Compares C axis from stim-aligned cache with S axis from sacc-aligned cache,
properly handling coordinate system alignment.

Case iii: C from stim (vertical) vs S from sacc (horizontal)
Case iv:  C from stim (pooled) vs S from sacc (pooled)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Session
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--sid", help="Single session ID")
    session_group.add_argument("--sid_list", help="File with list of session IDs")
    
    # Paths
    parser.add_argument("--out_root", default="out", help="Output root directory")
    
    # C axis settings (typically stim-aligned)
    parser.add_argument("--align_C", default="stim", choices=["stim", "sacc"],
                        help="Alignment for C axis")
    parser.add_argument("--axes_tag_C", required=True,
                        help="Axes tag for C (e.g., axes_peakbin_C-stim-vertical-20mssw)")
    parser.add_argument("--orientation_C", default="vertical",
                        help="Orientation filter for C")
    parser.add_argument("--win_C", nargs=2, type=float, default=[0.10, 0.30],
                        help="Window for C epoch [start, end] in seconds")
    
    # S axis settings (typically sacc-aligned)
    parser.add_argument("--align_S", default="sacc", choices=["stim", "sacc"],
                        help="Alignment for S axis")
    parser.add_argument("--axes_tag_S", required=True,
                        help="Axes tag for S (e.g., axes_peakbin_saccCS-sacc-horizontal-20mssw)")
    parser.add_argument("--orientation_S", default="horizontal",
                        help="Orientation filter for S")
    parser.add_argument("--win_S", nargs=2, type=float, default=[-0.10, -0.03],
                        help="Window for S epoch [start, end] in seconds")
    
    # Common settings
    parser.add_argument("--pt_min_ms", type=float, default=200.0)
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_rich_cond", action="store_true", default=False,
                        help="Use rich condition IDs for geometry estimation")
    parser.add_argument("--no_rich_cond", action="store_true", default=True,
                        help="Disable rich condition IDs (default: True, use simple C/S labels)")
    
    # QC thresholds
    parser.add_argument("--qc_threshold_C", type=float, default=0.0)
    parser.add_argument("--qc_threshold_S", type=float, default=0.0)
    
    # Output
    parser.add_argument("--tag", default="cross",
                        help="Output tag for result files")
    
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    use_rich_cond = args.use_rich_cond and not args.no_rich_cond
    
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
    
    print("="*70)
    print("CROSS-ALIGNMENT AXIS ANALYSIS")
    print("="*70)
    print(f"[info] Processing {len(sids)} session(s)")
    print(f"[info] C: {args.align_C}/{args.axes_tag_C} ({args.orientation_C})")
    print(f"[info] S: {args.align_S}/{args.axes_tag_S} ({args.orientation_S})")
    print(f"[info] win_C: {args.win_C}, win_S: {args.win_S}")
    print(f"[info] Use rich conditions: {use_rich_cond}")
    
    # Output directory
    results_dir = out_root / "cross_alignment" / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    for sid in sids:
        print(f"\n{'='*60}")
        print(f"[{sid}] Processing cross-alignment...")
        
        result = analyze_cross_alignment(
            out_root=out_root,
            sid=sid,
            axes_tag_C=args.axes_tag_C,
            align_C=args.align_C,
            orientation_C=args.orientation_C,
            win_C=tuple(args.win_C),
            axes_tag_S=args.axes_tag_S,
            align_S=args.align_S,
            orientation_S=args.orientation_S,
            win_S=tuple(args.win_S),
            pt_min_ms=args.pt_min_ms,
            n_perms=args.n_perms,
            seed=args.seed,
            use_rich_cond=use_rich_cond,
            qc_threshold_C=args.qc_threshold_C,
            qc_threshold_S=args.qc_threshold_S,
        )
        
        if result is not None:
            save_result(result, results_dir, args.tag)
    
    print(f"\n{'='*70}")
    print(f"[done] Results saved to {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
