#!/usr/bin/env python3
"""
Axis Alignment Analysis: Category vs Saccade subspace alignment in FEF

This script tests Dave's hypothesis: Do category (C) and saccade (S) encoding
subspaces in FEF coexist in near-orthogonal (independent) subspaces?

THREE COMPLEMENTARY ANALYSES:

=============================================================================
ANALYSIS A: Alignment Index + Covariance-Matched Null (SC Nature Neuro style)
=============================================================================
Tests "less aligned than chance" for PCA-derived SUBSPACES (k-dimensional).

Key insight: In high dimensions, random vectors are nearly orthogonal, but neural
data lives on a low-dimensional manifold. "Chance" alignment must respect the
population covariance—otherwise we overstate "special orthogonality."

Method:
  1. Define category and saccade subspaces from two epochs (PCA on condition-averaged activity)
  2. Compute Alignment Index: AI = tr(U_S' Σ_C U_S) / tr(U_C' Σ_C U_C)
     - AI ≈ 1: saccade subspace captures as much category variance as category PCs
     - AI << 1: saccade subspace captures little category variance (low alignment)
  3. Null: sample random subspaces CONSTRAINED BY the population covariance
  4. P-value: P(AI_null ≤ AI_obs) tests "more orthogonal than chance"

This addresses: "Given the manifold constraints, is PCA subspace alignment unusually low?"

=============================================================================
ANALYSIS B: Axis Angle + Label-Shuffle Null (PNAS style)
=============================================================================
Tests whether C-S axis alignment is LABEL-SPECIFIC.

Method:
  1. Use pre-trained axes: sC (category) and sS_raw (saccade, NOT orthogonalized)
  2. Compute acute angle: θ = arccos(|sC·sS|)
  3. Null: shuffle S labels within (C,R) strata, retrain sS, recompute angle
  4. P-value: P(θ_null ≥ θ_obs) tests "more aligned than shuffle-chance"

This answers: "Is the C-S relationship label-specific?"

NOTE: This null destroys the label mapping, so it tests "more aligned than chance,"
NOT "less aligned." It's a sanity check, not the orthogonality test.

=============================================================================
ANALYSIS C: Axis Angle + Covariance-Constrained Null (DAVE'S QUESTION)
=============================================================================
Tests "less aligned than chance" for trained 1D AXES using geometry-constrained null.

This is the proper test for Dave's question about TRAINED AXES (not PCA subspaces).
Uses the same covariance-constrained logic as Analysis A, but for single axes.

Method:
  1. Use pre-trained axes: sC (category) and sS_raw (saccade, NOT orthogonalized)
  2. Compute acute angle: a_obs = |sC·sS|
  3. Build manifold covariance from combined C+S epoch activity
  4. Null: sample random AXES constrained by manifold covariance
     - s_rand = cov_sqrt @ z / ||cov_sqrt @ z||, where z ~ N(0, I)
  5. P-value: P(a_null ≤ a_obs) tests "more orthogonal than geometry-constrained chance"

Key interpretation:
  - If p_orth is small: "Given the neural manifold geometry, C-S axis alignment
    is LOWER than expected → supports separation into distinct subspaces"
  - Expected chance alignment depends on effective dimensionality D_eff:
    E[|cos(θ)|] ≈ sqrt(2/(π * D_eff))

=============================================================================
USAGE MODES:
  - Single session: python cli/axis_alignment_analysis.py --sid 20200101 ...
  - All sessions: python cli/axis_alignment_analysis.py --sid_list sid_list.txt ...
  
MODES:
  --mode AI          : Analysis A only (PCA subspaces)
  --mode axis_shuffle: Analysis B only (label-shuffle)
  --mode axis_cov    : Analysis C only (axis + covariance null) - DAVE'S QUESTION
  --mode both        : A + B (original)
  --mode all         : A + B + C (recommended)
=============================================================================
"""
from __future__ import annotations
import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperflow.axes import (
    cv_fit_binary_linear, unit_vec, window_mask, avg_over_window
)
from paperflow.norm import get_Z, rebin_cache_data, sliding_window_cache_data


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


def trial_mask(cache: Dict, orientation: str, pt_min_ms: Optional[float]) -> np.ndarray:
    """Build trial mask matching existing pipeline filters."""
    N = cache["Z"].shape[0]
    keep = np.ones(N, dtype=bool)
    
    # Correct trials only
    keep &= cache.get("lab_is_correct", np.ones(N, dtype=bool)).astype(bool)
    
    # Orientation filter
    if orientation != "pooled" and "lab_orientation" in cache:
        keep &= (cache["lab_orientation"].astype(str) == orientation)
    
    # PT filter
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))
    
    return keep


# =============================================================================
# ANALYSIS A: Alignment Index with Covariance-Matched Null
# =============================================================================

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
    
    This provides a much more stable manifold geometry estimate than
    using just 2 conditions per epoch (C=±1 or S=±1).
    
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
    Z: np.ndarray,           # (N_trials, B, U)
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
    
    Args:
        Z: Neural data (N_trials, B, U)
        labels: Simple condition labels (N_trials,) for basic averaging
        time_s: Time array (B,)
        window: Time window tuple (start, end) in seconds
        use_rich_cond: If True, use rich condition IDs for better manifold estimation
        C, S, R, orientation: Additional labels for rich condition IDs
    
    Returns:
        D: (U, n_conditions * n_time) condition-averaged activity
    """
    mask_t = (time_s >= window[0]) & (time_s <= window[1])
    if not np.any(mask_t):
        return np.array([])
    
    # Determine condition IDs
    if use_rich_cond and C is not None and S is not None:
        R_use = R if R is not None else np.full(len(C), np.nan)
        ori_use = orientation if orientation is not None else np.array(["pooled"] * len(C))
        cond_ids = build_rich_cond_id(C, S, R_use, ori_use)
        valid_conds = np.unique(cond_ids[cond_ids >= 0])
    else:
        # Simple: use provided labels
        valid = np.isfinite(labels)
        unique_labels = np.unique(labels[valid])
        cond_ids = np.full(len(labels), -1, dtype=int)
        for i, lab in enumerate(unique_labels):
            cond_ids[labels == lab] = i
        valid_conds = np.arange(len(unique_labels))
    
    if len(valid_conds) == 0:
        return np.array([])
    
    # Build condition-averaged matrix
    n_time = mask_t.sum()
    n_units = Z.shape[2]
    D_list = []
    
    for cond in valid_conds:
        trial_mask = (cond_ids == cond)
        if trial_mask.sum() > 0:
            # Mean across trials, shape (n_time, n_units)
            mean_act = np.nanmean(Z[trial_mask][:, mask_t, :], axis=0)
            D_list.append(mean_act)  # (n_time, U)
    
    if not D_list:
        return np.array([])
    
    # Stack: (n_conditions * n_time, U) then transpose to (U, n_conditions * n_time)
    D = np.vstack(D_list).T
    return D


def compute_pca_subspace(D: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute top-k PCA subspace of data matrix.
    
    Args:
        D: (U, n_samples) data matrix (units x time/conditions)
        k: number of PCs to keep
    
    Returns:
        U_k: (U, k) orthonormal basis for top-k subspace
        s: (k,) singular values
        var_explained: fraction of variance explained by top-k
    """
    # Center
    D_centered = D - D.mean(axis=1, keepdims=True)
    
    # SVD
    U, s, Vh = np.linalg.svd(D_centered, full_matrices=False)
    
    # Keep top k
    k = min(k, len(s))
    U_k = U[:, :k]
    s_k = s[:k]
    
    # Variance explained
    total_var = np.sum(s**2)
    var_explained = np.sum(s_k**2) / total_var if total_var > 0 else 0.0
    
    return U_k, s_k, var_explained


def compute_alignment_index(
    U_test: np.ndarray,   # (U, k) subspace to test
    Sigma: np.ndarray,    # (U, U) covariance matrix
    U_ref: np.ndarray,    # (U, k) reference subspace
) -> float:
    """
    Compute Alignment Index (SC Nature Neuro style).
    
    AI = tr(U_test' Σ U_test) / tr(U_ref' Σ U_ref)
    
    Measures how much variance in Σ is captured by U_test relative to U_ref.
    
    AI ≈ 1: U_test captures as much variance as U_ref
    AI << 1: U_test captures much less variance than U_ref
    """
    var_test = np.trace(U_test.T @ Sigma @ U_test)
    var_ref = np.trace(U_ref.T @ Sigma @ U_ref)
    
    if var_ref <= 0:
        return np.nan
    
    return float(var_test / var_ref)


def compute_principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces.
    
    Args:
        U1, U2: (U, k) orthonormal bases
    
    Returns:
        angles: (min(k1, k2),) principal angles in radians
    """
    # SVD of U1' @ U2 gives cos of principal angles
    M = U1.T @ U2
    _, s, _ = np.linalg.svd(M)
    
    # Clamp for numerical stability
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    
    return angles


def compute_subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute normalized subspace overlap.
    
    overlap = (1/k) * ||U1' U2||_F^2 = (1/k) * Σ cos²(φ_i)
    
    where φ_i are the principal angles.
    
    overlap = 1: subspaces are identical
    overlap = 0: subspaces are orthogonal
    overlap = k/N: expected for random subspaces in isotropic N-dim space
    """
    k = min(U1.shape[1], U2.shape[1])
    M = U1.T @ U2
    overlap = np.sum(M**2) / k
    return float(overlap)


def sample_covariance_matched_subspace(
    cov_sqrt: np.ndarray,  # (U, r) matrix such that cov_sqrt @ cov_sqrt.T ≈ Σ
    k: int,                # subspace dimension
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a random k-dimensional subspace constrained by the covariance structure.
    
    This is the key to the SC-style null: random subspaces that respect the
    neural manifold geometry, not uniform random in the full N-dim space.
    
    Method: 
      1. Draw random Gaussian vectors
      2. Project through sqrt(covariance) to constrain to manifold
      3. Orthonormalize
    
    Args:
        cov_sqrt: (U, r) factor such that Σ ≈ cov_sqrt @ cov_sqrt.T
                  Typically from SVD: cov_sqrt = U @ diag(sqrt(s))
        k: dimensionality of subspace to sample
        rng: random generator
    
    Returns:
        U_rand: (U, k) orthonormal basis for random covariance-matched subspace
    """
    U_dim = cov_sqrt.shape[0]
    r = cov_sqrt.shape[1]
    
    # Random Gaussian vectors in the reduced space
    V_rand = rng.standard_normal((r, k))
    
    # Project through covariance structure
    V_proj = cov_sqrt @ V_rand  # (U, k)
    
    # Orthonormalize via QR
    Q, R = np.linalg.qr(V_proj)
    
    return Q[:, :k]


def analyze_session_alignment_index(
    out_root: Path,
    sid: str,
    orientation: str,
    pt_min_ms: float,
    win_C: Tuple[float, float],    # Category epoch window (e.g., sample period)
    win_S: Tuple[float, float],    # Saccade epoch window (e.g., pre-saccade)
    k: int,                        # Subspace dimensionality
    n_perms: int,
    seed: int,
    sliding_window_bins: int,
    sliding_step_bins: int,
    qc_threshold_C: float,
    qc_threshold_S: float,
    axes_tag: Optional[str] = None,  # For QC check
) -> Optional[Dict]:
    """
    ANALYSIS A: Alignment Index with Covariance-Matched Null.
    
    Tests whether category and saccade subspaces are less aligned than expected
    given the neural manifold constraints.
    
    This is the SC Nature Neuroscience approach adapted for within-task comparison.
    """
    area = get_fef_area(sid)
    
    # Load sacc-aligned cache (both epochs from same alignment)
    cache_path = out_root / "sacc" / sid / "caches" / f"area_{area}.npz"
    
    if not cache_path.exists():
        print(f"  [skip] No cache: {cache_path}")
        return None
    
    cache = load_npz(cache_path)
    
    # Apply sliding window if needed
    if sliding_window_bins > 0 and sliding_step_bins > 0:
        cache, _ = sliding_window_cache_data(cache, sliding_window_bins, sliding_step_bins)
    
    # Build trial mask
    keep = trial_mask(cache, orientation, pt_min_ms)
    
    # Get data and labels
    Z = cache["Z"][keep].astype(float)  # (N_trials, B, U)
    time_s = cache["time"].astype(float)
    
    C_labels = cache.get("lab_C", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    S_labels = cache.get("lab_S", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    
    # Check valid trials
    valid_C = np.isfinite(C_labels)
    valid_S = np.isfinite(S_labels)
    
    if valid_C.sum() < 40 or valid_S.sum() < 40:
        print(f"  [skip] Not enough valid trials: C={valid_C.sum()}, S={valid_S.sum()}")
        return None
    
    # Build condition-averaged activity matrices
    print(f"  [AI] Building condition-averaged matrices...")
    print(f"       Category window: {win_C}")
    print(f"       Saccade window: {win_S}")
    
    # Category epoch: condition on C labels
    D_C = condition_average_activity(Z, C_labels, time_s, win_C)
    
    # Saccade epoch: condition on S labels  
    D_S = condition_average_activity(Z, S_labels, time_s, win_S)
    
    if D_C.size == 0 or D_S.size == 0:
        print(f"  [skip] Empty condition-averaged matrix")
        return None
    
    print(f"       D_C shape: {D_C.shape}, D_S shape: {D_S.shape}")
    
    # Compute PCA subspaces
    # Note: k should be <= min(n_units, n_conditions*n_time) for each epoch
    k_actual = min(k, D_C.shape[0], D_C.shape[1], D_S.shape[0], D_S.shape[1])
    if k_actual < k:
        print(f"  [warn] Reducing k from {k} to {k_actual} (data rank limited)")
        k = k_actual
    
    U_C, s_C, var_C = compute_pca_subspace(D_C, k)
    U_S, s_S, var_S = compute_pca_subspace(D_S, k)
    
    # Verify subspaces are valid
    if U_C.shape[1] < k or U_S.shape[1] < k:
        print(f"  [skip] Could not extract {k}-dim subspaces: U_C={U_C.shape}, U_S={U_S.shape}")
        return None
    
    print(f"       Category subspace: {U_C.shape}, var_explained={var_C:.3f}")
    print(f"       Saccade subspace: {U_S.shape}, var_explained={var_S:.3f}")
    
    # Compute covariance of category-epoch data
    # Σ_C = (1/n) * Σ_i (x_i - μ)(x_i - μ)' where x_i are condition-averaged activity vectors
    D_C_centered = D_C - D_C.mean(axis=1, keepdims=True)
    Sigma_C = D_C_centered @ D_C_centered.T / D_C.shape[1]
    
    # Sanity check: covariance should be symmetric positive semidefinite
    # (by construction it is, but check for numerical issues)
    eigvals = np.linalg.eigvalsh(Sigma_C)
    if np.any(eigvals < -1e-10):
        print(f"  [warn] Sigma_C has negative eigenvalues (numerical issue): min={eigvals.min():.2e}")
    
    # Compute observed Alignment Index
    AI_obs = compute_alignment_index(U_S, Sigma_C, U_C)
    print(f"  [AI] Observed AI = {AI_obs:.4f}")
    
    # Also compute principal angles and overlap for reference
    principal_angles = compute_principal_angles(U_C, U_S)
    overlap_obs = compute_subspace_overlap(U_C, U_S)
    mean_angle_deg = float(np.degrees(np.mean(principal_angles)))
    
    print(f"       Principal angles (deg): {np.degrees(principal_angles)}")
    print(f"       Mean angle: {mean_angle_deg:.1f}°, Overlap: {overlap_obs:.4f}")
    
    # Build covariance-matched null
    # 
    # Key insight: We want random subspaces constrained to the neural manifold.
    # The manifold is estimated from the combined C+S activity (both epochs share
    # the same neurons, so their activity lies on the same low-dim manifold).
    #
    # Mathematical construction:
    #   1. D_combined = [D_C, D_S] captures the full task-evoked activity
    #   2. SVD: D = U @ diag(s) @ V.T
    #   3. cov_sqrt = U @ diag(sqrt(s)) satisfies: cov_sqrt @ cov_sqrt.T = U @ diag(s) @ U.T
    #   4. Random subspace: project random Gaussians through cov_sqrt, then orthonormalize
    #
    D_combined = np.hstack([D_C, D_S])
    D_combined_centered = D_combined - D_combined.mean(axis=1, keepdims=True)
    
    # SVD for covariance structure (not full covariance, but same subspace structure)
    U_cov, s_cov, _ = np.linalg.svd(D_combined_centered, full_matrices=False)
    
    # Keep components explaining 99% variance (defines the effective manifold)
    # This is critical: constraining to manifold makes "chance" more realistic
    cumvar = np.cumsum(s_cov**2) / np.sum(s_cov**2)
    n_keep = min(np.searchsorted(cumvar, 0.99) + 1, len(s_cov))
    n_keep = max(n_keep, k + 1)  # At least k+1 dimensions to sample k-dim subspace
    
    # IMPORTANT: Use s (singular values), NOT sqrt(s)!
    # 
    # Math: If D = U @ diag(s) @ V.T, then DD.T = U @ diag(s²) @ U.T
    # A valid factor L such that LL.T = DD.T is: L = U @ diag(s)
    # 
    # Using sqrt(s) would give LL.T = U @ diag(s) @ U.T, which has the WRONG
    # eigenspectrum (proportional to s, not s²). This changes the geometry!
    #
    cov_sqrt = U_cov[:, :n_keep] * s_cov[:n_keep]  # NOT sqrt!
    
    # Sanity check: effective dimensionality
    effective_dim = n_keep
    expected_random_overlap = k / effective_dim  # for isotropic random subspaces
    print(f"  [null] Manifold dim: {n_keep} (from {len(s_cov)} total)")
    print(f"         Expected random overlap (isotropic): ~{expected_random_overlap:.3f}")
    
    # Generate null distribution
    rng = np.random.default_rng(seed)
    null_AI = np.empty(n_perms, dtype=float)
    null_overlap = np.empty(n_perms, dtype=float)
    null_mean_angle = np.empty(n_perms, dtype=float)
    
    print(f"  [null] Running {n_perms} covariance-matched null samples...")
    for b in range(n_perms):
        # Sample random subspace constrained by manifold covariance
        U_rand = sample_covariance_matched_subspace(cov_sqrt, k, rng)
        
        # Compute AI for random subspace
        null_AI[b] = compute_alignment_index(U_rand, Sigma_C, U_C)
        null_overlap[b] = compute_subspace_overlap(U_C, U_rand)
        null_mean_angle[b] = np.degrees(np.mean(compute_principal_angles(U_C, U_rand)))
        
        if (b + 1) % 200 == 0:
            print(f"    [{b+1}/{n_perms}] null_AI_mean={np.nanmean(null_AI[:b+1]):.4f}")
    
    # Compute statistics
    null_AI_mean = float(np.nanmean(null_AI))
    null_AI_std = float(np.nanstd(null_AI))
    null_overlap_mean = float(np.nanmean(null_overlap))
    null_angle_mean = float(np.nanmean(null_mean_angle))
    
    # Sanity check: null distribution should be concentrated (not too wide)
    # and have reasonable mean (not 0 or 1)
    if null_AI_std > 0.5 * null_AI_mean:
        print(f"  [warn] Null AI distribution may be too wide: mean={null_AI_mean:.3f}, std={null_AI_std:.3f}")
    if null_AI_mean < 0.01 or null_AI_mean > 0.99:
        print(f"  [warn] Null AI mean is extreme: {null_AI_mean:.3f} (check manifold estimation)")
    
    # P-values (with +1 correction for finite-sample validity)
    # 
    # AI interpretation:
    #   - High AI (≈1): saccade subspace captures as much category variance as category subspace → ALIGNED
    #   - Low AI (<<1): saccade subspace captures little category variance → ORTHOGONAL
    #
    # p_less = P(AI_null ≤ AI_obs): 
    #   - If AI_obs is unusually LOW, most null values are ABOVE it, so p_less is SMALL
    #   - Small p_less → "more orthogonal than chance" (saccade captures less category variance than random)
    #
    # p_greater = P(AI_null ≥ AI_obs):
    #   - If AI_obs is unusually HIGH, most null values are BELOW it, so p_greater is SMALL
    #   - Small p_greater → "more aligned than chance" (saccade captures more category variance than random)
    #
    p_less = float((1 + np.sum(null_AI <= AI_obs)) / (1 + n_perms))
    p_greater = float((1 + np.sum(null_AI >= AI_obs)) / (1 + n_perms))
    
    # Effect size
    delta_AI = AI_obs - null_AI_mean
    
    print(f"  [result] AI_obs={AI_obs:.4f}, null_mean={null_AI_mean:.4f}±{null_AI_std:.4f}")
    print(f"           Δ_AI={delta_AI:.4f}")
    print(f"           p(less aligned)={p_less:.4f}, p(more aligned)={p_greater:.4f}")
    
    result = {
        "sid": sid,
        "area": area,
        "monkey": get_monkey(sid),
        "analysis": "alignment_index",
        
        # Observed metrics
        "AI_obs": AI_obs,
        "overlap_obs": overlap_obs,
        "mean_angle_obs_deg": mean_angle_deg,
        "principal_angles_deg": np.degrees(principal_angles).tolist(),
        
        # Variance explained by subspaces
        "var_explained_C": float(var_C),
        "var_explained_S": float(var_S),
        
        # Null distribution
        "null_AI_mean": null_AI_mean,
        "null_AI_std": null_AI_std,
        "null_AI": null_AI,
        "null_overlap_mean": null_overlap_mean,
        "null_angle_mean_deg": null_angle_mean,
        
        # P-values
        "p_less": p_less,      # P(null ≤ obs) - less aligned than chance
        "p_greater": p_greater, # P(null ≥ obs) - more aligned than chance
        "n_perms": n_perms,
        
        # Effect size
        "delta_AI": delta_AI,
        
        # Settings
        "win_C": list(win_C),
        "win_S": list(win_S),
        "k": k,
        "manifold_dim": n_keep,
        "n_trials": int(keep.sum()),
        "n_valid_C": int(valid_C.sum()),
        "n_valid_S": int(valid_S.sum()),
    }
    
    return result


# =============================================================================
# ANALYSIS B: Axis Angle + Label-Shuffle Null (existing approach, refined)
# =============================================================================

def permute_S_within_CR(C: np.ndarray, R: np.ndarray, S: np.ndarray, 
                        rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle S labels within each (C, R) stratum.
    
    This preserves the category-direction structure while breaking
    the neural → saccade mapping.
    """
    S_shuf = S.copy()
    
    # Get unique C and R values
    C_sign = np.sign(C)
    C_vals = np.unique(C_sign[np.isfinite(C_sign)])
    R_vals = np.unique(R[np.isfinite(R)])
    
    for c_val in C_vals:
        for r_val in R_vals:
            mask = (C_sign == c_val) & (R == r_val) & np.isfinite(S)
            n_in_stratum = mask.sum()
            if n_in_stratum > 1:
                S_shuf[mask] = rng.permutation(S[mask])
    
    return S_shuf


def retrain_saccade_axis(
    Z: np.ndarray,          # (N_keep, B, U)
    S_labels: np.ndarray,   # (N_keep,) - ±1
    C_labels: np.ndarray,   # (N_keep,) - for sample weighting
    time_s: np.ndarray,
    winS: Tuple[float, float],
    clf_binary: str,
    C_grid: List[float],
    lda_shrinkage: str = "auto",
) -> np.ndarray:
    """
    Retrain saccade axis with given labels.
    Returns unit vector sS.
    """
    # Window mask
    mask_t = window_mask(time_s, winS)
    if not np.any(mask_t):
        return np.zeros(Z.shape[2])
    
    # Average over window
    Xs = avg_over_window(Z, mask_t)  # (N, U)
    
    # Valid trials
    ok = np.isfinite(S_labels)
    if ok.sum() < 20:
        return np.zeros(Z.shape[2])
    
    Xs = Xs[ok]
    ys = S_labels[ok]
    Cs = C_labels[ok]
    
    # Sample weights for balanced training (within S×C)
    ok2 = np.isfinite(Cs)
    if ok2.sum() < 20:
        return np.zeros(Z.shape[2])
    
    Xs = Xs[ok2]
    ys = ys[ok2]
    Cs = Cs[ok2]
    
    w = np.ones_like(ys, dtype=float)
    for sign_s in [False, True]:
        for sign_c in [False, True]:
            m = ((ys > 0) == sign_s) & ((Cs > 0) == sign_c)
            cnt = m.sum()
            if cnt > 0:
                w[m] = 1.0 / cnt
    
    # Fit classifier
    wS, _, _ = cv_fit_binary_linear(Xs, ys, clf_binary, C_grid, w, lda_shrinkage)
    
    return unit_vec(wS)


def compute_axis_alignment(sC: np.ndarray, sS: np.ndarray) -> Tuple[float, float]:
    """
    Compute alignment between two unit vectors.
    
    Returns:
        a: |cos(θ)| ∈ [0, 1]
        theta_deg: angle in degrees ∈ [0, 90]
    """
    # Ensure unit vectors
    sC = unit_vec(sC)
    sS = unit_vec(sS)
    
    # Dot product
    dot = float(np.dot(sC, sS))
    a = abs(dot)
    
    # Clamp for numerical stability
    a = min(1.0, max(0.0, a))
    
    # Angle in degrees
    theta_deg = float(np.degrees(np.arccos(a)))
    
    return a, theta_deg


def analyze_session_axis_shuffle(
    out_root: Path,
    sid: str,
    axes_tag: str,
    orientation: str,
    pt_min_ms: float,
    n_perms: int,
    seed: int,
    sliding_window_bins: int,
    sliding_step_bins: int,
    clf_binary: str,
    C_grid: List[float],
    lda_shrinkage: str,
    qc_threshold_C: float,
    qc_threshold_S: float,
) -> Optional[Dict]:
    """
    ANALYSIS B: Axis angle + label-shuffle null.
    
    Tests whether the C-S axis alignment is label-specific (i.e., whether
    the alignment is greater than what you'd expect if S labels were arbitrary).
    
    NOTE: This null destroys the true label mapping, so it typically produces
    LOWER alignment. Use this to test "more aligned than chance," NOT
    "less aligned than chance."
    """
    area = get_fef_area(sid)
    
    # Paths
    cache_path = out_root / "sacc" / sid / "caches" / f"area_{area}.npz"
    axes_path = out_root / "sacc" / sid / "axes" / axes_tag / f"axes_{area}.npz"
    qc_path = out_root / "sacc" / sid / "qc" / axes_tag / f"qc_axes_{area}.json"
    
    if not cache_path.exists():
        print(f"  [skip] No cache: {cache_path}")
        return None
    if not axes_path.exists():
        print(f"  [skip] No axes: {axes_path}")
        return None
    
    # Load data
    cache = load_npz(cache_path)
    axes = load_npz(axes_path)
    meta = axes.get("meta", {})
    if isinstance(meta, str):
        meta = json.loads(meta)
    
    # Apply sliding window if needed
    if sliding_window_bins > 0 and sliding_step_bins > 0:
        cache, _ = sliding_window_cache_data(cache, sliding_window_bins, sliding_step_bins)
    
    # Get axes - MUST use sS_raw (not orthogonalized sS_inv)
    sC = axes.get("sC", np.array([]))
    sS_raw = axes.get("sS_raw", np.array([]))
    
    if sC.size == 0 or sS_raw.size == 0:
        print(f"  [skip] Missing sC or sS_raw")
        return None
    
    sC = sC.ravel()
    sS_raw = sS_raw.ravel()
    
    if sC.size != sS_raw.size:
        print(f"  [skip] Dimension mismatch: sC={sC.size}, sS={sS_raw.size}")
        return None
    
    # QC check
    if qc_path.exists():
        try:
            with open(qc_path) as f:
                qc_data = json.load(f)
            
            # Check C AUC
            auc_C = qc_data.get("auc_C", [])
            if auc_C and max(auc_C) < qc_threshold_C:
                print(f"  [qc-fail] auc_C max={max(auc_C):.3f} < {qc_threshold_C}")
                return None
            
            # Check S AUC (try sS_raw first, then sS_inv)
            auc_S = qc_data.get("auc_S_raw", qc_data.get("auc_S_inv", []))
            if auc_S and max(auc_S) < qc_threshold_S:
                print(f"  [qc-fail] auc_S max={max(auc_S):.3f} < {qc_threshold_S}")
                return None
        except Exception as e:
            warnings.warn(f"QC check failed: {e}")
    
    # Compute observed alignment
    a_obs, theta_obs = compute_axis_alignment(sC, sS_raw)
    print(f"  [axis] Observed: |cos(θ)|={a_obs:.4f}, θ={theta_obs:.1f}°")
    
    # Get training window for S (from meta)
    winS_selected = meta.get("winS_selected")
    if winS_selected is None:
        winS = (-0.10, -0.03)
    else:
        winS = tuple(winS_selected)
    
    # Build trial mask
    keep = trial_mask(cache, orientation, pt_min_ms)
    
    # Get labels
    C_labels = cache.get("lab_C", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    R_labels = cache.get("lab_R", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    S_labels = cache.get("lab_S", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    
    # Check valid trials
    valid = np.isfinite(C_labels) & np.isfinite(R_labels) & np.isfinite(S_labels)
    if valid.sum() < 40:
        print(f"  [skip] Not enough valid trials: {valid.sum()}")
        return None
    
    # Get normalized data
    time_s = cache["time"].astype(float)
    Z = cache["Z"][keep].astype(float)
    
    # Generate null distribution by shuffling S labels
    rng = np.random.default_rng(seed)
    null_alignments = np.empty(n_perms, dtype=float)
    null_angles = np.empty(n_perms, dtype=float)
    
    print(f"  [null] Running {n_perms} label-shuffle permutations...")
    for b in range(n_perms):
        # Shuffle S within (C, R) strata
        S_shuf = permute_S_within_CR(C_labels, R_labels, S_labels, rng)
        
        # Retrain S axis with shuffled labels
        sS_shuf = retrain_saccade_axis(
            Z, S_shuf, C_labels, time_s, winS,
            clf_binary, C_grid, lda_shrinkage
        )
        
        if np.linalg.norm(sS_shuf) < 1e-6:
            null_alignments[b] = np.nan
            null_angles[b] = np.nan
        else:
            null_alignments[b], null_angles[b] = compute_axis_alignment(sC, sS_shuf)
        
        if (b + 1) % 100 == 0:
            print(f"    [{b+1}/{n_perms}] null_mean={np.nanmean(null_alignments[:b+1]):.4f}")
    
    # Compute statistics
    null_mean = float(np.nanmean(null_alignments))
    null_std = float(np.nanstd(null_alignments))
    null_median = float(np.nanmedian(null_alignments))
    null_angle_mean = float(np.nanmean(null_angles))
    
    # P-values (with +1 correction for finite-sample validity)
    valid_null = null_alignments[np.isfinite(null_alignments)]
    n_valid = len(valid_null)
    
    # Shuffle null interpretation:
    #   - Shuffling S labels BREAKS the true neural→saccade mapping
    #   - So shuffled axes should be LESS aligned with C than the true S axis
    #   - We expect: a_obs > null_mean (observed is more aligned than shuffle)
    #
    # p_greater = P(null ≥ obs):
    #   - If a_obs is unusually HIGH (above most null values), p_greater is SMALL
    #   - Small p_greater → alignment is LABEL-SPECIFIC (more aligned than random S mapping)
    #   - This is the PRIMARY test for this analysis
    #
    # p_less = P(null ≤ obs):
    #   - If a_obs is unusually LOW (below most null values), p_less is SMALL
    #   - Small p_less would mean alignment is LESS than random S mapping
    #   - This is NOT meaningful for testing orthogonality (use Analysis A instead)
    #
    p_greater = (1 + np.sum(valid_null >= a_obs)) / (1 + n_valid)
    p_less = (1 + np.sum(valid_null <= a_obs)) / (1 + n_valid)
    
    # Effect size
    delta = a_obs - null_mean
    delta_theta = theta_obs - null_angle_mean
    
    print(f"  [result] Δ={delta:.4f}, p(more aligned)={p_greater:.4f}")
    print(f"           Note: shuffle null tests 'more aligned than chance'")
    
    result = {
        "sid": sid,
        "area": area,
        "monkey": get_monkey(sid),
        "analysis": "axis_shuffle",
        "axes_tag": axes_tag,
        
        # Observed
        "a_obs": a_obs,
        "theta_obs_deg": theta_obs,
        
        # Null distribution
        "null_mean": null_mean,
        "null_std": null_std,
        "null_median": null_median,
        "null_alignments": null_alignments,
        "null_angle_mean_deg": null_angle_mean,
        
        # P-values
        "p_greater": float(p_greater),  # Main test: more aligned than shuffle
        "p_less": float(p_less),        # NOT recommended for orthogonality test
        "n_perms_valid": int(n_valid),
        
        # Effect size
        "delta": delta,
        "delta_theta_deg": delta_theta,
        
        # Settings
        "winS": list(winS),
        "n_trials": int(keep.sum()),
        "n_valid_trials": int(valid.sum()),
    }
    
    return result


# =============================================================================
# ANALYSIS C: Axis Angle + Covariance-Constrained Null (NEW - Dave's question)
# =============================================================================

def sample_covariance_matched_axis(
    cov_sqrt: np.ndarray,  # (U, r) matrix such that cov_sqrt @ cov_sqrt.T ∝ Σ
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a random unit vector constrained by the covariance structure.
    
    This is the axis-level analog of sample_covariance_matched_subspace.
    
    Method:
      1. Draw random Gaussian vector z ~ N(0, I_r)
      2. Project through covariance: v = cov_sqrt @ z
      3. Normalize: s_rand = v / ||v||
    
    Returns:
        s_rand: (U,) unit vector on the covariance-constrained manifold
    """
    r = cov_sqrt.shape[1]
    
    # Random Gaussian vector
    z = rng.standard_normal(r)
    
    # Project through covariance structure
    v = cov_sqrt @ z  # (U,)
    
    # Normalize to unit vector
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros(cov_sqrt.shape[0])
    
    return v / norm


def analyze_session_axis_covariance(
    out_root: Path,
    sid: str,
    axes_tag: str,
    orientation: str,
    pt_min_ms: float,
    win_C: Tuple[float, float],
    win_S: Tuple[float, float],
    n_perms: int,
    seed: int,
    sliding_window_bins: int,
    sliding_step_bins: int,
    qc_threshold_C: float,
    qc_threshold_S: float,
    use_rich_cond: bool = True,
) -> Optional[Dict]:
    """
    ANALYSIS C: Axis angle + covariance-constrained null.
    
    This is the proper test for Dave's question: "Are sC and sS more orthogonal
    than expected given the neural manifold geometry?"
    
    Uses:
      - Pre-trained axes (sC, sS_raw) from the axes pipeline
      - Covariance-constrained null (SC Nature Neuroscience style)
    
    The null asks: "If sS were a random direction on the neural manifold,
    how aligned would it be with sC?"
    
    If p_orth is small, we can say:
      "Given the covariance-defined neural geometry, the measured C-S axis
       alignment is smaller than expected under a geometry-constrained null;
       this supports separation of categorical and saccadic representations
       into distinct subspaces."
    """
    area = get_fef_area(sid)
    
    # Paths
    cache_path = out_root / "sacc" / sid / "caches" / f"area_{area}.npz"
    axes_path = out_root / "sacc" / sid / "axes" / axes_tag / f"axes_{area}.npz"
    qc_path = out_root / "sacc" / sid / "qc" / axes_tag / f"qc_axes_{area}.json"
    
    if not cache_path.exists():
        print(f"  [skip] No cache: {cache_path}")
        return None
    if not axes_path.exists():
        print(f"  [skip] No axes: {axes_path}")
        return None
    
    # Load data
    cache = load_npz(cache_path)
    axes = load_npz(axes_path)
    
    # Apply sliding window if needed
    if sliding_window_bins > 0 and sliding_step_bins > 0:
        cache, _ = sliding_window_cache_data(cache, sliding_window_bins, sliding_step_bins)
    
    # Get axes - MUST use sS_raw (not orthogonalized sS_inv)
    sC = axes.get("sC", np.array([]))
    sS_raw = axes.get("sS_raw", np.array([]))
    
    if sC.size == 0 or sS_raw.size == 0:
        print(f"  [skip] Missing sC or sS_raw")
        return None
    
    sC = unit_vec(sC.ravel())
    sS_raw = unit_vec(sS_raw.ravel())
    
    if sC.size != sS_raw.size:
        print(f"  [skip] Dimension mismatch: sC={sC.size}, sS={sS_raw.size}")
        return None
    
    # QC check
    if qc_path.exists():
        try:
            with open(qc_path) as f:
                qc_data = json.load(f)
            
            auc_C = qc_data.get("auc_C", [])
            if auc_C and max(auc_C) < qc_threshold_C:
                print(f"  [qc-fail] auc_C max={max(auc_C):.3f} < {qc_threshold_C}")
                return None
            
            auc_S = qc_data.get("auc_S_raw", qc_data.get("auc_S_inv", []))
            if auc_S and max(auc_S) < qc_threshold_S:
                print(f"  [qc-fail] auc_S max={max(auc_S):.3f} < {qc_threshold_S}")
                return None
        except Exception as e:
            warnings.warn(f"QC check failed: {e}")
    
    # Compute observed alignment
    a_obs, theta_obs = compute_axis_alignment(sC, sS_raw)
    print(f"  [axis-cov] Observed: |cos(θ)|={a_obs:.4f}, θ={theta_obs:.1f}°")
    
    # Build trial mask
    keep = trial_mask(cache, orientation, pt_min_ms)
    
    # Get data and labels
    Z = cache["Z"][keep].astype(float)
    time_s = cache["time"].astype(float)
    
    C_labels = cache.get("lab_C", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    S_labels = cache.get("lab_S", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    
    valid_C = np.isfinite(C_labels)
    valid_S = np.isfinite(S_labels)
    
    if valid_C.sum() < 40 or valid_S.sum() < 40:
        print(f"  [skip] Not enough valid trials: C={valid_C.sum()}, S={valid_S.sum()}")
        return None
    
    # Get additional labels for rich condition IDs
    R_labels = cache.get("lab_R", np.full(cache["Z"].shape[0], np.nan))[keep].astype(float)
    ori_labels = cache.get("lab_orientation", np.array(["pooled"] * cache["Z"].shape[0]))[keep]
    
    # Build condition-averaged activity matrices
    # Using rich condition IDs gives a much more stable manifold estimate
    D_C = condition_average_activity(
        Z, C_labels, time_s, win_C,
        use_rich_cond=use_rich_cond,
        C=C_labels, S=S_labels, R=R_labels, orientation=ori_labels
    )
    D_S = condition_average_activity(
        Z, S_labels, time_s, win_S,
        use_rich_cond=use_rich_cond,
        C=C_labels, S=S_labels, R=R_labels, orientation=ori_labels
    )
    
    if D_C.size == 0 or D_S.size == 0:
        print(f"  [skip] Empty condition-averaged matrix")
        return None
    
    # Build manifold covariance estimate from combined data
    D_combined = np.hstack([D_C, D_S])
    D_combined_centered = D_combined - D_combined.mean(axis=1, keepdims=True)
    
    # SVD for covariance structure
    U_cov, s_cov, _ = np.linalg.svd(D_combined_centered, full_matrices=False)
    
    # Keep components explaining 99% variance
    cumvar = np.cumsum(s_cov**2) / np.sum(s_cov**2)
    n_keep = min(np.searchsorted(cumvar, 0.99) + 1, len(s_cov))
    n_keep = max(n_keep, 2)  # At least 2 dimensions
    
    # Covariance factor: L = U @ diag(s), so LL.T = U @ diag(s²) @ U.T ∝ Σ
    cov_sqrt = U_cov[:, :n_keep] * s_cov[:n_keep]  # NOT sqrt!
    
    # Compute effective dimensionality for reference
    D_eff = (np.sum(s_cov**2)**2) / np.sum(s_cov**4)
    expected_cos2 = 1.0 / D_eff
    expected_cos = np.sqrt(expected_cos2)
    expected_angle = np.degrees(np.arccos(expected_cos))
    
    print(f"  [null] Manifold dim: {n_keep}, D_eff={D_eff:.1f}")
    print(f"         Expected |cos(θ)| ≈ {expected_cos:.3f} (angle ≈ {expected_angle:.1f}°)")
    
    # Generate null distribution
    rng = np.random.default_rng(seed)
    null_alignments = np.empty(n_perms, dtype=float)
    null_angles = np.empty(n_perms, dtype=float)
    
    print(f"  [null] Running {n_perms} covariance-constrained null samples...")
    for b in range(n_perms):
        # Sample random axis from manifold
        s_rand = sample_covariance_matched_axis(cov_sqrt, rng)
        
        if np.linalg.norm(s_rand) < 1e-6:
            null_alignments[b] = np.nan
            null_angles[b] = np.nan
        else:
            null_alignments[b], null_angles[b] = compute_axis_alignment(sC, s_rand)
        
        if (b + 1) % 200 == 0:
            print(f"    [{b+1}/{n_perms}] null_mean={np.nanmean(null_alignments[:b+1]):.4f}")
    
    # Compute statistics
    null_mean = float(np.nanmean(null_alignments))
    null_std = float(np.nanstd(null_alignments))
    null_angle_mean = float(np.nanmean(null_angles))
    
    valid_null = null_alignments[np.isfinite(null_alignments)]
    n_valid = len(valid_null)
    
    # P-values
    # p_orth = P(null ≤ obs): if small, observed is unusually LOW → more orthogonal than chance
    # p_align = P(null ≥ obs): if small, observed is unusually HIGH → more aligned than chance
    p_orth = float((1 + np.sum(valid_null <= a_obs)) / (1 + n_valid))
    p_align = float((1 + np.sum(valid_null >= a_obs)) / (1 + n_valid))
    
    # Effect size (z-score)
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
        print(f"  [**] SIGNIFICANT: C-S alignment is LOWER than geometry-constrained chance!")
    
    result = {
        "sid": sid,
        "area": area,
        "monkey": get_monkey(sid),
        "analysis": "axis_covariance",
        "axes_tag": axes_tag,
        
        # Observed
        "a_obs": a_obs,
        "theta_obs_deg": theta_obs,
        
        # Null distribution
        "null_mean": null_mean,
        "null_std": null_std,
        "null_alignments": null_alignments,
        "null_angle_mean_deg": null_angle_mean,
        
        # P-values
        "p_orth": p_orth,      # Main test: more orthogonal than chance
        "p_align": p_align,    # More aligned than chance
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
        
        # Settings
        "win_C": list(win_C),
        "win_S": list(win_S),
        "n_trials": int(keep.sum()),
        "n_valid_C": int(valid_C.sum()),
        "n_valid_S": int(valid_S.sum()),
        "use_rich_cond": use_rich_cond,
    }
    
    return result


# =============================================================================
# Save per-session results
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


def save_session_result(result: Dict, out_dir: Path, analysis_type: str):
    """Save per-session result to file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sid = result["sid"]
    
    # Save JSON (without large arrays, convert numpy types)
    json_result = {k: _convert_to_json_serializable(v) 
                   for k, v in result.items() 
                   if k not in ["null_AI", "null_alignments"]}
    
    with open(out_dir / f"{analysis_type}_{sid}.json", "w") as f:
        json.dump(json_result, f, indent=2)
    
    # Save NPZ (with null distributions)
    npz_data = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            npz_data[k] = v
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            npz_data[k] = np.array(v)
        else:
            npz_data[k] = np.array([v])
    
    np.savez_compressed(out_dir / f"{analysis_type}_{sid}.npz", **npz_data)
    
    print(f"  [saved] {out_dir / f'{analysis_type}_{sid}.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="""
Category-Saccade subspace alignment analysis in FEF.

THREE complementary analyses:
  A) Alignment Index + covariance-matched null (SC-style): Tests "less aligned than chance" for PCA subspaces
  B) Axis angle + label-shuffle null (PNAS-style): Tests "more aligned than shuffle chance"
  C) Axis angle + covariance-matched null: Tests "less aligned than chance" for trained axes (DAVE'S QUESTION)

Analysis C is the proper test for Dave's question: uses trained axes (sC, sS_raw) with
a covariance-constrained null that respects the neural manifold geometry.

Usage modes:
  Single session: python cli/axis_alignment_analysis.py --sid 20200101 ...
  All sessions:   python cli/axis_alignment_analysis.py --sid_list sid_list.txt ...
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Session selection (mutually exclusive)
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--sid", help="Single session ID to process")
    session_group.add_argument("--sid_list", help="File with list of session IDs")
    
    # Mode selection
    parser.add_argument("--mode", choices=["AI", "axis_shuffle", "axis_cov", "all", "both"], default="all",
                        help="Analysis mode: 'AI', 'axis_shuffle', 'axis_cov', 'all' (all three), or 'both' (A+B only)")
    
    # Data paths
    parser.add_argument("--out_root", default="out", help="Output root directory")
    
    # Alignment Index settings (Analysis A)
    parser.add_argument("--win_C", nargs=2, type=float, default=[-0.30, -0.10],
                        help="Category epoch window [start, end] in seconds (relative to saccade)")
    parser.add_argument("--win_S", nargs=2, type=float, default=[-0.10, -0.03],
                        help="Saccade epoch window [start, end] in seconds")
    parser.add_argument("--k", type=int, default=6,
                        help="Subspace dimensionality for PCA")
    
    # Axis-shuffle settings (Analysis B)
    parser.add_argument("--axes_tag", default="axes_peakbin_saccCS-sacc-horizontal-20mssw",
                        help="Axes tag for axis-shuffle mode (C+S on sacc)")
    
    # Common settings
    parser.add_argument("--orientation", default="horizontal",
                        help="Orientation filter")
    parser.add_argument("--pt_min_ms", type=float, default=200.0)
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    
    # QC thresholds
    parser.add_argument("--qc_threshold_C", type=float, default=0.60)
    parser.add_argument("--qc_threshold_S", type=float, default=0.60)
    
    # Sliding window (must match axes training)
    parser.add_argument("--sliding_window_ms", type=float, default=20.0)
    parser.add_argument("--sliding_step_ms", type=float, default=10.0)
    
    # Classifier settings (for null retraining in axis-shuffle)
    parser.add_argument("--clf_binary", default="logreg")
    parser.add_argument("--C_grid", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--lda_shrinkage", default="auto")
    
    # Rich condition IDs for covariance estimation
    parser.add_argument("--use_rich_cond", action="store_true", default=False,
                        help="Use rich condition IDs (C×S×R×orientation) for manifold estimation")
    parser.add_argument("--no_rich_cond", action="store_true", default=True,
                        help="Disable rich condition IDs, use simple C/S labels only (default: True)")
    
    # Output
    parser.add_argument("--tag", default="alignment")
    
    args = parser.parse_args()
    
    out_root = Path(args.out_root)
    
    # Determine sessions to process
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
    print("CATEGORY-SACCADE SUBSPACE ALIGNMENT ANALYSIS")
    print("="*70)
    print(f"[info] Processing {len(sids)} session(s)")
    print(f"[info] Mode: {args.mode}")
    print(f"[info] out_root: {args.out_root}")
    print(f"[info] n_perms: {args.n_perms}")
    
    if args.mode in ["AI", "both", "all"]:
        print(f"\n[Analysis A] Alignment Index + Covariance-Matched Null (PCA subspaces)")
        print(f"             Category window: {args.win_C}")
        print(f"             Saccade window: {args.win_S}")
        print(f"             Subspace dim k: {args.k}")
        print(f"             Tests: 'less aligned than chance' (orthogonality)")
    
    if args.mode in ["axis_shuffle", "both", "all"]:
        print(f"\n[Analysis B] Axis Angle + Label-Shuffle Null")
        print(f"             Axes tag: {args.axes_tag}")
        print(f"             Tests: 'more aligned than shuffle chance' (label-specific)")
    
    if args.mode in ["axis_cov", "all"]:
        print(f"\n[Analysis C] Axis Angle + Covariance-Constrained Null (DAVE'S QUESTION)")
        print(f"             Axes tag: {args.axes_tag}")
        print(f"             Uses trained sC and sS_raw axes")
        print(f"             Tests: 'less aligned than geometry-constrained chance'")
    
    # Compute sliding window bins
    native_bin_ms_sacc = 5.0
    sw_bins_sacc = int(args.sliding_window_ms / native_bin_ms_sacc)
    sw_step_sacc = int(args.sliding_step_ms / native_bin_ms_sacc)
    
    # Rich condition IDs
    use_rich_cond = args.use_rich_cond and not args.no_rich_cond
    
    # Output directory for per-session results
    results_dir = out_root / "sacc" / "alignment" / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    for sid in sids:
        print(f"\n{'='*60}")
        print(f"[{sid}] Processing...")
        
        # Analysis A: Alignment Index (PCA subspaces)
        if args.mode in ["AI", "both", "all"]:
            print(f"\n  [Analysis A] Alignment Index (PCA subspaces)...")
            result = analyze_session_alignment_index(
                out_root=out_root,
                sid=sid,
                orientation=args.orientation,
                pt_min_ms=args.pt_min_ms,
                win_C=tuple(args.win_C),
                win_S=tuple(args.win_S),
                k=args.k,
                n_perms=args.n_perms,
                seed=args.seed,
                sliding_window_bins=sw_bins_sacc,
                sliding_step_bins=sw_step_sacc,
                qc_threshold_C=args.qc_threshold_C,
                qc_threshold_S=args.qc_threshold_S,
                axes_tag=args.axes_tag,
            )
            if result is not None:
                save_session_result(result, results_dir, "AI")
        
        # Analysis B: Axis shuffle (label-specific test)
        if args.mode in ["axis_shuffle", "both", "all"]:
            print(f"\n  [Analysis B] Axis angle + shuffle...")
            result = analyze_session_axis_shuffle(
                out_root=out_root,
                sid=sid,
                axes_tag=args.axes_tag,
                orientation=args.orientation,
                pt_min_ms=args.pt_min_ms,
                n_perms=args.n_perms,
                seed=args.seed,
                sliding_window_bins=sw_bins_sacc,
                sliding_step_bins=sw_step_sacc,
                clf_binary=args.clf_binary,
                C_grid=args.C_grid,
                lda_shrinkage=args.lda_shrinkage,
                qc_threshold_C=args.qc_threshold_C,
                qc_threshold_S=args.qc_threshold_S,
            )
            if result is not None:
                save_session_result(result, results_dir, "axis_shuffle")
        
        # Analysis C: Axis angle + covariance-constrained null (DAVE'S QUESTION)
        if args.mode in ["axis_cov", "all"]:
            print(f"\n  [Analysis C] Axis angle + covariance-constrained null...")
            result = analyze_session_axis_covariance(
                out_root=out_root,
                sid=sid,
                axes_tag=args.axes_tag,
                orientation=args.orientation,
                pt_min_ms=args.pt_min_ms,
                win_C=tuple(args.win_C),
                win_S=tuple(args.win_S),
                n_perms=args.n_perms,
                seed=args.seed,
                sliding_window_bins=sw_bins_sacc,
                sliding_step_bins=sw_step_sacc,
                qc_threshold_C=args.qc_threshold_C,
                qc_threshold_S=args.qc_threshold_S,
                use_rich_cond=use_rich_cond,
            )
            if result is not None:
                save_session_result(result, results_dir, "axis_cov")
    
    print(f"\n{'='*70}")
    print(f"[done] Per-session results saved to {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
