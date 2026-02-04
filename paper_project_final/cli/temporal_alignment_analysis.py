#!/usr/bin/env python
"""
Temporal C-S Alignment Analysis: Sliding Window Category Axis

This analysis studies how the alignment between Category (C) axis and Saccade (S)
axis changes over time by:

1. Training a fixed S axis using saccade-aligned data with window search
2. Training C axes at multiple time points using a sliding window
3. Computing C-S alignment at each time point to create temporal curves

For each trial type (horizontal, pooled, vertical), we get a curve showing
how C-S alignment evolves over time relative to saccade onset.

Usage:
    python cli/temporal_alignment_analysis.py --sid 20200327 --orientation horizontal
    python cli/temporal_alignment_analysis.py --sid 20200327 --orientation pooled
    python cli/temporal_alignment_analysis.py --sid 20200327 --orientation vertical
    
    # For all trials (no correct filter)
    python cli/temporal_alignment_analysis.py --sid 20200327 --orientation pooled --out_root out_nofilter
"""

from __future__ import annotations
import argparse
import json
import os
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from paperflow.axes import (
    cv_fit_binary_linear, unit_vec, window_mask, avg_over_window
)
from paperflow.norm import sliding_window_cache_data


def get_monkey(sid: str) -> str:
    """Determine monkey from session ID."""
    year = int(sid[:4])
    if year >= 2023:
        return "S"
    return "M"


def trial_mask(cache: dict, orientation: str, pt_min_ms: float = 200.0) -> np.ndarray:
    """Create trial mask based on orientation and reaction time."""
    pt_ms = cache["lab_PT_ms"]
    keep = np.isfinite(pt_ms) & (pt_ms >= pt_min_ms)
    
    if orientation == "horizontal":
        # Cache uses lab_orientation with string values 'horizontal'/'vertical'
        lab_ori = cache["lab_orientation"]
        keep &= (lab_ori == "horizontal")
    elif orientation == "vertical":
        lab_ori = cache["lab_orientation"]
        keep &= (lab_ori == "vertical")
    # pooled: no additional filter
    
    return keep


def load_cache(out_root: str, align: str, sid: str, area: str,
               sliding_window_bins: int = 0, sliding_step_bins: int = 0) -> dict:
    """Load cache with optional sliding window."""
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    
    if sliding_window_bins > 0 and sliding_step_bins > 0:
        cache, _ = sliding_window_cache_data(cache, sliding_window_bins, sliding_step_bins)
    
    return cache


def load_axes(out_root: str, align: str, sid: str, area: str, tag: str) -> Tuple[dict, dict]:
    """Load pre-trained axes and metadata."""
    axes_dir = os.path.join(out_root, align, sid, "axes", tag)
    axes_path = os.path.join(axes_dir, f"axes_{area}.npz")
    
    if not os.path.exists(axes_path):
        raise FileNotFoundError(f"Axes not found: {axes_path}")
    
    d = np.load(axes_path, allow_pickle=True)
    axes = {k: d[k] for k in d.files}
    
    # Load meta
    meta_str = str(d.get("meta", "{}"))
    meta = json.loads(meta_str)
    
    return axes, meta


def train_C_axis_for_window(cache: dict, time_s: np.ndarray, 
                            win_start: float, win_end: float,
                            keep: np.ndarray,
                            clf_binary: str = "logreg",
                            C_grid: List[float] = [1.0],
                            lda_shrinkage: float = 0.5) -> Optional[np.ndarray]:
    """Train a category axis for a specific time window."""
    # Get data
    Z = cache["Z"]  # (N, U, T)
    # Cache uses lab_C for category labels (-1/1)
    y_cat = cache["lab_C"].astype(int)
    
    # Average over window
    mask = window_mask(time_s, (win_start, win_end))
    if not np.any(mask):
        return None
    
    Xc = avg_over_window(Z, mask)  # (N, U)
    
    # Apply trial mask
    Xc = Xc[keep]
    yc = y_cat[keep]
    
    if len(np.unique(yc)) < 2:
        return None
    
    # Train classifier
    try:
        wC, _, _ = cv_fit_binary_linear(Xc, yc, clf_binary, C_grid, None, lda_shrinkage)
        return unit_vec(wC)
    except Exception:
        return None


def compute_alignment(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute |cos(theta)| between two vectors."""
    v1 = unit_vec(v1.ravel())
    v2 = unit_vec(v2.ravel())
    return abs(float(np.dot(v1, v2)))


def compute_null_alignment(Z: np.ndarray, keep: np.ndarray, sS: np.ndarray,
                           n_perms: int = 500, seed: int = 42) -> Dict:
    """
    Compute null distribution for C-S alignment using covariance-constrained random axes.
    
    The null represents: "What alignment would we expect between a random direction
    in neural space and the fixed S axis?"
    
    Method:
    1. Compute covariance matrix of neural activity (trial-averaged)
    2. Generate random axes constrained by this covariance: s_rand = cov_sqrt @ z / ||...||
    3. Compute alignment of each random axis with S axis
    
    Parameters:
    -----------
    Z : np.ndarray
        Neural data (N, T, U) - trials x time x units
    keep : np.ndarray
        Boolean mask for valid trials
    sS : np.ndarray
        Fixed S axis (unit vector)
    n_perms : int
        Number of random axes to generate
    seed : int
        Random seed
    
    Returns:
    --------
    dict : null_mean, null_std, null_alignments, D_eff, expected_cos
    """
    # Get trial-averaged activity
    # Z has shape (N, T, U) = (trials, time, units)
    Z_keep = Z[keep]  # (n_trials, T, U)
    n_trials, n_time, n_units = Z_keep.shape
    
    # Average over time for each trial: (n_trials, U)
    Z_mean = np.mean(Z_keep, axis=1)  # (n_trials, U)
    
    # Compute covariance
    Z_centered = Z_mean - np.mean(Z_mean, axis=0, keepdims=True)
    cov = (Z_centered.T @ Z_centered) / (n_trials - 1)
    
    # SVD of covariance
    U_cov, s_cov, _ = np.linalg.svd(cov, full_matrices=False)
    
    # Keep significant dimensions (eigenvalues > 1e-10)
    n_keep = np.sum(s_cov > 1e-10 * s_cov[0])
    n_keep = max(2, min(n_keep, n_units - 1))
    
    # Covariance factor: L = U @ diag(s), so LL.T ∝ Σ
    cov_sqrt = U_cov[:, :n_keep] * s_cov[:n_keep]
    
    # Effective dimensionality
    D_eff = (np.sum(s_cov**2)**2) / np.sum(s_cov**4) if np.sum(s_cov**4) > 0 else n_keep
    
    # Expected alignment under random null: E[|cos(θ)|] ≈ sqrt(2/(π * D_eff))
    expected_cos = np.sqrt(2.0 / (np.pi * D_eff))
    
    # Generate null distribution
    rng = np.random.default_rng(seed)
    null_alignments = np.empty(n_perms, dtype=float)
    
    for i in range(n_perms):
        # Random direction in neural space, constrained by covariance
        z = rng.standard_normal(n_keep)
        s_rand = cov_sqrt @ z
        norm = np.linalg.norm(s_rand)
        if norm > 1e-10:
            s_rand = s_rand / norm
            null_alignments[i] = abs(float(np.dot(s_rand, sS)))
        else:
            null_alignments[i] = np.nan
    
    # Remove NaN values
    valid_null = null_alignments[np.isfinite(null_alignments)]
    
    return {
        "null_mean": float(np.mean(valid_null)) if len(valid_null) > 0 else np.nan,
        "null_std": float(np.std(valid_null)) if len(valid_null) > 0 else np.nan,
        "null_median": float(np.median(valid_null)) if len(valid_null) > 0 else np.nan,
        "null_alignments": valid_null.tolist(),
        "D_eff": float(D_eff),
        "expected_cos": float(expected_cos),
        "manifold_dim": int(n_keep),
    }


def run_temporal_alignment_analysis(
    sid: str,
    out_root: str,
    orientation: str,
    area: str,
    axes_tag: str,
    window_length_ms: float,
    step_ms: float = 20.0,
    time_range_ms: Tuple[float, float] = (-350.0, 200.0),
    pt_min_ms: float = 200.0,
    clf_binary: str = "logreg",
    C_grid: List[float] = [1.0],
    sliding_window_ms: float = 20.0,
    sliding_step_ms: float = 10.0,
) -> Optional[Dict]:
    """
    Run temporal alignment analysis for a single session.
    
    Parameters:
    -----------
    sid : str
        Session ID
    out_root : str
        Output root directory (e.g., "out" or "out_nofilter")
    orientation : str
        Trial orientation ("horizontal", "pooled", "vertical")
    area : str
        Brain area (e.g., "MFEF", "SFEF")
    axes_tag : str
        Tag for pre-trained axes (to get S axis)
    window_length_ms : float
        Length of sliding window for C axis training
    step_ms : float
        Step size for sliding window (default 20ms)
    time_range_ms : tuple
        (start, end) time range for sliding window centers (ms from saccade)
    pt_min_ms : float
        Minimum reaction time threshold
    clf_binary : str
        Classifier type
    C_grid : list
        Regularization grid
    sliding_window_ms : float
        Sliding window for cache preprocessing (to match axes training)
    sliding_step_ms : float
        Step size for cache preprocessing
    
    Returns:
    --------
    dict : Results including alignment curve and metadata
    """
    print(f"\n[temporal] Session {sid}, orientation={orientation}, area={area}")
    
    # Determine native bin size and compute sliding window params
    native_bin_ms = 5.0  # Sacc-aligned caches use 5ms bins
    sw_bins = int(sliding_window_ms / native_bin_ms)
    sw_step = int(sliding_step_ms / native_bin_ms)
    
    # Load cache
    try:
        cache = load_cache(out_root, "sacc", sid, area, sw_bins, sw_step)
    except FileNotFoundError as e:
        print(f"  [skip] Cache not found: {e}")
        return None
    
    time_s = cache["time"].astype(float)
    bin_ms = (time_s[1] - time_s[0]) * 1000 if len(time_s) > 1 else sliding_step_ms
    
    # Load pre-trained axes to get S axis
    try:
        axes, meta = load_axes(out_root, "sacc", sid, area, axes_tag)
    except FileNotFoundError as e:
        print(f"  [skip] Axes not found: {e}")
        return None
    
    # Get S axis (use raw, not orthogonalized)
    sS = axes.get("sS_raw", axes.get("sS", np.array([])))
    if sS.size == 0:
        print(f"  [skip] No S axis in axes")
        return None
    sS = unit_vec(sS.ravel())
    
    # Get S window info
    winS_selected = meta.get("winS_selected")
    if winS_selected:
        print(f"  [S axis] Window: [{winS_selected[0]*1000:.0f}, {winS_selected[1]*1000:.0f}] ms")
    
    # Build trial mask
    keep = trial_mask(cache, orientation, pt_min_ms)
    n_trials = np.sum(keep)
    print(f"  [trials] {n_trials} trials (orientation={orientation}, PT>={pt_min_ms}ms)")
    
    if n_trials < 20:
        print(f"  [skip] Too few trials ({n_trials})")
        return None
    
    # Convert window length to seconds
    win_len_s = window_length_ms / 1000.0
    
    # Generate sliding window centers
    t_start_s = time_range_ms[0] / 1000.0
    t_end_s = time_range_ms[1] / 1000.0
    step_s = step_ms / 1000.0
    
    window_centers = []
    t = t_start_s
    while t <= t_end_s:
        window_centers.append(t)
        t += step_s
    
    print(f"  [sliding] {len(window_centers)} window positions from {t_start_s*1000:.0f}ms to {t_end_s*1000:.0f}ms")
    print(f"  [window] Length={window_length_ms:.0f}ms, Step={step_ms:.0f}ms")
    
    # Train C axis at each position and compute alignment
    alignments = []
    valid_centers = []
    
    for center_s in window_centers:
        win_start = center_s - win_len_s / 2
        win_end = center_s + win_len_s / 2
        
        # Train C axis
        sC = train_C_axis_for_window(
            cache, time_s, win_start, win_end, keep,
            clf_binary, C_grid
        )
        
        if sC is not None:
            # Compute alignment with S axis
            a = compute_alignment(sC, sS)
            alignments.append(a)
            valid_centers.append(center_s * 1000)  # Store in ms
    
    if len(alignments) < 3:
        print(f"  [skip] Too few valid windows ({len(alignments)})")
        return None
    
    alignments = np.array(alignments)
    valid_centers = np.array(valid_centers)
    
    print(f"  [result] {len(alignments)} valid windows")
    print(f"  [result] Alignment range: [{alignments.min():.3f}, {alignments.max():.3f}]")
    print(f"  [result] Peak alignment: {alignments.max():.3f} at {valid_centers[np.argmax(alignments)]:.0f}ms")
    
    # Compute null distribution
    # Use the SAME SW-transformed cache that was used for analysis
    # This ensures consistency: S axis was trained on SW data, null should use same data
    print(f"  [null] Computing covariance-constrained null...")
    try:
        # Use the already-loaded SW-transformed cache (not raw cache!)
        Z_for_null = cache["Z"]
        null_stats = compute_null_alignment(Z_for_null, keep, sS, n_perms=500, seed=42)
    except Exception as e:
        print(f"  [null] Failed to compute null: {e}")
        null_stats = {
            "null_mean": np.nan, "null_std": np.nan, "null_median": np.nan,
            "D_eff": np.nan, "expected_cos": np.nan, "manifold_dim": 0
        }
    print(f"  [null] D_eff={null_stats['D_eff']:.1f}, expected |cos|={null_stats['expected_cos']:.3f}")
    print(f"  [null] null_mean={null_stats['null_mean']:.3f} ± {null_stats['null_std']:.3f}")
    
    # Get reaction time info
    rt = cache["lab_PT_ms"][keep]
    rt_valid = rt[np.isfinite(rt)]
    rt_mean = float(np.mean(rt_valid)) if len(rt_valid) > 0 else np.nan
    rt_std = float(np.std(rt_valid)) if len(rt_valid) > 0 else np.nan
    
    result = {
        "sid": sid,
        "monkey": get_monkey(sid),
        "orientation": orientation,
        "area": area,
        "axes_tag": axes_tag,
        
        # Alignment curve
        "window_centers_ms": valid_centers.tolist(),
        "alignments": alignments.tolist(),
        
        # Summary stats
        "alignment_mean": float(np.mean(alignments)),
        "alignment_std": float(np.std(alignments)),
        "alignment_max": float(np.max(alignments)),
        "alignment_min": float(np.min(alignments)),
        "peak_time_ms": float(valid_centers[np.argmax(alignments)]),
        
        # Trial info
        "n_trials": int(n_trials),
        "rt_mean_ms": rt_mean,
        "rt_std_ms": rt_std,
        
        # Null distribution
        "null_mean": null_stats["null_mean"],
        "null_std": null_stats["null_std"],
        "null_median": null_stats["null_median"],
        "D_eff": null_stats["D_eff"],
        "expected_cos": null_stats["expected_cos"],
        "manifold_dim": null_stats["manifold_dim"],
        
        # Settings
        "window_length_ms": window_length_ms,
        "step_ms": step_ms,
        "time_range_ms": list(time_range_ms),
        "pt_min_ms": pt_min_ms,
        
        # S axis info
        "winS_selected": winS_selected,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Temporal C-S Alignment Analysis")
    
    # Required
    parser.add_argument("--sid", type=str, required=True, help="Session ID")
    parser.add_argument("--orientation", type=str, required=True,
                        choices=["horizontal", "pooled", "vertical"],
                        help="Trial orientation")
    
    # Paths
    parser.add_argument("--out_root", type=str, default="out",
                        help="Output root (out or out_nofilter)")
    
    # Axes
    parser.add_argument("--axes_tag", type=str, default=None,
                        help="Axes tag (auto-generated if not provided)")
    
    # Sliding window settings
    parser.add_argument("--window_length_ms", type=float, default=None,
                        help="C axis window length in ms (auto from axes if not provided)")
    parser.add_argument("--step_ms", type=float, default=20.0,
                        help="Step size for sliding window (ms)")
    parser.add_argument("--time_range", type=float, nargs=2, default=[-350.0, 200.0],
                        help="Time range for window centers (ms from saccade)")
    
    # Cache preprocessing
    parser.add_argument("--sliding_window_ms", type=float, default=20.0,
                        help="Sliding window for cache preprocessing")
    parser.add_argument("--sliding_step_ms", type=float, default=10.0,
                        help="Step for cache preprocessing")
    
    # Trial selection
    parser.add_argument("--pt_min_ms", type=float, default=200.0,
                        help="Minimum reaction time")
    
    # Classifier
    parser.add_argument("--clf_binary", type=str, default="logreg",
                        help="Classifier type")
    parser.add_argument("--C_grid", type=float, nargs="+", default=[1.0],
                        help="Regularization grid")
    
    # Output
    parser.add_argument("--tag", type=str, default=None,
                        help="Output tag (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Determine areas
    sid = args.sid
    year = int(sid[:4])
    if year >= 2023:
        areas = ["SFEF", "SLIP"]
    else:
        areas = ["MFEF", "MLIP"]
    
    # Auto-generate axes tag if needed
    if args.axes_tag is None:
        args.axes_tag = f"axes_peakbin_saccCS-sacc-{args.orientation}-20mssw"
    
    # Auto-generate output tag
    filter_mode = "correctonly" if args.out_root == "out" else "alltrials"
    if args.tag is None:
        args.tag = f"temporal_alignment_{args.orientation}_{filter_mode}"
    
    print("=" * 70)
    print("TEMPORAL C-S ALIGNMENT ANALYSIS")
    print("=" * 70)
    print(f"Session: {sid}")
    print(f"Orientation: {args.orientation}")
    print(f"Filter mode: {filter_mode}")
    print(f"Axes tag: {args.axes_tag}")
    print("=" * 70)
    
    # Get window length from pre-trained axes if not specified
    window_length_ms = args.window_length_ms
    if window_length_ms is None:
        # Try to load from axes metadata
        for area in areas:
            try:
                _, meta = load_axes(args.out_root, "sacc", sid, area, args.axes_tag)
                winC_selected = meta.get("winC_selected")
                if winC_selected:
                    window_length_ms = (winC_selected[1] - winC_selected[0]) * 1000
                    print(f"[auto] Using C window length from axes: {window_length_ms:.0f}ms")
                    break
            except:
                continue
        
        if window_length_ms is None:
            window_length_ms = 50.0  # Default fallback
            print(f"[default] Using default C window length: {window_length_ms:.0f}ms")
    
    # Run analysis for each area
    results = []
    for area in areas:
        result = run_temporal_alignment_analysis(
            sid=sid,
            out_root=args.out_root,
            orientation=args.orientation,
            area=area,
            axes_tag=args.axes_tag,
            window_length_ms=window_length_ms,
            step_ms=args.step_ms,
            time_range_ms=tuple(args.time_range),
            pt_min_ms=args.pt_min_ms,
            clf_binary=args.clf_binary,
            C_grid=args.C_grid,
            sliding_window_ms=args.sliding_window_ms,
            sliding_step_ms=args.sliding_step_ms,
        )
        
        if result is not None:
            results.append(result)
    
    if not results:
        print("\n[WARN] No results generated")
        return
    
    # Save results
    out_dir = Path(args.out_root) / "temporal_alignment" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for result in results:
        out_path = out_dir / f"temporal_{sid}_{result['area']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[saved] {out_path}")
    
    print("\n[done] Temporal alignment analysis complete")


if __name__ == "__main__":
    main()
