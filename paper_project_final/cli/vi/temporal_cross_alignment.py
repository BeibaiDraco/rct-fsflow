#!/usr/bin/env python3
"""
Time-Resolved Cross-Alignment Analysis

This analysis trains C axes at multiple time points during the stimulus period
and measures alignment with the fixed S axis (trained on saccade period).

This helps understand:
1. When during stimulus period does the C-S alignment emerge?
2. Is the alignment stable across time or concentrated at specific time points?
3. Does alignment peak at category-informative times?

Similar to temporal_alignment_analysis.py but for cross-alignment case vi.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperflow.axes import cv_fit_binary_linear, unit_vec, window_mask, avg_over_window
from paperflow.norm import sliding_window_cache_data

# Add parent to path for importing shared functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from axis_alignment_cross import condition_average_activity


def load_npz(p: Path) -> Dict:
    """Load NPZ file."""
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta" in out and not isinstance(out["meta"], dict):
        try:
            out["meta"] = json.loads(out["meta"].item())
        except Exception:
            pass
    return out


def get_monkey(sid: str) -> str:
    """Return 'M' for 2020 sessions, 'S' for 2023."""
    return "M" if sid.startswith("2020") else "S"


def get_fef_area(sid: str) -> str:
    """Return FEF area name based on monkey."""
    return "MFEF" if get_monkey(sid) == "M" else "SFEF"


def trial_mask(cache: Dict, orientation: str, pt_min_ms: Optional[float]) -> np.ndarray:
    """Build trial mask."""
    N = cache["Z"].shape[0]
    keep = np.ones(N, dtype=bool)
    
    if orientation != "pooled" and "lab_orientation" in cache:
        keep &= (cache["lab_orientation"].astype(str) == orientation)
    
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))
    
    return keep


def get_cache_sigma(cache: Dict) -> np.ndarray:
    """Get per-unit standard deviation from cache."""
    if "X" in cache:
        X = cache["X"]
        Xf = X.reshape(-1, X.shape[-1]).astype(np.float64)
        sd = np.nanstd(Xf, axis=0, ddof=1)
    else:
        Z = cache["Z"]
        Zf = Z.reshape(-1, Z.shape[-1]).astype(np.float64)
        sd = np.nanstd(Zf, axis=0, ddof=1)
    
    sd[~np.isfinite(sd) | (sd < 1e-10)] = 1.0
    return sd


def train_C_axis_at_window(
    Z: np.ndarray,           # (N_trials, B, U)
    C_labels: np.ndarray,    # (N_trials,)
    time_s: np.ndarray,      # (B,)
    window: Tuple[float, float],
    clf_binary: str = "logreg",
    C_grid: List[float] = [0.1, 0.3, 1.0, 3.0, 10.0],
) -> np.ndarray:
    """
    Train a category axis at a specific time window.
    Returns unit vector sC.
    """
    mask_t = window_mask(time_s, window)
    if not np.any(mask_t):
        return np.zeros(Z.shape[2])
    
    # Average over window
    Xc = avg_over_window(Z, mask_t)  # (N, U)
    
    # Valid trials
    ok = np.isfinite(C_labels) & np.all(np.isfinite(Xc), axis=1)
    if ok.sum() < 40:
        return np.zeros(Z.shape[2])
    
    Xc = Xc[ok]
    yc = C_labels[ok]
    
    # Sample weights for balanced training
    w = np.ones_like(yc, dtype=float)
    for sign in [False, True]:
        m = (yc > 0) == sign
        cnt = m.sum()
        if cnt > 0:
            w[m] = 1.0 / cnt
    
    # Fit classifier
    wC, _, _ = cv_fit_binary_linear(Xc, yc, clf_binary, C_grid, w, "auto")
    
    return unit_vec(wC)


def analyze_temporal_cross_alignment(
    out_root: Path,
    sid: str,
    orientation_C: str,      # orientation for C axis (e.g., "vertical")
    orientation_S: str,      # orientation for S axis (e.g., "horizontal")
    axes_tag_S: str,         # axes tag for S axis
    window_length_ms: float = 50.0,
    step_ms: float = 20.0,
    time_range_ms: Tuple[float, float] = (0.0, 500.0),  # stim period
    pt_min_ms: float = 200.0,
    n_perms: int = 500,
    seed: int = 42,
    clf_binary: str = "logreg",
    C_grid: List[float] = [0.1, 0.3, 1.0, 3.0, 10.0],
    sliding_window_ms: float = 20.0,
    sliding_step_ms: float = 10.0,
) -> Optional[Dict]:
    """
    Temporal cross-alignment analysis.
    
    Train C axis at multiple time points during stim period,
    measure alignment with fixed S axis from sacc period.
    """
    area = get_fef_area(sid)
    
    # Load stim-aligned cache for C
    cache_stim_path = out_root / "stim" / sid / "caches" / f"area_{area}.npz"
    if not cache_stim_path.exists():
        print(f"  [skip] No stim cache: {cache_stim_path}")
        return None
    
    cache_stim = load_npz(cache_stim_path)
    
    # Load sacc-aligned axes for S
    axes_sacc_path = out_root / "sacc" / sid / "axes" / axes_tag_S / f"axes_{area}.npz"
    if not axes_sacc_path.exists():
        print(f"  [skip] No sacc axes: {axes_sacc_path}")
        return None
    
    axes_sacc = load_npz(axes_sacc_path)
    
    # Get S axis
    sS_Z = axes_sacc.get("sS_raw", np.array([]))
    if sS_Z.size == 0:
        print(f"  [skip] Missing sS_raw in sacc axes")
        return None
    sS_Z = unit_vec(sS_Z.ravel())
    
    # Apply sliding window to stim cache if needed
    native_bin_ms = 10.0  # typical for stim-aligned caches
    if sliding_window_ms > 0 and sliding_step_ms > 0:
        sw_bins = int(sliding_window_ms / native_bin_ms)
        sw_step = int(sliding_step_ms / native_bin_ms)
        if sw_bins > 1:
            cache_stim, _ = sliding_window_cache_data(cache_stim, sw_bins, sw_step)
    
    # Build trial masks
    mask_C = trial_mask(cache_stim, orientation_C, pt_min_ms)
    
    if mask_C.sum() < 40:
        print(f"  [skip] Not enough C trials: {mask_C.sum()}")
        return None
    
    # Get data
    Z_stim = cache_stim["Z"][mask_C].astype(np.float64)
    time_stim = cache_stim["time"].astype(float)
    C_labels = cache_stim.get("lab_C", np.full(cache_stim["Z"].shape[0], np.nan))[mask_C].astype(float)
    
    # Get sigma for coordinate conversion
    sigma_stim = get_cache_sigma(cache_stim)
    
    # Also load sacc cache for sigma and null computation
    cache_sacc_path = out_root / "sacc" / sid / "caches" / f"area_{area}.npz"
    if not cache_sacc_path.exists():
        print(f"  [skip] No sacc cache for null: {cache_sacc_path}")
        return None
    
    cache_sacc = load_npz(cache_sacc_path)
    sigma_sacc = get_cache_sigma(cache_sacc)
    
    # Convert S axis to raw space then to a common standardized space
    # For simplicity, use stim sigma as reference
    sigma_ref = sigma_stim
    sS_raw = sS_Z / sigma_sacc
    sS_std = unit_vec(sS_raw * sigma_ref)
    
    # Generate time windows
    window_length_s = window_length_ms / 1000.0
    step_s = step_ms / 1000.0
    time_range_s = (time_range_ms[0] / 1000.0, time_range_ms[1] / 1000.0)
    
    window_centers_s = []
    current_center = time_range_s[0] + window_length_s / 2
    while current_center <= time_range_s[1] - window_length_s / 2:
        window_centers_s.append(current_center)
        current_center += step_s
    
    if not window_centers_s:
        print(f"  [skip] No valid time windows")
        return None
    
    print(f"  [temporal] {len(window_centers_s)} time windows from {time_range_ms[0]:.0f}ms to {time_range_ms[1]:.0f}ms")
    print(f"             C orientation: {orientation_C}, S orientation: {orientation_S}")
    
    # Train C axis at each time window and compute alignment
    alignments = []
    window_centers_ms = []
    
    for center_s in window_centers_s:
        win_start = center_s - window_length_s / 2
        win_end = center_s + window_length_s / 2
        window = (win_start, win_end)
        
        # Train C axis at this window
        sC_Z = train_C_axis_at_window(Z_stim, C_labels, time_stim, window, clf_binary, C_grid)
        
        if np.linalg.norm(sC_Z) < 1e-6:
            alignments.append(np.nan)
            window_centers_ms.append(center_s * 1000)
            continue
        
        # Convert C axis to standardized space
        sC_raw = sC_Z / sigma_stim
        sC_std = unit_vec(sC_raw * sigma_ref)
        
        # Compute alignment
        a = abs(np.dot(sC_std, sS_std))
        a = min(1.0, max(0.0, a))
        alignments.append(a)
        window_centers_ms.append(center_s * 1000)
    
    alignments = np.array(alignments)
    window_centers_ms = np.array(window_centers_ms)
    
    # Compute null distribution using C-EPOCH methodology
    # The null should ask: "How aligned would a RANDOM C axis be with S?"
    # So we sample random axes from the C-epoch manifold (stim-aligned, orientation_C)
    # NOT from the S-epoch manifold (which is what we were doing wrong before)
    n_units = Z_stim.shape[2]
    rng = np.random.default_rng(seed)
    null_alignments = []
    
    print(f"  [null] Computing null distribution ({n_perms} samples)...")
    print(f"         Using C-EPOCH geometry (stim-aligned, {orientation_C} orientation)")
    
    # Build covariance from C-epoch (stim cache) condition-averaged activity
    # Use same window range as the temporal analysis (0 to 500ms)
    win_C_null = (0.0, 0.5)  # C epoch window matching the analysis range
    
    # Get labels from stim cache
    S_labels_stim = cache_stim.get("lab_S", np.full(cache_stim["Z"].shape[0], np.nan))[mask_C].astype(float)
    R_labels_stim = cache_stim.get("lab_R", np.full(cache_stim["Z"].shape[0], np.nan))[mask_C].astype(float)
    ori_stim = cache_stim.get("lab_orientation", np.array(["pooled"] * cache_stim["Z"].shape[0]))[mask_C]
    
    # Build D_C from C epoch - C labels (already z-scored)
    D_C_for_null = condition_average_activity(
        Z_stim, C_labels, time_stim, win_C_null,
        use_rich_cond=False,
        C=C_labels, S=S_labels_stim, R=R_labels_stim, orientation=ori_stim
    )
    
    # Build D_S from C epoch - S labels in same stim window (for richer geometry)
    D_S_from_C_epoch = condition_average_activity(
        Z_stim, S_labels_stim, time_stim, win_C_null,
        use_rich_cond=False,
        C=C_labels, S=S_labels_stim, R=R_labels_stim, orientation=ori_stim
    )
    
    # Combine: D_C and D_S from C epoch (z-scored, stim-aligned)
    if D_C_for_null.size > 0 and D_S_from_C_epoch.size > 0:
        D_C_combined = np.hstack([D_C_for_null, D_S_from_C_epoch])
        D_C_centered = D_C_combined - D_C_combined.mean(axis=1, keepdims=True)
        print(f"         D_C_combined shape: {D_C_combined.shape}")
    elif D_C_for_null.size > 0:
        # Fallback to D_C only
        D_C_centered = D_C_for_null - D_C_for_null.mean(axis=1, keepdims=True)
        print(f"         [warn] Using D_C only (D_S_from_C_epoch was empty)")
    else:
        print(f"  [skip] Empty condition-averaged matrices for null")
        return None
    
    # SVD for covariance structure
    U_cov, s_cov, _ = np.linalg.svd(D_C_centered, full_matrices=False)
    cumvar = np.cumsum(s_cov**2) / np.sum(s_cov**2)
    n_keep = min(np.searchsorted(cumvar, 0.99) + 1, len(s_cov))
    n_keep = max(n_keep, 2)
    
    # cov_sqrt is (units, n_keep)
    cov_sqrt = U_cov[:, :n_keep] * s_cov[:n_keep]
    D_eff = (np.sum(s_cov**2)**2) / np.sum(s_cov**4)
    
    print(f"         Manifold dim: {n_keep}, D_eff={D_eff:.1f}")
    print(f"         Using C-epoch geometry (correct null for temporal analysis)")
    
    # Sample random C axes from C-epoch manifold, compare with fixed S
    for _ in range(n_perms):
        z = rng.standard_normal(n_keep)
        v = cov_sqrt @ z  # Random axis in C-epoch manifold
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            # Convert to standardized space (same as observed C axes)
            c_rand_raw = v / sigma_stim
            c_rand_std = unit_vec(c_rand_raw * sigma_ref)
            null_a = abs(np.dot(c_rand_std, sS_std))
            null_alignments.append(min(1.0, max(0.0, null_a)))
    
    null_alignments = np.array(null_alignments)
    null_mean = float(np.nanmean(null_alignments))
    null_std = float(np.nanstd(null_alignments))
    
    # Summary statistics
    alignment_mean = float(np.nanmean(alignments))
    alignment_std = float(np.nanstd(alignments))
    alignment_max = float(np.nanmax(alignments))
    alignment_min = float(np.nanmin(alignments))
    
    # Peak time
    valid_idx = np.isfinite(alignments)
    if np.any(valid_idx):
        peak_idx = np.nanargmax(alignments)
        peak_time_ms = window_centers_ms[peak_idx]
    else:
        peak_time_ms = np.nan
    
    print(f"  [result] Mean alignment: {alignment_mean:.3f}, Max: {alignment_max:.3f}")
    print(f"           Null: {null_mean:.3f}±{null_std:.3f}")
    print(f"           Peak time: {peak_time_ms:.0f}ms")
    
    return {
        "sid": sid,
        "monkey": get_monkey(sid),
        "area": area,
        "analysis": "temporal_cross_alignment",
        
        "orientation_C": orientation_C,
        "orientation_S": orientation_S,
        "axes_tag_S": axes_tag_S,
        
        "window_centers_ms": [float(x) for x in window_centers_ms],
        "alignments": [float(x) if np.isfinite(x) else None for x in alignments],
        
        "alignment_mean": float(alignment_mean),
        "alignment_std": float(alignment_std),
        "alignment_max": float(alignment_max),
        "alignment_min": float(alignment_min),
        "peak_time_ms": float(peak_time_ms) if np.isfinite(peak_time_ms) else None,
        
        "null_mean": float(null_mean),
        "null_std": float(null_std),
        "null_mode": "c_epoch",  # Null using C-epoch geometry (correct for temporal analysis)
        "D_eff": float(D_eff),
        "manifold_dim": int(n_keep),
        
        "window_length_ms": float(window_length_ms),
        "step_ms": float(step_ms),
        "time_range_ms": [float(x) for x in time_range_ms],
        "pt_min_ms": float(pt_min_ms),
        
        "n_trials_C": int(mask_C.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Temporal cross-alignment analysis"
    )
    
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--sid", help="Single session ID")
    session_group.add_argument("--sid_list", help="File with session IDs")
    
    parser.add_argument("--out_root", default="out")
    parser.add_argument("--orientation_C", default="vertical", help="C axis orientation")
    parser.add_argument("--orientation_S", default="horizontal", help="S axis orientation")
    parser.add_argument("--axes_tag_S", default="axes_peakbin_saccCS-sacc-horizontal-20mssw")
    parser.add_argument("--window_length_ms", type=float, default=50.0)
    parser.add_argument("--step_ms", type=float, default=20.0)
    parser.add_argument("--time_range_start_ms", type=float, default=0.0)
    parser.add_argument("--time_range_end_ms", type=float, default=500.0)
    parser.add_argument("--pt_min_ms", type=float, default=200.0)
    parser.add_argument("--n_perms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="temporal_cross")
    parser.add_argument("--output_subdir", default="temporal_cross_alignment",
                        help="Output subdirectory under vi/ (default: temporal_cross_alignment)")
    
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
    print("TEMPORAL CROSS-ALIGNMENT ANALYSIS")
    print("=" * 70)
    print(f"[info] Processing {len(sids)} session(s)")
    print(f"[info] C: {args.orientation_C} (stim-aligned)")
    print(f"[info] S: {args.orientation_S} (sacc-aligned, tag: {args.axes_tag_S})")
    
    # Use project root for output (always under out/vi/), not out_root which may be out_nofilter
    project_root = out_root.parent if out_root.name in ("out", "out_nofilter") else out_root
    results_dir = project_root / "out" / "vi" / args.output_subdir / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for sid in sids:
        print(f"\n{'=' * 60}")
        print(f"[{sid}] Processing...")
        
        result = analyze_temporal_cross_alignment(
            out_root=out_root,
            sid=sid,
            orientation_C=args.orientation_C,
            orientation_S=args.orientation_S,
            axes_tag_S=args.axes_tag_S,
            window_length_ms=args.window_length_ms,
            step_ms=args.step_ms,
            time_range_ms=(args.time_range_start_ms, args.time_range_end_ms),
            pt_min_ms=args.pt_min_ms,
            n_perms=args.n_perms,
            seed=args.seed,
        )
        
        if result is not None:
            # Save per-session result
            with open(results_dir / f"{args.tag}_{sid}_{get_fef_area(sid)}.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"  [saved] {results_dir / f'{args.tag}_{sid}_{get_fef_area(sid)}.json'}")
            all_results.append(result)
    
    # Generate summary figure if we have results
    if all_results:
        print(f"\n{'=' * 60}")
        print("Generating summary figure...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
        
        for result, color in zip(all_results, colors):
            t = np.array(result["window_centers_ms"])
            a = np.array(result["alignments"])
            ax.plot(t, a, color=color, alpha=0.5, linewidth=1, label=result["sid"])
        
        # Compute mean across sessions
        all_alignments = []
        common_times = None
        
        for result in all_results:
            t = np.array(result["window_centers_ms"])
            a = np.array(result["alignments"])
            if common_times is None:
                common_times = t
            all_alignments.append(a)
        
        if all_alignments:
            all_alignments = np.array(all_alignments)
            mean_alignment = np.nanmean(all_alignments, axis=0)
            ax.plot(common_times, mean_alignment, 'k-', linewidth=3, label='Mean')
            
            # Null reference
            null_means = [r["null_mean"] for r in all_results]
            ax.axhline(np.mean(null_means), color='red', linestyle='--', linewidth=2, label=f'Null mean ({np.mean(null_means):.3f})')
        
        ax.set_xlabel('Time from Stimulus Onset (ms)')
        ax.set_ylabel('|cos(θ)| Alignment')
        ax.set_title(f'Temporal Cross-Alignment: C({args.orientation_C}, stim) vs S({args.orientation_S}, sacc)\n'
                     f'N={len(all_results)} sessions')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_xlim(args.time_range_start_ms, args.time_range_end_ms)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        fig.savefig(results_dir / f"{args.tag}_summary.png", dpi=150, bbox_inches='tight')
        fig.savefig(results_dir / f"{args.tag}_summary.pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {results_dir / f'{args.tag}_summary.png'}")
    
    print(f"\n{'=' * 70}")
    print(f"[done] Results saved to {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
