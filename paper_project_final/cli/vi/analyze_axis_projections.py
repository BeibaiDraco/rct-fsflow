#!/usr/bin/env python3
"""
Analysis 1: Direct Axis Projection Analysis

For each session, this analysis:
1. Loads C (stim-vertical) and S (sacc-horizontal) axes
2. Loads neural covariance from both alignments
3. Projects both axes onto the top principal components
4. Identifies which PCs both axes share significant loadings on
5. Quantifies the shared variance

This helps understand WHY case vi shows high alignment by revealing
which neural dimensions the C and S axes share.

Usage:
    python cli/analyze_axis_projections.py --sid 20200327
    python cli/analyze_axis_projections.py --all
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_fef_area(sid: str) -> str:
    """Get FEF area name for a session."""
    return "MFEF" if int(sid[:4]) <= 2021 else "SFEF"


def load_axis(axes_path: Path) -> Optional[np.ndarray]:
    """Load axis from npz file."""
    if not axes_path.exists():
        return None
    d = np.load(axes_path, allow_pickle=True)
    
    # Try different key names
    for key in ["sC", "sS_raw", "sS_inv"]:
        if key in d:
            arr = d[key]
            if arr.size > 0:
                return arr.flatten()
    return None


def load_cache_and_compute_cov(cache_path: Path, win_start: float, win_end: float,
                                orientation: str = None, pt_min_ms: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load neural cache and compute covariance matrix within time window.
    
    Cache structure: X is (trials, time, neurons), time array is (time,)
    """
    if not cache_path.exists():
        return None, None
    
    d = np.load(cache_path, allow_pickle=True)
    X = d["X"]  # (trials, time, neurons)
    t_grid = d["time"]  # time axis (corresponds to axis 1 of X)
    
    # Trial mask
    pt_ms = d.get("lab_PT_ms", np.full(X.shape[0], 300.0))
    keep = np.isfinite(pt_ms) & (pt_ms >= pt_min_ms)
    
    if orientation and "lab_orientation" in d:
        lab_ori = d["lab_orientation"]
        if orientation == "horizontal":
            keep &= (lab_ori == "horizontal")
        elif orientation == "vertical":
            keep &= (lab_ori == "vertical")
    
    X = X[keep]
    
    # Time window mask - use the cache's own time grid
    time_mask = (t_grid >= win_start) & (t_grid <= win_end)
    
    # Check if any time points fall within the window
    if time_mask.sum() == 0:
        # Window doesn't overlap with this cache's time range, use all time points
        X_win = X.mean(axis=1)  # (trials, neurons) - average over time (axis 1)
    else:
        X_win = X[:, time_mask, :].mean(axis=1)  # (trials, neurons)
    
    # Center and compute covariance
    X_centered = X_win - X_win.mean(axis=0, keepdims=True)
    cov = X_centered.T @ X_centered / max(1, X_win.shape[0] - 1)
    
    return cov, X_win


def project_axis_onto_pcs(w: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Project axis onto principal components."""
    w = w / (np.linalg.norm(w) + 1e-10)
    return eigvecs.T @ w


def analyze_session(sid: str, project_root: Path, out_root: str = "out", axes_suffix: str = "20mssw") -> Dict:
    """Analyze axis projections for one session."""
    
    area = get_fef_area(sid)
    out_path = project_root / out_root
    
    result = {
        "sid": sid,
        "area": area,
        "analysis": "axis_projections",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Load C axis (stim-vertical)
    c_axes_path = out_path / "stim" / sid / "axes" / f"axes_peakbin_stimC-stim-vertical-{axes_suffix}" / f"axes_{area}.npz"
    w_C = load_axis(c_axes_path)
    
    # Load S axis (sacc-horizontal)  
    s_axes_path = out_path / "sacc" / sid / "axes" / f"axes_peakbin_saccCS-sacc-horizontal-{axes_suffix}" / f"axes_{area}.npz"
    d = np.load(s_axes_path, allow_pickle=True) if s_axes_path.exists() else {}
    w_S = d.get("sS_raw", d.get("sS_inv", None))
    if w_S is not None and w_S.size > 0:
        w_S = w_S.flatten()
    else:
        w_S = None
    
    if w_C is None or w_S is None:
        result["error"] = "Missing axes"
        return result
    
    # Compute alignment
    w_C_norm = w_C / np.linalg.norm(w_C)
    w_S_norm = w_S / np.linalg.norm(w_S)
    cos_CS = np.abs(np.dot(w_C_norm, w_S_norm))
    result["cos_CS"] = float(cos_CS)
    result["angle_CS_deg"] = float(np.degrees(np.arccos(np.clip(cos_CS, 0, 1))))
    
    # Load stim cache and compute covariance
    stim_cache_path = out_path / "stim" / sid / "caches" / f"area_{area}.npz"
    cov_stim, X_stim = load_cache_and_compute_cov(stim_cache_path, 0.0, 0.5, "vertical")
    
    # Load sacc cache and compute covariance
    sacc_cache_path = out_path / "sacc" / sid / "caches" / f"area_{area}.npz"
    cov_sacc, X_sacc = load_cache_and_compute_cov(sacc_cache_path, -0.2, 0.05, "horizontal")
    
    if cov_stim is None or cov_sacc is None:
        result["error"] = "Missing caches"
        return result
    
    # Combined covariance (pooled from both windows)
    cov_combined = (cov_stim + cov_sacc) / 2
    
    # PCA on combined covariance
    eigvals, eigvecs = np.linalg.eigh(cov_combined)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Variance explained
    total_var = eigvals.sum()
    var_explained = eigvals / total_var
    cumvar = np.cumsum(var_explained)
    
    result["n_neurons"] = len(eigvals)
    result["var_explained_top10"] = var_explained[:10].tolist()
    result["cumvar_top10"] = cumvar[:10].tolist()
    
    # Project axes onto PCs
    proj_C = project_axis_onto_pcs(w_C, eigvecs)
    proj_S = project_axis_onto_pcs(w_S, eigvecs)
    
    result["proj_C_top10"] = proj_C[:10].tolist()
    result["proj_S_top10"] = proj_S[:10].tolist()
    
    # Squared projections (variance contribution)
    proj_C_sq = proj_C ** 2
    proj_S_sq = proj_S ** 2
    
    result["proj_C_sq_top10"] = proj_C_sq[:10].tolist()
    result["proj_S_sq_top10"] = proj_S_sq[:10].tolist()
    
    # Identify shared PCs (both axes have significant loading)
    threshold = 0.1  # At least 10% of axis in this PC
    shared_pcs = []
    for i in range(min(20, len(proj_C))):
        if proj_C_sq[i] > threshold and proj_S_sq[i] > threshold:
            shared_pcs.append({
                "pc": i,
                "var_explained": float(var_explained[i]),
                "proj_C_sq": float(proj_C_sq[i]),
                "proj_S_sq": float(proj_S_sq[i]),
                "product": float(proj_C_sq[i] * proj_S_sq[i]),
            })
    
    result["shared_pcs"] = shared_pcs
    result["n_shared_pcs"] = len(shared_pcs)
    
    # Compute overlap metric: sum of product of squared projections
    overlap = np.sum(proj_C_sq * proj_S_sq)
    result["overlap_metric"] = float(overlap)
    
    # Effective shared dimensionality
    # (how many PCs contribute significantly to both axes)
    joint_loading = np.sqrt(proj_C_sq * proj_S_sq)
    eff_shared_dim = 1 / np.sum(joint_loading ** 2) if np.sum(joint_loading ** 2) > 0 else 0
    result["eff_shared_dim"] = float(eff_shared_dim)
    
    # Top 3 PCs that both axes load onto
    joint_scores = proj_C_sq * proj_S_sq
    top_joint_pcs = np.argsort(joint_scores)[::-1][:3]
    result["top_joint_pcs"] = top_joint_pcs.tolist()
    result["top_joint_scores"] = joint_scores[top_joint_pcs].tolist()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze axis projections onto PCs")
    parser.add_argument("--sid", type=str, help="Single session ID")
    parser.add_argument("--all", action="store_true", help="Run all sessions")
    parser.add_argument("--out_root", default="out", help="Output root for reading data")
    parser.add_argument("--output_subdir", default="axis_projection_analysis",
                        help="Output subdirectory under out/vi/ (default: axis_projection_analysis)")
    parser.add_argument("--axes_suffix", default="20mssw",
                        help="Axes tag suffix (e.g., 20mssw or 20mssw_nofilter)")
    
    args = parser.parse_args()
    
    project_root = Path("/project/bdoiron/dracoxu/rct-fsflow/paper_project_final")
    
    # Get sessions
    if args.sid:
        sids = [args.sid]
    elif args.all:
        sid_list_path = project_root / "sid_list.txt"
        if sid_list_path.exists():
            with open(sid_list_path) as f:
                sids = [line.strip() for line in f if line.strip()]
        else:
            sacc_dir = project_root / args.out_root / "sacc"
            sids = sorted([d.name for d in sacc_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    else:
        parser.print_help()
        return
    
    # Output directory (always under out/vi/, not out_root which may be out_nofilter)
    out_dir = project_root / "out" / "vi" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing {len(sids)} sessions...")
    
    all_results = []
    for sid in sorted(sids):
        print(f"  Processing {sid}...")
        try:
            result = analyze_session(sid, project_root, args.out_root, args.axes_suffix)
            all_results.append(result)
            
            # Save per-session result
            out_file = out_dir / f"proj_{sid}.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            if "error" not in result:
                print(f"    cos(C,S)={result['cos_CS']:.3f}, n_shared_pcs={result['n_shared_pcs']}, overlap={result['overlap_metric']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            all_results.append({"sid": sid, "error": str(e)})
    
    # Summary statistics
    valid_results = [r for r in all_results if "error" not in r]
    
    if valid_results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        cos_vals = [r["cos_CS"] for r in valid_results]
        overlap_vals = [r["overlap_metric"] for r in valid_results]
        n_shared = [r["n_shared_pcs"] for r in valid_results]
        
        print(f"N sessions analyzed: {len(valid_results)}")
        print(f"Mean cos(C,S): {np.mean(cos_vals):.3f} ± {np.std(cos_vals):.3f}")
        print(f"Mean overlap metric: {np.mean(overlap_vals):.4f} ± {np.std(overlap_vals):.4f}")
        print(f"Mean N shared PCs: {np.mean(n_shared):.1f} ± {np.std(n_shared):.1f}")
        
        # Correlation between cos(C,S) and overlap metric
        corr = np.corrcoef(cos_vals, overlap_vals)[0, 1]
        print(f"Correlation(cos, overlap): r = {corr:.3f}")
        
        # Save summary
        summary = {
            "analysis": "axis_projections",
            "n_sessions": len(valid_results),
            "mean_cos_CS": float(np.mean(cos_vals)),
            "std_cos_CS": float(np.std(cos_vals)),
            "mean_overlap": float(np.mean(overlap_vals)),
            "std_overlap": float(np.std(overlap_vals)),
            "mean_n_shared_pcs": float(np.mean(n_shared)),
            "corr_cos_overlap": float(corr),
            "per_session": valid_results,
        }
        
        summary_file = out_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {summary_file}")


if __name__ == "__main__":
    main()
