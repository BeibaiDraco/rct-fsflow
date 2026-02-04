#!/usr/bin/env python3
"""
Analysis 2: Target Location Confound Check

This analysis checks whether the high alignment in case vi is due to 
shared target location encoding.

The task has 4 target locations (2 vertical, 2 horizontal positions).
We check:
1. Whether vertical-C axis aligns with target position encoding
2. Whether horizontal-S axis aligns with target position encoding  
3. Whether the C-S alignment can be explained by shared target location coding

Specifically:
- Train target position axes from neural data
- Compare C and S axes to these target axes
- Check if C-S alignment disappears when projecting out target location

Usage:
    python cli/analyze_target_location_confound.py --sid 20200327
    python cli/analyze_target_location_confound.py --all
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def get_fef_area(sid: str) -> str:
    return "MFEF" if int(sid[:4]) <= 2021 else "SFEF"


def load_axis(axes_path: Path, key: str = "sC") -> Optional[np.ndarray]:
    """Load axis from npz file."""
    if not axes_path.exists():
        return None
    d = np.load(axes_path, allow_pickle=True)
    if key in d and d[key].size > 0:
        return d[key].flatten()
    return None


def load_cache(cache_path: Path) -> Optional[dict]:
    """Load neural cache."""
    if not cache_path.exists():
        return None
    d = np.load(cache_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def compute_alignment(w1: np.ndarray, w2: np.ndarray) -> float:
    """Compute |cos(θ)| between two axes."""
    w1 = w1.flatten() / (np.linalg.norm(w1) + 1e-10)
    w2 = w2.flatten() / (np.linalg.norm(w2) + 1e-10)
    return float(np.abs(np.dot(w1, w2)))


def train_target_axis(X: np.ndarray, labels: np.ndarray, C: float = 1.0) -> Tuple[np.ndarray, float]:
    """Train axis to discriminate target positions using logistic regression."""
    # Remove NaN labels
    valid = np.isfinite(labels)
    X = X[valid]
    labels = labels[valid]
    
    if len(np.unique(labels)) < 2:
        return None, 0.0
    
    clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
    
    # Cross-validation score
    try:
        scores = cross_val_score(clf, X, labels, cv=5, scoring='roc_auc')
        auc = np.mean(scores)
    except:
        auc = 0.5
    
    # Fit on all data to get axis
    clf.fit(X, labels)
    w = clf.coef_.flatten()
    
    return w, auc


def analyze_session(sid: str, project_root: Path, out_root: str = "out", axes_suffix: str = "20mssw") -> Dict:
    """Analyze target location confound for one session."""
    
    area = get_fef_area(sid)
    out_path = project_root / out_root
    
    result = {
        "sid": sid,
        "area": area,
        "analysis": "target_location_confound",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Load C axis (stim-vertical)
    c_axes_path = out_path / "stim" / sid / "axes" / f"axes_peakbin_stimC-stim-vertical-{axes_suffix}" / f"axes_{area}.npz"
    w_C = load_axis(c_axes_path, "sC")
    
    # Load S axis (sacc-horizontal)
    s_axes_path = out_path / "sacc" / sid / "axes" / f"axes_peakbin_saccCS-sacc-horizontal-{axes_suffix}" / f"axes_{area}.npz"
    w_S = load_axis(s_axes_path, "sS_raw")
    if w_S is None:
        w_S = load_axis(s_axes_path, "sS_inv")
    
    if w_C is None or w_S is None:
        result["error"] = "Missing axes"
        return result
    
    # Original C-S alignment
    cos_CS_original = compute_alignment(w_C, w_S)
    result["cos_CS_original"] = cos_CS_original
    
    # Load stim cache for target position analysis
    stim_cache_path = out_path / "stim" / sid / "caches" / f"area_{area}.npz"
    stim_cache = load_cache(stim_cache_path)
    
    # Load sacc cache
    sacc_cache_path = out_path / "sacc" / sid / "caches" / f"area_{area}.npz"
    sacc_cache = load_cache(sacc_cache_path)
    
    if stim_cache is None or sacc_cache is None:
        result["error"] = "Missing caches"
        return result
    
    # Get target position labels
    # Try different possible label names
    target_labels = None
    for key in ["lab_target_pos", "lab_targ_pos", "lab_target", "target_pos", "targ_pos"]:
        if key in stim_cache:
            target_labels = stim_cache[key]
            break
    
    # If no explicit target position, try to construct from saccade direction
    if target_labels is None and "lab_saccade_dir" in stim_cache:
        target_labels = stim_cache["lab_saccade_dir"]
    
    # Get saccade direction for S axis analysis
    sacc_dir = None
    for key in ["lab_saccade_dir", "lab_sacc_dir", "saccade_dir"]:
        if key in sacc_cache:
            sacc_dir = sacc_cache[key]
            break
    
    # Get category labels - key is 'lab_C' in the cache
    cat_labels = None
    for key in ["lab_C", "lab_category", "lab_cat", "category"]:
        if key in stim_cache:
            cat_labels = stim_cache[key]
            break
    
    # Average neural data over relevant time windows
    # Cache structure: X is (trials, time, neurons)
    X_stim = stim_cache["X"]  # (trials, time, neurons)
    t_stim = stim_cache["time"]  # time axis (corresponds to axis 1)
    
    X_sacc = sacc_cache["X"]  # (trials, time, neurons)
    t_sacc = sacc_cache["time"]  # time axis (corresponds to axis 1)
    
    # Stim window for C axis: [0, 0.5]
    stim_mask = (t_stim >= 0.0) & (t_stim <= 0.5)
    if stim_mask.sum() == 0:
        stim_mask = np.ones(len(t_stim), dtype=bool)  # use all if window doesn't overlap
    X_stim_avg = X_stim[:, stim_mask, :].mean(axis=1)  # (trials, neurons)
    
    # Sacc window for S axis: [-0.2, 0.05]
    sacc_mask = (t_sacc >= -0.2) & (t_sacc <= 0.05)
    if sacc_mask.sum() == 0:
        sacc_mask = np.ones(len(t_sacc), dtype=bool)  # use all if window doesn't overlap
    X_sacc_avg = X_sacc[:, sacc_mask, :].mean(axis=1)  # (trials, neurons)
    
    # Train target position axis on stim data (if available)
    if target_labels is not None:
        # Convert target labels to binary (e.g., vertical vs horizontal position)
        unique_targets = np.unique(target_labels[np.isfinite(target_labels)])
        
        if len(unique_targets) >= 2:
            # Try to find a meaningful binary split
            # Assuming targets might be encoded as positions or directions
            target_binary = (target_labels > np.median(target_labels[np.isfinite(target_labels)])).astype(float)
            
            w_target, auc_target = train_target_axis(X_stim_avg, target_binary)
            
            if w_target is not None:
                result["target_axis_auc"] = auc_target
                result["cos_C_target"] = compute_alignment(w_C, w_target)
                result["cos_S_target"] = compute_alignment(w_S, w_target)
    
    # Train saccade direction axis on sacc data (if available)
    if sacc_dir is not None:
        unique_dirs = np.unique(sacc_dir[np.isfinite(sacc_dir)])
        
        if len(unique_dirs) >= 2:
            # Binary: left vs right (for horizontal saccades)
            sacc_binary = (sacc_dir > np.median(sacc_dir[np.isfinite(sacc_dir)])).astype(float)
            
            w_sacc_dir, auc_sacc_dir = train_target_axis(X_sacc_avg, sacc_binary)
            
            if w_sacc_dir is not None:
                result["sacc_dir_axis_auc"] = auc_sacc_dir
                result["cos_C_sacc_dir"] = compute_alignment(w_C, w_sacc_dir)
                result["cos_S_sacc_dir"] = compute_alignment(w_S, w_sacc_dir)
    
    # Project out shared component and recompute alignment
    # Method: Orthogonalize C against the C-S shared direction
    shared_dir = w_C + w_S
    shared_dir = shared_dir / (np.linalg.norm(shared_dir) + 1e-10)
    
    # C orthogonal to shared direction
    w_C_orth = w_C - np.dot(w_C, shared_dir) * shared_dir
    w_C_orth = w_C_orth / (np.linalg.norm(w_C_orth) + 1e-10)
    
    # S orthogonal to shared direction  
    w_S_orth = w_S - np.dot(w_S, shared_dir) * shared_dir
    w_S_orth = w_S_orth / (np.linalg.norm(w_S_orth) + 1e-10)
    
    result["cos_CS_after_orth"] = compute_alignment(w_C_orth, w_S_orth)
    
    # Check orientation-specific encoding
    # Compare C axis trained on vertical vs horizontal trials
    c_horiz_path = out_path / "stim" / sid / "axes" / f"axes_peakbin_stimC-stim-horizontal-{axes_suffix}" / f"axes_{area}.npz"
    w_C_horiz = load_axis(c_horiz_path, "sC")
    
    if w_C_horiz is not None:
        result["cos_Cvert_Choriz"] = compute_alignment(w_C, w_C_horiz)
    
    # Compare S axis trained on vertical vs horizontal trials
    s_vert_path = out_path / "sacc" / sid / "axes" / f"axes_peakbin_saccCS-sacc-vertical-{axes_suffix}" / f"axes_{area}.npz"
    w_S_vert = load_axis(s_vert_path, "sS_raw")
    if w_S_vert is None:
        w_S_vert = load_axis(s_vert_path, "sS_inv")
    
    if w_S_vert is not None:
        result["cos_Shoriz_Svert"] = compute_alignment(w_S, w_S_vert)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze target location confound")
    parser.add_argument("--sid", type=str, help="Single session ID")
    parser.add_argument("--all", action="store_true", help="Run all sessions")
    parser.add_argument("--out_root", default="out", help="Output root for reading data")
    parser.add_argument("--output_subdir", default="target_location_analysis",
                        help="Output subdirectory under out/vi/ (default: target_location_analysis)")
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
            out_file = out_dir / f"target_{sid}.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            if "error" not in result:
                print(f"    cos(C,S)={result['cos_CS_original']:.3f}")
                if "cos_C_target" in result:
                    print(f"    cos(C,target)={result['cos_C_target']:.3f}, cos(S,target)={result['cos_S_target']:.3f}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"sid": sid, "error": str(e)})
    
    # Summary
    valid_results = [r for r in all_results if "error" not in r]
    
    if valid_results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print(f"N sessions: {len(valid_results)}")
        
        cos_orig = [r["cos_CS_original"] for r in valid_results]
        print(f"Mean cos(C,S) original: {np.mean(cos_orig):.3f} ± {np.std(cos_orig):.3f}")
        
        if any("cos_C_target" in r for r in valid_results):
            cos_C_targ = [r["cos_C_target"] for r in valid_results if "cos_C_target" in r]
            cos_S_targ = [r["cos_S_target"] for r in valid_results if "cos_S_target" in r]
            print(f"Mean cos(C,target): {np.mean(cos_C_targ):.3f} ± {np.std(cos_C_targ):.3f}")
            print(f"Mean cos(S,target): {np.mean(cos_S_targ):.3f} ± {np.std(cos_S_targ):.3f}")
        
        # Save summary
        summary = {
            "analysis": "target_location_confound",
            "n_sessions": len(valid_results),
            "mean_cos_CS_original": float(np.mean(cos_orig)),
            "per_session": valid_results,
        }
        
        summary_file = out_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {summary_file}")


if __name__ == "__main__":
    main()
