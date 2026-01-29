#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import normalization utilities
from paperflow.norm import get_Z, get_axes_norm, get_axes_norm_mode, get_axes_baseline_win, rebin_cache_data, sliding_window_cache_data


def parse_window(s: str) -> Tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def load_npz(p: Path) -> Dict:
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta" in out and not isinstance(out["meta"], dict):
        try:
            out["meta"] = json.loads(out["meta"].item())
        except Exception:
            pass
    return out


def list_areas(cache_dir: Path):
    return sorted([p.name[5:-4] for p in cache_dir.glob("area_*.npz")])


def pick_area(areas, key: str) -> Optional[str]:
    hits = [a for a in areas if key.upper() in a.upper()]
    if not hits:
        return None
    return hits[0]


def cache_path(out_root: Path, align: str, sid: str, area: str) -> Path:
    return out_root / align / sid / "caches" / f"area_{area}.npz"


def axis_path(out_root: Path, align: str, sid: str, axes_tag: str, area: str) -> Path:
    return out_root / align / sid / "axes" / axes_tag / f"axes_{area}.npz"


def trial_mask(cache: Dict, orientation: str, pt_min_ms: float | None, require_C: bool = True) -> np.ndarray:
    N = cache["Z"].shape[0]
    keep = np.ones(N, dtype=bool)

    keep &= cache.get("lab_is_correct", np.ones(N, dtype=bool)).astype(bool)

    if orientation != "pooled" and "lab_orientation" in cache:
        keep &= (cache["lab_orientation"].astype(str) == orientation)

    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))

    if require_C:
        C = cache.get("lab_C", np.full(N, np.nan)).astype(float)
        keep &= np.isfinite(C)

    return keep


def project_1d(Z: np.ndarray, s: np.ndarray) -> np.ndarray:
    # Z: (N,B,U), s: (U,) or (U,K)
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1)
    else:
        # For multi-dimensional (e.g., sR), use first dimension
        s = s[:, 0].reshape(-1)
    
    if s.size != Z.shape[2]:
        raise ValueError(f"Axis dim {s.size} != n_units {Z.shape[2]}")
    return np.tensordot(Z, s, axes=([2], [0]))  # (N,B)


def gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    half = int(np.ceil(3 * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k


def smooth_timewise(Y: np.ndarray, sigma_bins: float) -> np.ndarray:
    # Y: (N,B)
    if sigma_bins <= 0:
        return Y
    k = gaussian_kernel(sigma_bins)
    out = np.empty_like(Y, dtype=float)
    for n in range(Y.shape[0]):
        out[n] = np.convolve(Y[n], k, mode="same")
    return out


def calculate_threshold(
    e: np.ndarray,              # (B,)
    time: np.ndarray,           # (B,)
    baseline: Tuple[float,float],
    k_sigma: float,
) -> float:
    """Calculate threshold for a trial: mu + k_sigma * sd"""
    bmask = (time >= baseline[0]) & (time <= baseline[1])
    if not np.any(bmask):
        return np.nan
    mu = float(np.nanmean(e[bmask]))
    sd = float(np.nanstd(e[bmask], ddof=1))
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1e-9
    return mu + k_sigma * sd


def onset_time_per_trial(
    e: np.ndarray,              # (B,)
    time: np.ndarray,           # (B,)
    baseline: Tuple[float,float],
    search: Tuple[float,float],
    k_sigma: float,
    runlen: int,
) -> float:
    # baseline stats per trial
    bmask = (time >= baseline[0]) & (time <= baseline[1])
    if not np.any(bmask):
        return np.nan
    mu = float(np.nanmean(e[bmask]))
    sd = float(np.nanstd(e[bmask], ddof=1))
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1e-9
    thr = mu + k_sigma * sd

    smask = (time >= search[0]) & (time <= search[1])
    idx = np.where(smask)[0]
    if idx.size == 0:
        return np.nan

    for i in idx:
        j = i + runlen
        if j > e.size:
            break
        seg = e[i:j]
        if np.all(np.isfinite(seg)) and np.all(seg > thr):
            return float(time[i])
    return np.nan


def get_monkey(sid: str) -> str:
    """Return 'M' for sessions starting with 2020, 'S' for 2023."""
    if sid.startswith("2020"):
        return "M"
    elif sid.startswith("2023"):
        return "S"
    else:
        return "Unknown"


def signflip_exact_pvalue_session(session_stats_ms: np.ndarray,
                                  alternative: str = "two-sided") -> Dict:
    """
    Exact sign-flip test on session-level stats (e.g. per-session mean dt_ms).
    
    H0: no systematic ordering; swapping areas per session is equally likely => m_s -> -m_s.
    
    Parameters
    ----------
    session_stats_ms : np.ndarray
        Array of per-session numbers (e.g. mean dt_ms per session).
    alternative : str
        'two-sided', 'greater' (obs > 0), 'less' (obs < 0)
    
    Returns
    -------
    dict
        {p, obs, n_sessions}
    """
    x = np.asarray(session_stats_ms, dtype=float)
    x = x[np.isfinite(x)]
    S = x.size
    if S < 3:
        return dict(p=np.nan, obs=np.nan, n_sessions=int(S))
    
    obs = float(np.mean(x))  # mean of session means
    n = 1 << S                # 2^S exact sign patterns
    null = np.empty(n, dtype=float)
    
    for mask in range(n):
        tot = 0.0
        for i in range(S):
            sign = -1.0 if ((mask >> i) & 1) else 1.0
            tot += sign * x[i]
        null[mask] = tot / S
    
    if alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (1 + n)
    elif alternative == "less":
        p = (1 + np.sum(null <= obs)) / (1 + n)
    else:  # two-sided
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n)
    
    return dict(p=float(p), obs=obs, n_sessions=int(S))


def trial_signflip_pvalue(dt_ms: np.ndarray, n_perm: int = 20000,
                          alternative: str = "two-sided", seed: int = 0) -> Dict:
    """
    Trial-level sign-flip permutation test on mean(dt_ms).
    (Descriptive; treats trials as IID.)
    
    Parameters
    ----------
    dt_ms : np.ndarray
        Per-trial differences (t2 - t1) in ms.
    n_perm : int
        Number of permutations (default: 20000)
    alternative : str
        'two-sided', 'greater' (obs > 0), 'less' (obs < 0)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        {p, obs, n_trials}
    """
    x = np.asarray(dt_ms, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < 20:
        return dict(p=np.nan, obs=np.nan, n_trials=int(N))
    
    rng = np.random.default_rng(seed)
    obs = float(np.mean(x))
    
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        signs = rng.choice([-1.0, +1.0], size=N, replace=True)
        null[k] = float(np.mean(signs * x))
    
    if alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    elif alternative == "less":
        p = (1 + np.sum(null <= obs)) / (1 + n_perm)
    else:  # two-sided
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm)
    
    return dict(p=float(p), obs=obs, n_trials=int(N))


def load_qc_json(out_root: Path, align: str, sid: str, qc_tag: str, area: str) -> Optional[Dict]:
    """
    Load QC JSON file for a given area.
    Returns the parsed JSON dict, or None if file doesn't exist.
    """
    qc_path = out_root / align / sid / "qc" / qc_tag / f"qc_axes_{area}.json"
    if not qc_path.exists():
        return None
    try:
        with open(qc_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        warnings.warn(f"Failed to load QC JSON {qc_path}: {e}")
        return None


def feature_to_qc_metric(feature: str) -> str:
    """
    Map feature name to QC metric name.
    C -> auc_C
    R -> acc_R_macro (ACC(R|sR) within C: macro accuracy of R conditional on category C)
    S -> auc_S_inv (prefer inverse over raw)
    T -> auc_T (target configuration)
    """
    mapping = {
        "C": "auc_C",
        "R": "acc_R_macro",  # ACC(R|sR) within C: computed separately for each C, then macro-averaged
        "S": "auc_S_inv",    # prefer inverse over raw
        "T": "auc_T",        # target configuration
    }
    return mapping.get(feature.upper(), "auc_C")  # default to auc_C


def check_qc_passes(qc_data: Dict, metric: str, threshold: float) -> bool:
    """
    Check if a QC metric ever reaches the threshold.
    
    Parameters
    ----------
    qc_data : dict
        Parsed QC JSON data
    metric : str
        Metric name (e.g., "auc_C", "acc_R_macro", "auc_S_inv")
    threshold : float
        Threshold value (e.g., 0.75)
    
    Returns
    -------
    bool
        True if the metric reaches threshold at any time point, False otherwise
    """
    # For S feature, try auc_S_inv first, fall back to auc_S_raw
    if metric == "auc_S_inv":
        if metric in qc_data and qc_data[metric] is not None:
            metric_to_check = metric
        elif "auc_S_raw" in qc_data and qc_data["auc_S_raw"] is not None:
            metric_to_check = "auc_S_raw"
        else:
            return False
    else:
        metric_to_check = metric
    
    if metric_to_check not in qc_data:
        return False
    
    metric_values = qc_data[metric_to_check]
    if metric_values is None:
        return False
    
    # Check if any value reaches threshold
    metric_arr = np.asarray(metric_values, dtype=float)
    valid_mask = np.isfinite(metric_arr)
    if not np.any(valid_mask):
        return False
    
    return bool(np.any(metric_arr[valid_mask] >= threshold))


def check_area_qc(
    out_root: Path,
    align: str,
    sid: str,
    axes_tag: str,
    area: str,
    feature: str,
    qc_threshold: float,
    qc_root: Optional[Path] = None,
    qc_tag: Optional[str] = None,
) -> bool:
    """
    Check if an area passes QC for a given feature.
    
    Parameters
    ----------
    qc_root : Path, optional
        Root directory for QC data. If None, uses out_root.
        This allows using QC computed on different data (e.g., all trials vs correct-only).
    qc_tag : str, optional
        QC tag to use. If None, uses axes_tag.
    
    Returns True if QC passes (or if QC data is unavailable), False if QC fails.
    """
    # Use qc_root if provided, otherwise use out_root
    effective_qc_root = qc_root if qc_root is not None else out_root
    # Use qc_tag if provided, otherwise use axes_tag
    effective_qc_tag = qc_tag if qc_tag is not None else axes_tag
    
    # Check if QC directory exists with this tag
    qc_path = effective_qc_root / align / sid / "qc" / effective_qc_tag
    if not qc_path.exists():
        # If no QC tag found, we can't check - default to passing
        # (could also return False to be strict, but user might want to be lenient)
        return True
    
    # Load QC data
    qc_data = load_qc_json(effective_qc_root, align, sid, effective_qc_tag, area)
    if qc_data is None:
        # No QC data available - default to passing
        return True
    
    # Get the appropriate metric for this feature
    metric = feature_to_qc_metric(feature)
    
    # Check if metric reaches threshold
    return check_qc_passes(qc_data, metric, qc_threshold)


def process_one_session(
    out_root: Path,
    align: str,
    sid: str,
    orientation: str,
    pt_min_ms: float,
    axes_tag: str,
    baseline: Tuple[float, float],
    search: Tuple[float, float],
    k_sigma: float,
    runlen: int,
    smooth_ms: float,
    feature: str,  # "C" for category, "R" for direction, "S" for saccade, "T" for target
    area1: str,    # First area (e.g., "FEF", "LIP", "SC")
    area2: str,    # Second area
    tag: str,
    qc_threshold: float = 0.75,  # QC threshold (default: 0.75)
    # === NEW: normalization parameters ===
    norm: Optional[str] = None,  # 'auto', 'global', 'baseline', 'none'
    norm_baseline_win: Optional[Tuple[float, float]] = None,
    # === NEW: rebinning parameter ===
    rebin_factor: int = 1,  # 1 = no rebinning, 2 = combine pairs of bins, etc.
    # === NEW: sliding window parameters ===
    sliding_window_bins: int = 0,  # window size in bins
    sliding_step_bins: int = 0,    # step size in bins
    # === NEW: separate QC root/tag for mixed training ===
    qc_root: Optional[Path] = None,  # QC root (defaults to out_root)
    qc_tag: Optional[str] = None,    # QC tag (defaults to axes_tag)
) -> Optional[Dict]:
    """Process one session for one pair and feature. Returns dict with results or None if failed."""
    
    cache_dir = out_root / align / sid / "caches"
    if not cache_dir.exists():
        return None
    
    areas = list_areas(cache_dir)
    a1 = pick_area(areas, area1)
    a2 = pick_area(areas, area2)
    
    if a1 is None or a2 is None:
        return None
    
    # QC filtering: check if areas pass QC threshold
    # SYMMETRIC rejection: if EITHER area fails QC, skip the entire session
    # for this pair. This ensures both areas have the same QC status.
    # Note: Category (C) and Direction (R) are checked separately with their own metrics
    qc_pass_a1 = check_area_qc(out_root, align, sid, axes_tag, a1, feature, qc_threshold, qc_root=qc_root, qc_tag=qc_tag)
    qc_pass_a2 = check_area_qc(out_root, align, sid, axes_tag, a2, feature, qc_threshold, qc_root=qc_root, qc_tag=qc_tag)
    
    # Skip session if EITHER area fails QC (symmetric rejection)
    if not qc_pass_a1 or not qc_pass_a2:
        failed_areas = []
        if not qc_pass_a1:
            failed_areas.append(a1)
        if not qc_pass_a2:
            failed_areas.append(a2)
        metric = feature_to_qc_metric(feature)
        feature_name = {"C": "category", "R": "direction", "S": "saccade", "T": "target"}.get(feature, feature)
        print(f"[qc-filter] {sid}: {', '.join(failed_areas)} fails QC for {feature_name} "
              f"(metric={metric}, threshold={qc_threshold}), skipping {a1}-{a2} pair")
        return None
    
    try:
        cache1 = load_npz(cache_path(out_root, align, sid, a1))
        cache2 = load_npz(cache_path(out_root, align, sid, a2))
        
        # Apply sliding window or rebinning if requested (must match axes training)
        # Sliding window takes precedence over rebinning
        if sliding_window_bins > 0 and sliding_step_bins > 0:
            cache1, _ = sliding_window_cache_data(cache1, sliding_window_bins, sliding_step_bins)
            cache2, _ = sliding_window_cache_data(cache2, sliding_window_bins, sliding_step_bins)
        elif rebin_factor > 1:
            cache1, _ = rebin_cache_data(cache1, rebin_factor)
            cache2, _ = rebin_cache_data(cache2, rebin_factor)
        
        # For sacc and targ alignments, don't require C to be finite
        require_C = (align == "stim" and feature in ["C", "R"])
        keep = trial_mask(cache1, orientation, pt_min_ms, require_C=require_C) & trial_mask(cache2, orientation, pt_min_ms, require_C=require_C)
        if keep.sum() < 60:
            return None
        
        # Load axes
        axes1 = load_npz(axis_path(out_root, align, sid, axes_tag, a1))
        axes2 = load_npz(axis_path(out_root, align, sid, axes_tag, a2))
        
        if feature == "C":
            s1 = axes1.get("sC", np.array([]))
            s2 = axes2.get("sC", np.array([]))
        elif feature == "R":
            s1 = axes1.get("sR", np.array([[]]))
            s2 = axes2.get("sR", np.array([[]]))
            if s1.size == 0 or s2.size == 0:
                return None
        elif feature == "S":
            # Try sS_inv first (invariant version), fall back to sS_raw, then sS
            s1 = axes1.get("sS_inv", axes1.get("sS_raw", axes1.get("sS", np.array([]))))
            s2 = axes2.get("sS_inv", axes2.get("sS_raw", axes2.get("sS", np.array([]))))
            if s1.size == 0 or s2.size == 0:
                return None
        elif feature == "T":
            s1 = axes1.get("sT", np.array([]))
            s2 = axes2.get("sT", np.array([]))
            if s1.size == 0 or s2.size == 0:
                return None
        else:
            return None
        
        s1 = s1.ravel() if s1.ndim > 1 and s1.shape[1] == 1 else (s1[:, 0] if s1.ndim > 1 else s1)
        s2 = s2.ravel() if s2.ndim > 1 and s2.shape[1] == 1 else (s2[:, 0] if s2.ndim > 1 else s2)
        
        if s1.size == 0 or s2.size == 0:
            return None
        
        time = cache1["time"].astype(float)
        
        # === Apply proper normalization ===
        # Determine normalization mode from axes if auto
        if norm is None or norm == "auto":
            norm_mode = get_axes_norm_mode(axes1)
        else:
            norm_mode = norm
        
        # Get baseline window for normalization
        if norm_baseline_win is not None:
            bwin = norm_baseline_win
        elif norm_mode == "baseline":
            bwin = get_axes_baseline_win(axes1)
        else:
            bwin = None
        
        # Get normalization parameters from axes
        axes_norm1 = get_axes_norm(axes1)
        axes_norm2 = get_axes_norm(axes2)
        
        # Get normalized data
        Z1, _ = get_Z(cache1, time, keep, norm_mode, bwin, axes_norm1)
        Z2, _ = get_Z(cache2, time, keep, norm_mode, bwin, axes_norm2)
        
        if feature == "C":
            C = cache1["lab_C"][keep].astype(float)
            C = np.sign(C)
            if np.unique(C).size < 2:
                return None
        elif feature == "R":
            R = cache1.get("lab_R", np.full(len(keep), np.nan))[keep].astype(float)
            if not np.any(np.isfinite(R)):
                return None
            # For direction, encode as sign or use a binary encoding
            # Round to nearest integer for discrete directions
            R_rounded = np.round(R)
            R_vals = np.unique(R_rounded[np.isfinite(R_rounded)])
            if R_vals.size < 2:
                return None
            # Use sign of centered R (positive vs negative directions)
            R_centered = R_rounded - np.nanmedian(R_rounded)
            R_sign = np.sign(R_centered)
            # If all signs are same, try alternative encoding
            if np.unique(R_sign[np.isfinite(R_sign)]).size < 2:
                # Try splitting by median
                R_median = np.nanmedian(R_rounded)
                R_sign = np.where(R_rounded > R_median, 1, -1)
                if np.unique(R_sign[np.isfinite(R_sign)]).size < 2:
                    return None
        elif feature == "S":
            S = cache1.get("lab_S", np.full(len(keep), np.nan))[keep].astype(float)
            if not np.any(np.isfinite(S)):
                return None
            S = np.sign(S)
            if np.unique(S).size < 2:
                return None
        elif feature == "T":
            T = cache1.get("lab_T", np.full(len(keep), np.nan))[keep].astype(float)
            if not np.any(np.isfinite(T)):
                return None
            T = np.sign(T)
            if np.unique(T).size < 2:
                return None
        
        # Projections
        y1 = project_1d(Z1, s1)  # (N,B)
        y2 = project_1d(Z2, s2)  # (N,B)
        
        # Signed evidence
        if feature == "C":
            e1 = (C[:, None] * y1)
            e2 = (C[:, None] * y2)
        elif feature == "R":
            e1 = (R_sign[:, None] * y1)
            e2 = (R_sign[:, None] * y2)
        elif feature == "S":
            e1 = (S[:, None] * y1)
            e2 = (S[:, None] * y2)
        elif feature == "T":
            e1 = (T[:, None] * y1)
            e2 = (T[:, None] * y2)
        
        # Smooth
        dt = float(np.nanmedian(np.diff(time)))
        sigma_bins = (smooth_ms / 1000.0) / dt if dt > 0 else 0.0
        e1 = smooth_timewise(e1, sigma_bins)
        e2 = smooth_timewise(e2, sigma_bins)
        
        # Calculate thresholds
        thr1 = np.array([calculate_threshold(e1[i], time, baseline, k_sigma)
                         for i in range(e1.shape[0])], dtype=float)
        thr2 = np.array([calculate_threshold(e2[i], time, baseline, k_sigma)
                         for i in range(e2.shape[0])], dtype=float)
        
        # Onsets
        t1 = np.array([onset_time_per_trial(e1[i], time, baseline, search, k_sigma, runlen)
                       for i in range(e1.shape[0])], dtype=float)
        t2 = np.array([onset_time_per_trial(e2[i], time, baseline, search, k_sigma, runlen)
                       for i in range(e2.shape[0])], dtype=float)
        
        # Keep trials where both exist
        good = np.isfinite(t1) & np.isfinite(t2)
        n_good = int(good.sum())
        
        if n_good < 10:
            return None
        
        return {
            "sid": sid,
            "area1": a1,
            "area2": a2,
            "feature": feature,
            "t1": t1,
            "t2": t2,
            "good": good,
            "n_good": n_good,
            "thr1": thr1,
            "thr2": thr2,
        }
    except Exception as e:
        print(f"[warning] Failed to process {sid} {area1}-{area2} {feature}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Comprehensive trial onset analysis: category, direction, saccade, and target, all pairs, all sessions.")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--sid_list", default="sid_list.txt", help="File with list of session IDs")
    ap.add_argument("--align", nargs="+", choices=["stim", "sacc", "targ"], default=["stim"],
                    help="Alignment(s) to process (default: stim). Can specify multiple: --align stim sacc targ")
    
    # Alignment-specific orientation parameters
    ap.add_argument("--orientation_stim", choices=["vertical","horizontal","pooled"], default="vertical",
                    help="Orientation for stim alignment (default: vertical)")
    ap.add_argument("--orientation_sacc", choices=["vertical","horizontal","pooled"], default="horizontal",
                    help="Orientation for sacc alignment (default: horizontal)")
    ap.add_argument("--orientation_targ", choices=["vertical","horizontal","pooled"], default="vertical",
                    help="Orientation for targ alignment (default: vertical)")
    
    # Alignment-specific parameters with defaults
    ap.add_argument("--pt_min_ms_stim", type=float, default=200.0,
                    help="PT threshold (ms) for stim alignment (default: 200.0)")
    ap.add_argument("--pt_min_ms_sacc", type=float, default=200.0,
                    help="PT threshold (ms) for sacc alignment (default: 200.0)")
    ap.add_argument("--pt_min_ms_targ", type=float, default=200.0,
                    help="PT threshold (ms) for targ alignment (default: 200.0)")
    
    ap.add_argument("--axes_tag_stim", default="axes_peakbin_stimCR-stim-vertical-20mssw",
                    help="Axes tag for stim alignment (default: axes_peakbin_stimCR-stim-vertical-20mssw)")
    ap.add_argument("--axes_tag_sacc", default="axes_peakbin_saccS-sacc-horizontal-20mssw",
                    help="Axes tag for sacc alignment (default: axes_peakbin_saccS-sacc-horizontal-20mssw)")
    ap.add_argument("--axes_tag_targ", default="axes_sweep-targ-vertical",
                    help="Axes tag for targ alignment (default: axes_sweep-targ-vertical)")
    
    ap.add_argument("--baseline_stim", default="-0.20:0.00",
                    help="Baseline window (sec) for stim alignment (default: -0.20:0.00)")
    ap.add_argument("--baseline_sacc", default="-0.35:-0.20",
                    help="Baseline window (sec) for sacc alignment (default: -0.35:-0.20)")
    ap.add_argument("--baseline_targ", default="-0.15:0.00",
                    help="Baseline window (sec) for targ alignment (default: -0.15:0.00)")
    
    ap.add_argument("--search_stim", default="0.00:0.5",
                    help="Search window (sec) for stim alignment (default: 0.00:0.60)")
    ap.add_argument("--search_sacc", default="-0.20:0.20",
                    help="Search window (sec) for sacc alignment (default: -0.20:0.20)")
    ap.add_argument("--search_targ", default="0.00:0.35",
                    help="Search window (sec) for targ alignment (default: 0.00:0.35)")
    
    ap.add_argument("--k_sigma", type=float, default=4,
                    help="Threshold = baseline mean + k_sigma*baseline std")
    ap.add_argument("--runlen", type=int, default=5,
                    help="Consecutive bins above threshold")
    ap.add_argument("--smooth_ms", type=float, default=20.0,
                    help="Gaussian smoothing sigma in ms")
    ap.add_argument("--qc_threshold", type=float, default=0.65,
                    help="Default QC threshold for filtering (default: 0.65). Can be overridden by feature-specific thresholds.")
    ap.add_argument("--qc_threshold_C", type=float, default=None,
                    help="QC threshold for category (C) feature (default: uses --qc_threshold)")
    ap.add_argument("--qc_threshold_R", type=float, default=0.6,
                    help="QC threshold for direction (R) feature (default: 0.6)")
    ap.add_argument("--qc_threshold_S", type=float, default=None,
                    help="QC threshold for saccade (S) feature (default: uses --qc_threshold)")
    ap.add_argument("--qc_threshold_T", type=float, default=None,
                    help="QC threshold for target (T) feature (default: uses --qc_threshold)")
    ap.add_argument("--qc_root", default=None,
                    help="Root directory for QC data (default: uses --out_root). "
                         "Use this for mixed training: --qc_root out_nofilter to evaluate QC on all trials "
                         "while using correct-only trials for onset analysis.")
    ap.add_argument("--qc_tag_stim", default=None,
                    help="QC tag for stim alignment (default: uses --axes_tag_stim). "
                         "Useful for mixed training to point to QC computed on all trials.")
    ap.add_argument("--qc_tag_sacc", default=None,
                    help="QC tag for sacc alignment (default: uses --axes_tag_sacc).")
    ap.add_argument("--tag", default="trialonset_comprehensive_20mssw",
                    help="Output subfolder name (default: trialonset_comprehensive_20mssw)")
    
    # === NEW: normalization args ===
    ap.add_argument("--norm_stim", choices=["auto", "global", "baseline", "none"], default="auto",
                    help="Normalization for stim: 'auto' (use axes meta), 'global', 'baseline', 'none'")
    ap.add_argument("--norm_sacc", choices=["auto", "global", "baseline", "none"], default="auto",
                    help="Normalization for sacc: 'auto' (use axes meta), 'global', 'baseline', 'none'")
    ap.add_argument("--norm_targ", choices=["auto", "global", "baseline", "none"], default="auto",
                    help="Normalization for targ: 'auto' (use axes meta), 'global', 'baseline', 'none'")
    ap.add_argument("--norm_baseline_win_stim", default=None,
                    help="Baseline window for stim normalization 'a:b' (uses axes meta if not specified)")
    ap.add_argument("--norm_baseline_win_sacc", default=None,
                    help="Baseline window for sacc normalization 'a:b' (uses axes meta if not specified)")
    ap.add_argument("--norm_baseline_win_targ", default=None,
                    help="Baseline window for targ normalization 'a:b' (uses axes meta if not specified)")
    
    # === NEW: rebinning args (must match rebin_factor used in train_axes.py) ===
    ap.add_argument("--rebin_factor_stim", type=int, default=1,
                    help="Rebin factor for stim (1=no rebin, 2=10ms→20ms). Must match axes training.")
    ap.add_argument("--rebin_factor_sacc", type=int, default=1,
                    help="Rebin factor for sacc (1=no rebin, 2=5ms→10ms, 4=5ms→20ms). Must match axes training.")
    ap.add_argument("--rebin_factor_targ", type=int, default=1,
                    help="Rebin factor for targ (1=no rebin). Must match axes training.")
    
    # === NEW: sliding window args (alternative to rebinning, must match train_axes.py) ===
    ap.add_argument("--sliding_window_ms_stim", type=float, default=20.0,
                    help="Sliding window width in ms for stim (default: 20).")
    ap.add_argument("--sliding_step_ms_stim", type=float, default=10.0,
                    help="Sliding window step in ms for stim (default: 10).")
    ap.add_argument("--sliding_window_ms_sacc", type=float, default=20.0,
                    help="Sliding window width in ms for sacc (default: 20).")
    ap.add_argument("--sliding_step_ms_sacc", type=float, default=10.0,
                    help="Sliding window step in ms for sacc (default: 10).")
    ap.add_argument("--sliding_window_ms_targ", type=float, default=0.0,
                    help="Sliding window width in ms for targ. If > 0, uses sliding window.")
    ap.add_argument("--sliding_step_ms_targ", type=float, default=0.0,
                    help="Sliding window step in ms for targ.")
    
    args = ap.parse_args()
    
    out_root = Path(args.out_root)
    aligns = args.align
    
    # Read session IDs
    sid_list_path = Path(args.sid_list)
    if not sid_list_path.exists():
        raise SystemExit(f"Session list file not found: {sid_list_path}")
    
    sids = []
    with open(sid_list_path) as f:
        for line in f:
            sid = line.strip()
            if sid and not sid.startswith("#"):
                sids.append(sid)
    
    print(f"[info] Processing {len(sids)} sessions")
    print(f"[info] Alignments: {aligns}")
    
    # Define pairs
    # Note: SC areas are named MSC or SSC depending on monkey, but we search for "SC"
    pairs = [
        ("SC", "LIP"),
        ("SC", "FEF"),
        ("FEF", "LIP"),
    ]
    
    # Process each alignment separately
    for align in aligns:
        print(f"\n{'='*60}")
        print(f"[align={align}] Processing alignment")
        print(f"{'='*60}")
        
        # Get alignment-specific parameters
        if align == "stim":
            pt_min_ms = args.pt_min_ms_stim
            axes_tag = args.axes_tag_stim
            baseline = parse_window(args.baseline_stim)
            search = parse_window(args.search_stim)
            orientation = args.orientation_stim
            features = ["C", "R"]  # Category and Direction
        elif align == "sacc":
            pt_min_ms = args.pt_min_ms_sacc
            axes_tag = args.axes_tag_sacc
            baseline = parse_window(args.baseline_sacc)
            search = parse_window(args.search_sacc)
            orientation = args.orientation_sacc
            features = ["S"]  # Saccade
        elif align == "targ":
            pt_min_ms = args.pt_min_ms_targ
            axes_tag = args.axes_tag_targ
            baseline = parse_window(args.baseline_targ)
            search = parse_window(args.search_targ)
            orientation = args.orientation_targ
            features = ["T"]  # Target
        
        # === Get normalization settings for this alignment ===
        if align == "stim":
            norm = None if args.norm_stim == "auto" else args.norm_stim
            norm_baseline_win = parse_window(args.norm_baseline_win_stim) if args.norm_baseline_win_stim else None
            rebin_factor = args.rebin_factor_stim
            sliding_window_ms = args.sliding_window_ms_stim
            sliding_step_ms = args.sliding_step_ms_stim
            native_bin_ms = 10.0  # STIM native bin size
        elif align == "sacc":
            norm = None if args.norm_sacc == "auto" else args.norm_sacc
            norm_baseline_win = parse_window(args.norm_baseline_win_sacc) if args.norm_baseline_win_sacc else None
            rebin_factor = args.rebin_factor_sacc
            sliding_window_ms = args.sliding_window_ms_sacc
            sliding_step_ms = args.sliding_step_ms_sacc
            native_bin_ms = 5.0  # SACC native bin size
        else:  # targ
            norm = None if args.norm_targ == "auto" else args.norm_targ
            norm_baseline_win = parse_window(args.norm_baseline_win_targ) if args.norm_baseline_win_targ else None
            rebin_factor = args.rebin_factor_targ
            sliding_window_ms = args.sliding_window_ms_targ
            sliding_step_ms = args.sliding_step_ms_targ
            native_bin_ms = 10.0  # TARG native bin size (same as stim)
        
        # Compute sliding window bins from ms parameters
        sliding_window_bins = 0
        sliding_step_bins = 0
        if sliding_window_ms > 0 and sliding_step_ms > 0:
            if sliding_window_ms % native_bin_ms != 0:
                raise SystemExit(f"sliding_window_ms ({sliding_window_ms}) must be multiple of native bin ({native_bin_ms}ms)")
            if sliding_step_ms % native_bin_ms != 0:
                raise SystemExit(f"sliding_step_ms ({sliding_step_ms}) must be multiple of native bin ({native_bin_ms}ms)")
            sliding_window_bins = int(round(sliding_window_ms / native_bin_ms))
            sliding_step_bins = int(round(sliding_step_ms / native_bin_ms))
            print(f"[align={align}] sliding window={sliding_window_ms}ms ({sliding_window_bins} bins), "
                  f"step={sliding_step_ms}ms ({sliding_step_bins} bins)")
            rebin_factor = 1  # Disable rebinning when using sliding window
        
        # === Get QC root and tag for this alignment ===
        qc_root = Path(args.qc_root) if args.qc_root else None
        if align == "stim":
            qc_tag = args.qc_tag_stim
        elif align == "sacc":
            qc_tag = args.qc_tag_sacc
        else:  # targ - no separate qc_tag_targ, use axes_tag
            qc_tag = None
        
        print(f"[align={align}] pt_min_ms={pt_min_ms}, axes_tag={axes_tag}")
        if qc_root is not None:
            print(f"[align={align}] qc_root={qc_root} (separate from out_root)")
        if qc_tag is not None:
            print(f"[align={align}] qc_tag={qc_tag} (separate from axes_tag)")
        if rebin_factor > 1:
            print(f"[align={align}] rebin_factor={rebin_factor}")
        print(f"[align={align}] baseline={baseline}, search={search}")
        print(f"[align={align}] orientation={orientation}")
        print(f"[align={align}] features={features}")
        
        # Get feature-specific QC thresholds
        def get_qc_threshold(feature: str) -> float:
            """Get QC threshold for a specific feature."""
            if feature == "C":
                return args.qc_threshold_C if args.qc_threshold_C is not None else args.qc_threshold
            elif feature == "R":
                return args.qc_threshold_R if args.qc_threshold_R is not None else args.qc_threshold
            elif feature == "S":
                return args.qc_threshold_S if args.qc_threshold_S is not None else args.qc_threshold
            elif feature == "T":
                return args.qc_threshold_T if args.qc_threshold_T is not None else args.qc_threshold
            else:
                return args.qc_threshold
        
        # Print feature-specific thresholds
        for feature in features:
            threshold = get_qc_threshold(feature)
            feature_name = {"C": "category", "R": "direction", "S": "saccade", "T": "target"}.get(feature, feature)
            print(f"[align={align}] {feature_name} ({feature}) qc_threshold={threshold}")
        
        # Process all sessions
        all_results = {}  # (feature, area1, area2) -> list of results
        
        for feature in features:
            qc_threshold_feature = get_qc_threshold(feature)
            for area1, area2 in pairs:
                key = (feature, area1, area2)
                all_results[key] = []
                
                for sid in sids:
                    result = process_one_session(
                        out_root, align, sid, orientation, pt_min_ms,
                        axes_tag, baseline, search, args.k_sigma, args.runlen,
                        args.smooth_ms, feature, area1, area2, args.tag,
                        qc_threshold=qc_threshold_feature,
                        norm=norm,
                        norm_baseline_win=norm_baseline_win,
                        rebin_factor=rebin_factor,
                        sliding_window_bins=sliding_window_bins,
                        sliding_step_bins=sliding_step_bins,
                        qc_root=qc_root,
                        qc_tag=qc_tag,
                    )
                    if result is not None:
                        all_results[key].append(result)
                        print(f"[{sid}] {area1}-{area2} {feature}: {result['n_good']} trials")
        
        # Create output directory
        out_dir = out_root / align / "trialtiming" / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration as JSON
        config = {
            "created": datetime.now().isoformat(),
            "script": "cli/trial_onset_comprehensive.py",
            "align": align,
            "orientation": orientation,
            "pt_min_ms": pt_min_ms,
            "axes_tag": axes_tag,
            "baseline": list(baseline),
            "search": list(search),
            "k_sigma": float(args.k_sigma),
            "runlen": int(args.runlen),
            "smooth_ms": float(args.smooth_ms),
            "qc_threshold": float(args.qc_threshold),
            "qc_threshold_C": float(args.qc_threshold_C) if args.qc_threshold_C is not None else None,
            "qc_threshold_R": float(args.qc_threshold_R) if args.qc_threshold_R is not None else None,
            "qc_threshold_S": float(args.qc_threshold_S) if args.qc_threshold_S is not None else None,
            "qc_threshold_T": float(args.qc_threshold_T) if args.qc_threshold_T is not None else None,
            "tag": args.tag,
            "features": features,
            "pairs": [[a1, a2] for a1, a2 in pairs],
            "n_sessions": len(sids),
            "session_ids": sids,
            "norm": norm if norm else "auto",
            "norm_baseline_win": list(norm_baseline_win) if norm_baseline_win else None,
            "rebin_factor": rebin_factor,
            "sliding_window_ms": sliding_window_ms if sliding_window_bins > 0 else None,
            "sliding_step_ms": sliding_step_ms if sliding_step_bins > 0 else None,
            "sliding_window_bins": sliding_window_bins if sliding_window_bins > 0 else None,
            "sliding_step_bins": sliding_step_bins if sliding_step_bins > 0 else None,
        }
        config_path = out_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[ok] wrote {config_path}")
        
        # Aggregate by monkey and create summary plots
        for feature in features:
            if feature == "C":
                feature_name = "category"
            elif feature == "R":
                feature_name = "direction"
            elif feature == "S":
                feature_name = "saccade"
            elif feature == "T":
                feature_name = "target"
            else:
                feature_name = feature
            
            for area1, area2 in pairs:
                key = (feature, area1, area2)
                results = all_results[key]
                
                if len(results) == 0:
                    continue
                
                # Aggregate by monkey
                # monkey_data[monkey] = (t1_list, t2_list, sids_list)
                monkey_data = {"M": ([], [], []), "S": ([], [], [])}
                
                for result in results:
                    sid = result["sid"]
                    monkey = get_monkey(sid)
                    if monkey not in monkey_data:
                        continue
                    
                    t1 = result["t1"][result["good"]] * 1000.0  # Convert to ms
                    t2 = result["t2"][result["good"]] * 1000.0
                    
                    valid = np.isfinite(t1) & np.isfinite(t2)
                    if valid.sum() > 0:
                        monkey_data[monkey][0].append(t1[valid])
                        monkey_data[monkey][1].append(t2[valid])
                        monkey_data[monkey][2].append(sid)
                
                # Create plots for each monkey
                for monkey, (t1_list, t2_list, sids_list) in monkey_data.items():
                    if len(t1_list) == 0:
                        continue
                    
                    t1_all = np.concatenate(t1_list)
                    t2_all = np.concatenate(t2_list)
                    n_trials = len(t1_all)
                    n_sessions = len(set(sids_list))
                    
                    # Calculate statistics (pooled trials)
                    dt_ms = t2_all - t1_all
                    median_dt = np.nanmedian(dt_ms)
                    mean_dt = np.nanmean(dt_ms)
                    
                    # --- pooled-trials p-values (descriptive) ---
                    p_pool_2 = trial_signflip_pvalue(dt_ms, n_perm=20000, alternative="two-sided", seed=0)
                    p_pool_1 = trial_signflip_pvalue(dt_ms, n_perm=20000, alternative="greater", seed=0)  # area2 later
                    
                    # Define area names for display
                    monkey_name = "Monkey M" if monkey == "M" else "Monkey S"
                    area_prefix = "M" if monkey == "M" else "S"
                    # SC always stays as "SC", FEF and LIP get monkey prefix
                    a1_name = "SC" if area1 == "SC" else f"{area_prefix}{area1}"
                    a2_name = "SC" if area2 == "SC" else f"{area_prefix}{area2}"
                    
                    # Print results
                    print(f"[{monkey}] {area1}-{area2} {feature_name}: "
                          f"pooled mean(dt)={p_pool_2['obs']:.2f} ms (N_trials={p_pool_2['n_trials']}), "
                          f"p(two)={p_pool_2['p']:.4g}, p(later)={p_pool_1['p']:.4g}")
                    
                    # Create plot with explicit 5x5 inch plot area
                    # This ensures exact size matching with panel plots in summarize_flow_across_sessions.py
                    plot_size_in = 5.0  # 5x5 inch plot area
                    margin_left_in = 1.0
                    margin_right_in = 0.5
                    margin_bottom_in = 0.8
                    margin_top_in = 0.7
                    
                    fig_width = plot_size_in + margin_left_in + margin_right_in
                    fig_height = plot_size_in + margin_bottom_in + margin_top_in
                    
                    fig = plt.figure(figsize=(fig_width, fig_height))
                    # Position axes explicitly: [left, bottom, width, height] in figure fraction
                    ax = fig.add_axes([
                        margin_left_in / fig_width,
                        margin_bottom_in / fig_height,
                        plot_size_in / fig_width,
                        plot_size_in / fig_height
                    ])
                    ax.set_aspect('equal', adjustable='box')  # Square plot area
                    
                    # Clean area names for axis labels (remove monkey prefix)
                    a1_label = area1
                    a2_label = area2
                    
                    # Scatter plot (larger markers)
                    ax.plot(t1_all, t2_all, "k.", ms=7, alpha=0.5, label=f"N={n_trials} trials")
                    
                    # Mark mean and median
                    mean_t1 = np.nanmean(t1_all)
                    mean_t2 = np.nanmean(t2_all)
                    median_t1 = np.nanmedian(t1_all)
                    median_t2 = np.nanmedian(t2_all)
                    
                    # Plot median first (lower z-order), then mean on top
                    ax.plot(median_t1, median_t2, "bs", ms=12, markerfacecolor="blue", 
                            markeredgecolor="darkblue", markeredgewidth=2, zorder=5)
                    ax.plot(mean_t1, mean_t2, "ro", ms=12, markerfacecolor="red", 
                            markeredgecolor="darkred", markeredgewidth=2, zorder=6)
                    
                    # Create combined legend entry for mean/median
                    from matplotlib.lines import Line2D
                    legend_mean = Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                                         markeredgecolor='darkred', markersize=10, markeredgewidth=1.5, label='mean')
                    legend_median = Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                                           markeredgecolor='darkblue', markersize=10, markeredgewidth=1.5, label='median')
                    
                    # Use search window for axis limits (convert from sec to ms)
                    axis_lo = search[0] * 1000.0  # e.g., -0.30 sec = -300 ms
                    axis_hi = search[1] * 1000.0  # e.g., 0.20 sec = 200 ms
                    
                    # Diagonal line (using full axis range)
                    ax.plot([axis_lo, axis_hi], [axis_lo, axis_hi], "r--", lw=1.5, label="y=x", alpha=0.7)
                    
                    # Set axis limits
                    ax.set_xlim(axis_lo, axis_hi)
                    ax.set_ylim(axis_lo, axis_hi)
                    
                    # Set ticks every 100ms (compute based on range)
                    tick_lo = int(np.ceil(axis_lo / 100) * 100)  # first tick at round 100
                    tick_hi = int(np.floor(axis_hi / 100) * 100)  # last tick at round 100
                    tick_values = list(range(tick_lo, tick_hi + 1, 100))
                    ax.set_xticks(tick_values)
                    ax.set_yticks(tick_values)
                    ax.tick_params(axis='both', which='major', labelsize=16)
                    
                    # Axis labels (bigger font, clean names)
                    ax.set_xlabel(f"{a1_label} Latency (ms)", fontsize=18)
                    ax.set_ylabel(f"{a2_label} Latency (ms)", fontsize=18)
                    
                    # Title with all statistics
                    title = f"{monkey_name} ({a1_name} vs {a2_name}) - {feature_name} [{align}]\n"
                    title += f"{n_sessions} sessions, {n_trials} trials\n"
                    title += f"median({a2_label}-{a1_label})={median_dt:.1f} ms, "
                    title += f"mean({a2_label}-{a1_label})={mean_dt:.1f} ms\n"
                    title += f"p(two)={p_pool_2['p']:.3g}, p(later)={p_pool_1['p']:.3g}"
                    ax.set_title(title, fontsize=11)
                    
                    # Custom legend with mean/median on same line
                    from matplotlib.legend_handler import HandlerTuple
                    legend_scatter = Line2D([0], [0], marker='', linestyle='', label=f'N={n_trials} trials')
                    legend_diag = Line2D([0], [0], linestyle='--', color='red', alpha=0.7, lw=1.5, label='y=x')
                    ax.legend(handles=[legend_scatter, (legend_mean, legend_median), legend_diag],
                              labels=[f'N={n_trials} trials', 'mean / median', 'y=x'],
                              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
                              frameon=False, fontsize=15, loc='lower right',
                              handletextpad=0.5, labelspacing=0.4)
                    # No grid
                    ax.grid(False)
                    # No tight_layout() - we use explicit axes positioning
                    
                    # Save
                    out_png = out_dir / f"monkey_{monkey}_{area1}_vs_{area2}_{feature_name}_summary.png"
                    out_pdf = out_dir / f"monkey_{monkey}_{area1}_vs_{area2}_{feature_name}_summary.pdf"
                    out_svg = out_dir / f"monkey_{monkey}_{area1}_vs_{area2}_{feature_name}_summary.svg"
                    fig.savefig(out_png, dpi=500)
                    fig.savefig(out_pdf)
                    fig.savefig(out_svg)
                    plt.close(fig)
                    print(f"[ok] wrote {out_png}, {out_pdf}, {out_svg}")
                    
                    # Save data
                    out_npz = out_dir / f"monkey_{monkey}_{area1}_vs_{area2}_{feature_name}_summary.npz"
                    # Get feature-specific QC threshold
                    qc_threshold_feature = get_qc_threshold(feature)
                    meta_dict = dict(
                        monkey=monkey,
                        area1=area1,
                        area2=area2,
                        feature=feature,
                        feature_name=feature_name,
                        align=align,
                        n_trials=n_trials,
                        n_sessions=n_sessions,
                        median_dt_ms=float(median_dt),
                        mean_dt_ms=float(mean_dt),
                        qc_threshold=float(qc_threshold_feature),
                        pooled_mean_dt_ms=float(p_pool_2["obs"]),
                        pooled_p_two_sided=float(p_pool_2["p"]),
                        pooled_p_area2_later=float(p_pool_1["p"]),
                        pooled_n_trials_for_p=int(p_pool_2["n_trials"]),
                    )
                    np.savez_compressed(
                        out_npz,
                        t1_ms=t1_all,
                        t2_ms=t2_all,
                        dt_ms=dt_ms,
                        sids=np.array(sids_list),
                        meta=meta_dict
                    )
                    print(f"[ok] wrote {out_npz}")
            
            # === CREATE COMBINED FEF vs (LIP + SC) PLOT ===
            # Combines FEF-LIP and SC-FEF pairs into one figure
            # X-axis: FEF latency, Y-axis: LIP (black) and SC (green)
            for monkey in ["M", "S"]:
                # Get data for FEF-LIP pair (area1=FEF, area2=LIP, so t1=FEF, t2=LIP)
                key_fef_lip = (feature, "FEF", "LIP")
                results_fef_lip = all_results.get(key_fef_lip, [])
                
                # Get data for SC-FEF pair (area1=SC, area2=FEF, so t1=SC, t2=FEF)
                key_sc_fef = (feature, "SC", "FEF")
                results_sc_fef = all_results.get(key_sc_fef, [])
                
                # Aggregate FEF-LIP data for this monkey
                t_lip_list, t_fef_lip_list, sids_fef_lip = [], [], []
                for result in results_fef_lip:
                    sid = result["sid"]
                    if get_monkey(sid) != monkey:
                        continue
                    t_fef = result["t1"][result["good"]] * 1000.0  # FEF latency
                    t_lip = result["t2"][result["good"]] * 1000.0  # LIP latency
                    valid = np.isfinite(t_fef) & np.isfinite(t_lip)
                    if valid.sum() > 0:
                        t_lip_list.append(t_lip[valid])
                        t_fef_lip_list.append(t_fef[valid])
                        sids_fef_lip.append(sid)
                
                # Aggregate SC-FEF data for this monkey
                t_sc_list, t_fef_sc_list, sids_sc_fef = [], [], []
                for result in results_sc_fef:
                    sid = result["sid"]
                    if get_monkey(sid) != monkey:
                        continue
                    t_sc = result["t1"][result["good"]] * 1000.0   # SC latency
                    t_fef = result["t2"][result["good"]] * 1000.0  # FEF latency
                    valid = np.isfinite(t_sc) & np.isfinite(t_fef)
                    if valid.sum() > 0:
                        t_sc_list.append(t_sc[valid])
                        t_fef_sc_list.append(t_fef[valid])
                        sids_sc_fef.append(sid)
                
                # Skip if no data for either pair
                if len(t_lip_list) == 0 and len(t_sc_list) == 0:
                    continue
                
                # Concatenate data
                t_lip_all = np.concatenate(t_lip_list) if t_lip_list else np.array([])
                t_fef_lip_all = np.concatenate(t_fef_lip_list) if t_fef_lip_list else np.array([])
                t_sc_all = np.concatenate(t_sc_list) if t_sc_list else np.array([])
                t_fef_sc_all = np.concatenate(t_fef_sc_list) if t_fef_sc_list else np.array([])
                
                n_trials_lip = len(t_lip_all)
                n_trials_sc = len(t_sc_all)
                n_sessions_lip = len(set(sids_fef_lip))
                n_sessions_sc = len(set(sids_sc_fef))
                
                # Skip if not enough data
                if n_trials_lip < 10 and n_trials_sc < 10:
                    continue
                
                # Statistics: LIP - FEF and SC - FEF (positive = LIP/SC later than FEF)
                dt_lip_ms = t_lip_all - t_fef_lip_all if n_trials_lip > 0 else np.array([])  # LIP - FEF
                dt_sc_ms = t_sc_all - t_fef_sc_all if n_trials_sc > 0 else np.array([])      # SC - FEF
                
                median_dt_lip = np.nanmedian(dt_lip_ms) if n_trials_lip > 0 else np.nan
                median_dt_sc = np.nanmedian(dt_sc_ms) if n_trials_sc > 0 else np.nan
                mean_dt_lip = np.nanmean(dt_lip_ms) if n_trials_lip > 0 else np.nan
                mean_dt_sc = np.nanmean(dt_sc_ms) if n_trials_sc > 0 else np.nan
                
                # Naming
                monkey_name = "Monkey M" if monkey == "M" else "Monkey S"
                area_prefix = "M" if monkey == "M" else "S"
                
                # Create combined plot
                plot_size_in = 5.0
                margin_left_in = 1.0
                margin_right_in = 0.5
                margin_bottom_in = 0.8
                margin_top_in = 0.9  # Extra space for title
                
                fig_width = plot_size_in + margin_left_in + margin_right_in
                fig_height = plot_size_in + margin_bottom_in + margin_top_in
                
                fig = plt.figure(figsize=(fig_width, fig_height))
                ax = fig.add_axes([
                    margin_left_in / fig_width,
                    margin_bottom_in / fig_height,
                    plot_size_in / fig_width,
                    plot_size_in / fig_height
                ])
                ax.set_aspect('equal', adjustable='box')
                
                # Plot LIP vs FEF (darker red dots) - X=FEF, Y=LIP
                if n_trials_lip > 0:
                    ax.plot(t_fef_lip_all, t_lip_all, ".", color="#c00a37", ms=7, alpha=0.5, 
                            label=f"LIP vs FEF (N={n_trials_lip})")
                
                # Plot SC vs FEF (darker purple dots) - X=FEF, Y=SC
                if n_trials_sc > 0:
                    ax.plot(t_fef_sc_all, t_sc_all, ".", color="#7b4fa3", ms=7, alpha=0.5,
                            label=f"SC vs FEF (N={n_trials_sc})")
                
                # Mark means with different shapes (square for LIP, diamond for SC)
                # Filled with bright colors and black border to stand out from scatter
                from matplotlib.lines import Line2D
                if n_trials_sc > 0:
                    mean_fef_sc = np.nanmean(t_fef_sc_all)
                    mean_sc = np.nanmean(t_sc_all)
                    ax.plot(mean_fef_sc, mean_sc, "D", ms=14, markerfacecolor="#b388dd",
                            markeredgecolor="black", markeredgewidth=2, zorder=6)
                
                if n_trials_lip > 0:
                    mean_fef_lip = np.nanmean(t_fef_lip_all)
                    mean_lip = np.nanmean(t_lip_all)
                    ax.plot(mean_fef_lip, mean_lip, "s", ms=15, markerfacecolor="#ff6b6b",
                            markeredgecolor="black", markeredgewidth=2, zorder=7)
                
                # Axis limits from search window
                axis_lo = search[0] * 1000.0
                axis_hi = search[1] * 1000.0
                
                # Diagonal line
                ax.plot([axis_lo, axis_hi], [axis_lo, axis_hi], "r--", lw=1.5, alpha=0.7)
                
                ax.set_xlim(axis_lo, axis_hi)
                ax.set_ylim(axis_lo, axis_hi)
                
                # Ticks
                tick_lo = int(np.ceil(axis_lo / 100) * 100)
                tick_hi = int(np.floor(axis_hi / 100) * 100)
                tick_values = list(range(tick_lo, tick_hi + 1, 100))
                ax.set_xticks(tick_values)
                ax.set_yticks(tick_values)
                ax.tick_params(axis='both', which='major', labelsize=16)
                
                # Labels
                ax.set_xlabel("FEF Latency (ms)", fontsize=18)
                ax.set_ylabel("LIP / SC Latency (ms)", fontsize=18)
                
                # Title
                title = f"{monkey_name} - (LIP & SC) vs FEF - {feature_name} [{align}]\n"
                if n_trials_lip > 0:
                    title += f"LIP-FEF: {n_sessions_lip} sess, mean(LIP-FEF)={mean_dt_lip:.1f}ms\n"
                if n_trials_sc > 0:
                    title += f"SC-FEF: {n_sessions_sc} sess, mean(SC-FEF)={mean_dt_sc:.1f}ms"
                ax.set_title(title, fontsize=11)
                
                # Legend
                legend_lip = Line2D([0], [0], marker='.', color='#c00a37', linestyle='', 
                                    markersize=10, alpha=0.7, label=f'LIP (N={n_trials_lip})')
                legend_sc = Line2D([0], [0], marker='.', color='#7b4fa3', linestyle='',
                                   markersize=10, alpha=0.7, label=f'SC (N={n_trials_sc})')
                legend_mean_lip = Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff6b6b',
                                         markeredgecolor='black', markersize=12, markeredgewidth=1.5, label='mean (LIP)')
                legend_mean_sc = Line2D([0], [0], marker='D', color='w', markerfacecolor='#b388dd',
                                        markeredgecolor='black', markersize=11, markeredgewidth=1.5, label='mean (SC)')
                legend_diag = Line2D([0], [0], linestyle='--', color='gray', alpha=0.7, lw=1.5, label='y=x')
                
                handles = []
                if n_trials_lip > 0:
                    handles.extend([legend_lip, legend_mean_lip])
                if n_trials_sc > 0:
                    handles.extend([legend_sc, legend_mean_sc])
                handles.append(legend_diag)
                
                ax.legend(handles=handles, frameon=False, fontsize=12, loc='lower right',
                          handletextpad=0.5, labelspacing=0.4)
                ax.grid(False)
                
                # Save combined figure
                out_png = out_dir / f"monkey_{monkey}_FEF_vs_LIP_SC_combined_{feature_name}_summary.png"
                out_pdf = out_dir / f"monkey_{monkey}_FEF_vs_LIP_SC_combined_{feature_name}_summary.pdf"
                out_svg = out_dir / f"monkey_{monkey}_FEF_vs_LIP_SC_combined_{feature_name}_summary.svg"
                fig.savefig(out_png, dpi=500)
                fig.savefig(out_pdf)
                fig.savefig(out_svg)
                plt.close(fig)
                print(f"[ok] wrote combined {out_png}")
                
                # Save combined data
                out_npz = out_dir / f"monkey_{monkey}_FEF_vs_LIP_SC_combined_{feature_name}_summary.npz"
                meta_dict = dict(
                    monkey=monkey,
                    feature=feature,
                    feature_name=feature_name,
                    align=align,
                    x_axis="FEF",
                    y_axis="LIP_and_SC",
                    n_trials_lip=n_trials_lip,
                    n_trials_sc=n_trials_sc,
                    n_sessions_lip=n_sessions_lip,
                    n_sessions_sc=n_sessions_sc,
                    mean_dt_lip_minus_fef_ms=float(mean_dt_lip) if not np.isnan(mean_dt_lip) else None,
                    mean_dt_sc_minus_fef_ms=float(mean_dt_sc) if not np.isnan(mean_dt_sc) else None,
                    median_dt_lip_minus_fef_ms=float(median_dt_lip) if not np.isnan(median_dt_lip) else None,
                    median_dt_sc_minus_fef_ms=float(median_dt_sc) if not np.isnan(median_dt_sc) else None,
                )
                np.savez_compressed(
                    out_npz,
                    t_lip_ms=t_lip_all,
                    t_fef_lip_ms=t_fef_lip_all,
                    t_sc_ms=t_sc_all,
                    t_fef_sc_ms=t_fef_sc_all,
                    dt_lip_minus_fef_ms=dt_lip_ms,  # LIP - FEF (positive = LIP later)
                    dt_sc_minus_fef_ms=dt_sc_ms,    # SC - FEF (positive = SC later)
                    sids_fef_lip=np.array(sids_fef_lip),
                    sids_sc_fef=np.array(sids_sc_fef),
                    meta=meta_dict
                )
                print(f"[ok] wrote combined {out_npz}")


if __name__ == "__main__":
    main()

