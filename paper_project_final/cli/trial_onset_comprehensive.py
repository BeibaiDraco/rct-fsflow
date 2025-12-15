#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
) -> bool:
    """
    Check if an area passes QC for a given feature.
    
    Returns True if QC passes (or if QC data is unavailable), False if QC fails.
    """
    # For axes_tag, the QC tag should match directly (e.g., axes_sweep-stim-vertical)
    # Check if QC directory exists with this tag
    qc_path = out_root / align / sid / "qc" / axes_tag
    if not qc_path.exists():
        # If no QC tag found, we can't check - default to passing
        # (could also return False to be strict, but user might want to be lenient)
        return True
    
    # Load QC data
    qc_data = load_qc_json(out_root, align, sid, axes_tag, area)
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
    qc_pass_a1 = check_area_qc(out_root, align, sid, axes_tag, a1, feature, qc_threshold)
    qc_pass_a2 = check_area_qc(out_root, align, sid, axes_tag, a2, feature, qc_threshold)
    
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
        Z1 = cache1["Z"][keep].astype(float)  # (N,B,U)
        Z2 = cache2["Z"][keep].astype(float)
        
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
    
    ap.add_argument("--axes_tag_stim", default="axes_sweep-stim-vertical",
                    help="Axes tag for stim alignment (default: axes_sweep-stim-vertical)")
    ap.add_argument("--axes_tag_sacc", default="axes_sweep-sacc-horizontal",
                    help="Axes tag for sacc alignment (default: axes_sweep-sacc-horizontal)")
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
    ap.add_argument("--search_sacc", default="-0.30:0.20",
                    help="Search window (sec) for sacc alignment (default: -0.30:0.20)")
    ap.add_argument("--search_targ", default="0.00:0.35",
                    help="Search window (sec) for targ alignment (default: 0.00:0.35)")
    
    ap.add_argument("--k_sigma", type=float, default=6,
                    help="Threshold = baseline mean + k_sigma*baseline std")
    ap.add_argument("--runlen", type=int, default=5,
                    help="Consecutive bins above threshold")
    ap.add_argument("--smooth_ms", type=float, default=20.0,
                    help="Gaussian smoothing sigma in ms")
    ap.add_argument("--qc_threshold", type=float, default=0.75,
                    help="Default QC threshold for filtering (default: 0.75). Can be overridden by feature-specific thresholds.")
    ap.add_argument("--qc_threshold_C", type=float, default=None,
                    help="QC threshold for category (C) feature (default: uses --qc_threshold)")
    ap.add_argument("--qc_threshold_R", type=float, default=0.6,
                    help="QC threshold for direction (R) feature (default: 0.6)")
    ap.add_argument("--qc_threshold_S", type=float, default=None,
                    help="QC threshold for saccade (S) feature (default: uses --qc_threshold)")
    ap.add_argument("--qc_threshold_T", type=float, default=None,
                    help="QC threshold for target (T) feature (default: uses --qc_threshold)")
    ap.add_argument("--tag", default="trialonset_comprehensive",
                    help="Output subfolder name")
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
        
        print(f"[align={align}] pt_min_ms={pt_min_ms}, axes_tag={axes_tag}")
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
                        qc_threshold=qc_threshold_feature
                    )
                    if result is not None:
                        all_results[key].append(result)
                        print(f"[{sid}] {area1}-{area2} {feature}: {result['n_good']} trials")
        
        # Create output directory
        out_dir = out_root / align / "trialtiming" / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        
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
                    
                    # Mark mean and median (with legend labels)
                    mean_t1 = np.nanmean(t1_all)
                    mean_t2 = np.nanmean(t2_all)
                    median_t1 = np.nanmedian(t1_all)
                    median_t2 = np.nanmedian(t2_all)
                    
                    ax.plot(mean_t1, mean_t2, "ro", ms=12, markerfacecolor="red", 
                            markeredgecolor="darkred", markeredgewidth=2, zorder=5, label="mean")
                    ax.plot(median_t1, median_t2, "bs", ms=12, markerfacecolor="blue", 
                            markeredgecolor="darkblue", markeredgewidth=2, zorder=5, label="median")
                    
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
                    
                    ax.legend(frameon=False, loc="lower right", fontsize=15)
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


if __name__ == "__main__":
    main()

