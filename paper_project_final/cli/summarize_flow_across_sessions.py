"""
Summarize flow results across sessions, per tag/config, per align, per feature,
and per canonical pair, separately for each monkey (M vs S via area prefixes).

Supports three alignments:
  - stim: aligned to stimulus onset (features: C, R)
  - sacc: aligned to saccade onset (features: S)
  - targ: aligned to target onset (features: T)

For each (align, tag, feature, pair A-B, monkey_label) it:

  - finds all sessions with both directions present (A->B and B->A)
  - applies SYMMETRIC QC filtering: if either area in a pair fails QC for the
    feature, that session is excluded for that pair (ensures same N for both directions)
  - loads bits, null means/SDs, p-values
  - computes per-session:
        bits_AtoB(t), bits_BtoA(t),
        z_AtoB(t), z_BtoA(t),
        diff_bits(t) = bits_AtoB(t) - bits_BtoA(t),
        sig masks (p<alpha),
        window-averaged excess bits & z in a task window
  - aggregates across sessions:
        mean ± SE for bits, z, diff over time,
        fraction of sessions significant at each time,
        window-level mean ± SE and fraction sig

Outputs per tag/align/feature/pair:

  out/<align>/summary/<tag>/<feature>/summary_<A>_vs_<B>.npz
  out/<align>/summary/<tag>/<feature>/figs/<A>_vs_<B>.pdf/.png

The .npz contains:
  - time (sec)
  - mean_bits_AtoB, se_bits_AtoB, mean_bits_BtoA, se_bits_BtoA
  - mean_z_AtoB,   se_z_AtoB,   mean_z_BtoA,   se_z_BtoA
  - frac_sig_AtoB, frac_sig_BtoA
  - mean_diff_bits, se_diff_bits
  - window-level stats (excess bits, z, diff) + fraction sig
  - session_ids (string array)
  - meta_json (JSON-encoded dict: tag, align, feature, pair, monkey_label, alpha, window, n_sessions)

Usage examples (from paper_project_final/):

  # summarize stim + sacc (default 'both')
  python cli/summarize_flow_across_sessions.py \
      --out_root out \
      --align both \
      --alpha 0.05

  # summarize all alignments including targ
  python cli/summarize_flow_across_sessions.py \
      --out_root out \
      --align all \
      --alpha 0.05

  # summarize targ-align only
  python cli/summarize_flow_across_sessions.py \
      --out_root out \
      --align targ \
      --alpha 0.05
"""

from __future__ import annotations
import argparse, os, json, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- helpers --------------------

def find_sessions(out_root: Path, align: str) -> List[str]:
    base = out_root / align
    if not base.exists():
        return []
    sids = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "caches").is_dir():
            sids.append(p.name)
    return sids


def discover_tags(out_root: Path, align: str, sids: List[str]) -> List[str]:
    tags = set()
    for sid in sids:
        flow_root = out_root / align / sid / "flow"
        if not flow_root.is_dir():
            continue
        for tag_dir in flow_root.iterdir():
            if tag_dir.is_dir():
                tags.add(tag_dir.name)
    return sorted(tags)


def discover_features(out_root: Path, align: str, tag: str, sids: List[str]) -> List[str]:
    """
    Discover available features for a given tag by checking what feature
    directories exist across sessions. Returns sorted list of feature names.
    """
    features = set()
    for sid in sids:
        tag_root = out_root / align / sid / "flow" / tag
        if not tag_root.is_dir():
            continue
        for feat_dir in tag_root.iterdir():
            if feat_dir.is_dir():
                features.add(feat_dir.name)
    return sorted(features)


def canonical_pairs(monkey_label: str) -> List[Tuple[str, str]]:
    """
    Canonical area pairs per monkey type.
    M: MFEF, MLIP, MSC
    S: SFEF, SLIP, SSC
    """
    if monkey_label.upper() == "M":
        return [("MFEF", "MLIP"), ("MFEF", "MSC"), ("MLIP", "MSC")]
    else:
        return [("SFEF", "SLIP"), ("SFEF", "SSC"), ("SLIP", "SSC")]


def parse_window(s: str) -> Tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def find_qc_tag_for_flow_tag(out_root: Path, align: str, sid: str, flow_tag: str) -> Optional[str]:
    """
    Try to find a QC tag that corresponds to a flow tag.
    Looks for QC directories under out/<align>/<sid>/qc/ and tries to match
    based on common patterns (e.g., flow tag might have extra suffixes).
    
    Flow tags often have format like: "crsweep-stim-vertical-none-trial"
    QC tags often have format like: "axes_sweep-stim-vertical"
    
    Returns the QC tag if found, None otherwise.
    """
    qc_base = out_root / align / sid / "qc"
    if not qc_base.exists():
        return None
    
    # Try exact match first
    if (qc_base / flow_tag).exists():
        return flow_tag
    
    # Try to find a QC tag that's a prefix of the flow tag
    # Flow tags often have suffixes like "-none-trial", "-zreg-trial", etc.
    # QC tags might be the base part
    flow_parts = flow_tag.split("-")
    for i in range(len(flow_parts), 0, -1):
        candidate = "-".join(flow_parts[:i])
        if (qc_base / candidate).exists():
            return candidate
    
    # Try converting flow tag pattern to QC tag pattern
    # e.g., "crsweep-stim-vertical-none-trial" -> "axes_sweep-stim-vertical"
    # Remove common flow suffixes: "-none-trial", "-zreg-trial", "-none-circ", etc.
    flow_clean = flow_tag
    for suffix in ["-none-trial", "-zreg-trial", "-none-circ", "-zreg-circ", 
                   "-none-phase", "-zreg-phase", "-trial", "-circ", "-phase"]:
        if flow_clean.endswith(suffix):
            flow_clean = flow_clean[:-len(suffix)]
            break
    
    # Try replacing "cr" prefix with "axes_" prefix
    if flow_clean.startswith("cr"):
        qc_candidate = "axes" + flow_clean[2:]  # "crsweep" -> "axessweep"
        if (qc_base / qc_candidate).exists():
            return qc_candidate
        # Also try with underscore: "axes_sweep"
        qc_candidate2 = "axes_" + flow_clean[2:]
        if (qc_base / qc_candidate2).exists():
            return qc_candidate2
    
    # Try matching by key components (align and orientation)
    # Expand flow_parts to handle abbreviations (vert->vertical, horiz->horizontal)
    flow_parts_expanded = set(flow_parts)
    if "vert" in flow_parts_expanded:
        flow_parts_expanded.add("vertical")
    if "horiz" in flow_parts_expanded:
        flow_parts_expanded.add("horizontal")
    
    best_match = None
    best_score = 0
    for qc_dir in qc_base.iterdir():
        if qc_dir.is_dir():
            qc_tag_name = qc_dir.name
            qc_parts_set = set(qc_tag_name.split("-"))
            # Also expand QC parts for matching
            qc_parts_expanded = set(qc_parts_set)
            if "vert" in qc_parts_expanded:
                qc_parts_expanded.add("vertical")
            if "horiz" in qc_parts_expanded:
                qc_parts_expanded.add("horizontal")
            
            # Count matching parts
            common = flow_parts_expanded & qc_parts_expanded
            # Prefer matches that include align and orientation
            score = len(common)
            if align in common:
                score += 2
            if any(ori in common for ori in ["vertical", "horizontal", "pooled"]):
                score += 2
            if score > best_score:
                best_score = score
                best_match = qc_tag_name
    
    # Only return if we have a reasonable match (at least 2 common parts)
    if best_score >= 2:
        return best_match
    
    return None


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
    C -> auc_C, R -> acc_R_macro, S -> auc_S_inv (prefer inv over raw), T -> auc_T
    """
    mapping = {
        "C": "auc_C",
        "R": "acc_R_macro",
        "S": "auc_S_inv",  # prefer inverse over raw
        "T": "auc_T",      # target configuration
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
    flow_tag: str,
    area: str,
    feature: str,
    qc_threshold: float,
    explicit_qc_tag: Optional[str] = None,
) -> bool:
    """
    Check if an area passes QC for a given feature.
    
    Parameters
    ----------
    explicit_qc_tag : str, optional
        If provided, use this QC tag directly instead of auto-detecting.
    
    Returns True if QC passes (or if QC data is unavailable), False if QC fails.
    """
    # Use explicit QC tag if provided, otherwise auto-detect
    if explicit_qc_tag is not None:
        qc_tag = explicit_qc_tag
    else:
        qc_tag = find_qc_tag_for_flow_tag(out_root, align, sid, flow_tag)
    
    if qc_tag is None:
        # If no QC tag found, we can't check - default to passing
        # (could also return False to be strict, but user might want to be lenient)
        return True
    
    # Load QC data
    qc_data = load_qc_json(out_root, align, sid, qc_tag, area)
    if qc_data is None:
        # No QC data available - default to passing
        return True
    
    # Get the appropriate metric for this feature
    metric = feature_to_qc_metric(feature)
    
    # Check if metric reaches threshold
    return check_qc_passes(qc_data, metric, qc_threshold)


def safe_z(bits: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.full_like(bits, np.nan, dtype=float)
    mask = np.isfinite(bits) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    z[mask] = (bits[mask] - mu[mask]) / sd[mask]
    return z


# -------------------- Panel D helpers: bits/trial/dim, df-corrected --------------------

LN2 = np.log(2.0)


def _get_flow_meta(Z: np.lib.npyio.NpzFile) -> dict:
    """Robustly parse Z['meta'] into a python dict."""
    m = Z["meta"]
    if isinstance(m, dict):
        return m
    if hasattr(m, "item"):
        m = m.item()
    if isinstance(m, dict):
        return m
    # fallback: json string (older style)
    return json.loads(m)


def _lag_bins_from_meta(meta: dict, time_s: np.ndarray) -> int:
    """Match flow.py: L = round(lags_ms / bin_ms), at least 1."""
    if time_s.size < 2:
        return 1
    bin_s = float(time_s[1] - time_s[0])
    lags_ms = float(meta["lags_ms"])
    return int(max(1, round(lags_ms / (bin_s * 1000.0))))


def bits_per_trial_dim_corr(bits: np.ndarray, meta: dict, time_s: np.ndarray) -> np.ndarray:
    """
    Compute bits/(trial*target_dim) minus Wilks complexity baseline.

    bits_ptd(t) = bits(t) / (N * K_target)
    baseline_ptd = (L * K_source) / (2 * N * ln 2)

    Returns array shaped like bits (T,).
    """
    N = float(meta.get("N", np.nan))
    K_src = float(meta.get("K_A", np.nan))
    K_tgt = float(meta.get("K_B", np.nan))
    if not (np.isfinite(N) and N > 0 and np.isfinite(K_tgt) and K_tgt > 0 and np.isfinite(K_src) and K_src > 0):
        return np.full_like(bits, np.nan, dtype=float)

    L = _lag_bins_from_meta(meta, time_s)

    with np.errstate(invalid="ignore", divide="ignore"):
        bits_ptd = bits / (N * K_tgt)

    baseline_ptd = (L * K_src) / (2.0 * N * LN2)
    return bits_ptd - baseline_ptd


def uniform_smooth_1d(arr: np.ndarray, win_bins: int) -> np.ndarray:
    """
    Apply uniform (boxcar) moving average along the last axis.
    Preserves shape; edges use partial windows (same-padding style via cumsum).
    
    Parameters
    ----------
    arr : (..., T) array
        Input array, smoothing applied along last axis.
    win_bins : int
        Window size in bins. Must be >= 1.
    
    Returns
    -------
    smoothed : (..., T) array
        Smoothed array, same shape as input.
    """
    if win_bins <= 1:
        return arr.copy()
    
    # Handle NaN by treating as 0 in sum, track counts
    orig_shape = arr.shape
    T = orig_shape[-1]
    arr_flat = arr.reshape(-1, T)  # (N, T)
    
    result = np.empty_like(arr_flat)
    for i in range(arr_flat.shape[0]):
        row = arr_flat[i]
        valid = np.isfinite(row)
        row_clean = np.where(valid, row, 0.0)
        
        # Cumsum trick for O(T) moving average
        cumsum = np.cumsum(row_clean)
        cumcount = np.cumsum(valid.astype(float))
        
        half = win_bins // 2
        smoothed = np.empty(T, dtype=float)
        for t in range(T):
            lo = max(0, t - half)
            hi = min(T - 1, t + half)
            
            if lo == 0:
                s = cumsum[hi]
                c = cumcount[hi]
            else:
                s = cumsum[hi] - cumsum[lo - 1]
                c = cumcount[hi] - cumcount[lo - 1]
            
            if c > 0:
                smoothed[t] = s / c
            else:
                smoothed[t] = np.nan
        
        result[i] = smoothed
    
    return result.reshape(orig_shape)


def group_null_p_for_mean_diff(
    mean_diff: np.ndarray,
    dnull_list: List[np.ndarray],
    B: int = 4096,
    seed: int = 12345,
    smooth_bins: int = 0,
) -> np.ndarray:
    """
    Old-style empirical group null for DIFF:
      - For each replicate b=1..B, sample one permutation row from each
        session's dnull and average across sessions.
      - p(t) = Pr{ group-null-mean(t) >= observed-mean-diff(t) } (one-sided).

    Parameters
    ----------
    mean_diff : (T,) array
        Observed across-session mean DIFF(t) = mean(bits_AtoB - bits_BtoA).
    dnull_list : list of (P, T) arrays
        Per-session null difference matrices: fnull - rnull.
    B : int
        Number of group-null replicates.
    seed : int
        RNG seed for reproducibility.
    smooth_bins : int
        If > 0, apply uniform moving average of this width (in bins) to both
        observed mean_diff and group null before computing p-values.

    Returns
    -------
    p : (T,) array
        Empirical p-value at each time bin.
    """
    rng = np.random.default_rng(seed)
    T = mean_diff.size
    S = len(dnull_list)
    if S == 0:
        return np.full(T, np.nan)

    acc = np.zeros((B, T), dtype=float)
    for dnull in dnull_list:
        P = dnull.shape[0]
        idx = rng.integers(0, P, size=B)
        acc += dnull[idx, :]  # (B, T)

    group_mean_null = acc / float(S)  # (B, T)

    # Apply smoothing if requested
    if smooth_bins > 1:
        mean_diff_smooth = uniform_smooth_1d(mean_diff[None, :], smooth_bins)[0]
        group_mean_null_smooth = uniform_smooth_1d(group_mean_null, smooth_bins)
    else:
        mean_diff_smooth = mean_diff
        group_mean_null_smooth = group_mean_null

    p = np.full(T, np.nan, dtype=float)
    for t in range(T):
        obs = mean_diff_smooth[t]
        if not np.isfinite(obs):
            continue
        col = group_mean_null_smooth[:, t]
        m = np.isfinite(col)
        nv = int(np.sum(m))
        if nv == 0:
            continue
        ge = int(np.sum(col[m] >= obs))
        p[t] = (1 + ge) / (1 + nv)
    return p


def nanmean_se(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (mean, se, n_valid) along axis, ignoring NaNs.
    Suppresses warnings for empty slices or insufficient degrees of freedom.
    """
    with warnings.catch_warnings():
        # Suppress numpy warnings about empty slices and insufficient degrees of freedom
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(arr, axis=axis)
            n = np.sum(np.isfinite(arr), axis=axis).astype(float)
            std = np.nanstd(arr, axis=axis, ddof=1)
            se = std / np.sqrt(n)
    return mean, se, n


def downsample_bins(
    time: np.ndarray,
    data_2d: np.ndarray,
    factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample time series by combining adjacent bins.
    
    Parameters
    ----------
    time : (T,) array
        Time points (seconds).
    data_2d : (N_sessions, T) array
        Data values for each session at each time bin.
    factor : int
        Number of adjacent bins to combine. If 1, returns original data.
    
    Returns
    -------
    new_time : (T_new,) array
        Downsampled time points (bin centers).
    new_data : (N_sessions, T_new) array
        Downsampled data (mean of combined bins).
    """
    if factor <= 1:
        return time, data_2d
    
    T = time.shape[0]
    T_new = T // factor
    
    if T_new == 0:
        return time, data_2d
    
    # Truncate to multiple of factor
    T_trunc = T_new * factor
    time_trunc = time[:T_trunc]
    data_trunc = data_2d[:, :T_trunc]
    
    # Reshape and average
    time_reshaped = time_trunc.reshape(T_new, factor)
    new_time = np.mean(time_reshaped, axis=1)
    
    N = data_trunc.shape[0]
    data_reshaped = data_trunc.reshape(N, T_new, factor)
    with np.errstate(invalid="ignore"):
        new_data = np.nanmean(data_reshaped, axis=2)
    
    return new_time, new_data


def rebin_timeseries(
    time: np.ndarray,
    data_arr: np.ndarray,
    win_size_s: float,
    step_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-bin time-series data into overlapping windows.

    Parameters
    ----------
    time : (T,) array
        Time points (seconds).
    data_arr : (N_sessions, T) array
        Data values for each session at each time bin.
    win_size_s : float
        Window size in seconds (e.g. 0.05 for 50 ms).
    step_s : float
        Step size in seconds between window centers (e.g. 0.02).

    Returns
    -------
    new_time : (T_win,) array
        Centers of the rebinned windows (seconds).
    rebinned_data : (N_sessions, T_win) array
        For each session, mean of data within each window.
    """
    if win_size_s is None or step_s is None:
        return time, data_arr

    T = time.shape[0]
    tmin, tmax = float(time[0]), float(time[-1])
    window_masks = []
    centers = []

    t = tmin
    while t + win_size_s <= tmax + 1e-12:
        a = t
        b = t + win_size_s
        mask = (time >= a) & (time < b)
        if np.any(mask):
            window_masks.append(mask)
            centers.append(0.5 * (a + b))
        t += step_s

    if not window_masks:
        return time, data_arr

    window_masks = np.stack(window_masks, axis=0)  # (T_win, T)
    centers = np.array(centers)

    rebinned = []
    for i in range(data_arr.shape[0]):  # over sessions
        row = data_arr[i]  # (T,)
        vals = np.array([np.nanmean(row[m]) for m in window_masks])
        rebinned.append(vals)
    rebinned = np.stack(rebinned, axis=0)  # (N_sessions, T_win)

    return centers, rebinned


def plot_panel_a_paper(
    out_path: Path,
    time: np.ndarray,
    mean_bits_AB: np.ndarray,
    se_bits_AB: np.ndarray,
    mean_bits_BA: np.ndarray,
    se_bits_BA: np.ndarray,
    label_A: str,
    label_B: str,
    t_min_ms: Optional[float] = None,
    t_max_ms: Optional[float] = None,
    xlabel: str = "Time (ms)",
) -> None:
    """
    Paper-quality Panel A figure: bits ± SE (A->B, B->A).
    - Uses real area names instead of "A -> B bits"
    - No grey time window
    - Font sizes matched to trial_onset_comprehensive.py
    - Plot area height matches the square scatter plot (5 inches)
    - Plot area width is 2x height (10 inches)
    
    Parameters
    ----------
    t_min_ms, t_max_ms : float, optional
        Time range limits in milliseconds. If provided, restricts x-axis.
    """
    t_ms = time * 1000.0
    
    # Define plot area dimensions (in inches) to match scatter plot height
    # Scatter plot: 7x7 figure with ~5x5 inch plot area after tight_layout
    # Panel plot: want 10x5 inch plot area (2:1 aspect, same height)
    plot_width_in = 10.0
    plot_height_in = 5.0
    # Add margins for labels (approximate)
    margin_left_in = 1.0
    margin_right_in = 0.5
    margin_bottom_in = 0.8
    margin_top_in = 0.5
    
    fig_width = plot_width_in + margin_left_in + margin_right_in
    fig_height = plot_height_in + margin_bottom_in + margin_top_in
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    # Position axes explicitly: [left, bottom, width, height] in figure fraction
    ax = fig.add_axes([
        margin_left_in / fig_width,
        margin_bottom_in / fig_height,
        plot_width_in / fig_width,
        plot_height_in / fig_height
    ])
    
    # Vertical line at t=0
    ax.axvline(0, ls="--", c="k", lw=0.8)
    
    # Plot A->B
    ax.plot(t_ms, mean_bits_AB, color="C0", lw=2.0, label=f"{label_A}→{label_B}")
    ax.fill_between(
        t_ms,
        mean_bits_AB - se_bits_AB,
        mean_bits_AB + se_bits_AB,
        color="C0",
        alpha=0.25,
        linewidth=0,
    )
    
    # Plot B->A
    ax.plot(t_ms, mean_bits_BA, color="C1", lw=2.0, label=f"{label_B}→{label_A}")
    ax.fill_between(
        t_ms,
        mean_bits_BA - se_bits_BA,
        mean_bits_BA + se_bits_BA,
        color="C1",
        alpha=0.25,
        linewidth=0,
    )
    
    # Labels with matched font sizes (from trial_onset_comprehensive.py)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("ΔLL (bits)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc="upper left", frameon=False, fontsize=15)
    
    # Set time range if specified
    if t_min_ms is not None and t_max_ms is not None:
        ax.set_xlim(t_min_ms, t_max_ms)
    
    # No tight_layout() - we use explicit axes positioning
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_panel_c_paper(
    out_path: Path,
    time: np.ndarray,
    mean_diff: np.ndarray,
    se_diff: np.ndarray,
    sig_group_diff: Optional[np.ndarray] = None,
    t_min_ms: Optional[float] = None,
    t_max_ms: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    xlabel: str = "Time (ms)",
) -> None:
    """
    Paper-quality Panel C figure: Net Flow (diff bits ± SE).
    - No grey time window
    - No frac sig secondary axis
    - Blue color instead of red
    - Label: "Net Flow"
    - Significance dots in black
    - Font sizes matched to trial_onset_comprehensive.py
    - Plot area height matches the square scatter plot (5 inches)
    - Plot area width is 2x height (10 inches)
    
    Parameters
    ----------
    t_min_ms, t_max_ms : float, optional
        Time range limits in milliseconds. If provided, restricts x-axis.
    y_min, y_max : float, optional
        Y-axis limits. If provided, restricts y-axis.
    """
    t_ms = time * 1000.0
    
    # Define plot area dimensions (in inches) to match scatter plot height
    # Scatter plot: 7x7 figure with ~5x5 inch plot area after tight_layout
    # Panel plot: want 10x5 inch plot area (2:1 aspect, same height)
    plot_width_in = 10.0
    plot_height_in = 5.0
    # Add margins for labels (approximate)
    margin_left_in = 1.0
    margin_right_in = 0.5
    margin_bottom_in = 0.8
    margin_top_in = 0.5
    
    fig_width = plot_width_in + margin_left_in + margin_right_in
    fig_height = plot_height_in + margin_bottom_in + margin_top_in
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    # Position axes explicitly: [left, bottom, width, height] in figure fraction
    ax = fig.add_axes([
        margin_left_in / fig_width,
        margin_bottom_in / fig_height,
        plot_width_in / fig_width,
        plot_height_in / fig_height
    ])
    
    # Vertical and horizontal reference lines
    ax.axvline(0, ls="--", c="k", lw=0.8)
    ax.axhline(0, ls=":", c="k", lw=0.8)
    
    # Plot Net Flow in indigo
    ax.plot(t_ms, mean_diff, color="darkcyan", lw=3, label="Net Flow")
    ax.fill_between(
        t_ms,
        mean_diff - se_diff,
        mean_diff + se_diff,
        color="darkcyan",
        alpha=0.25,
        linewidth=0,
    )
    
    # Set y-axis limits if specified (before plotting significance dots)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    
    # Plot significance dots in black (larger size)
    if sig_group_diff is not None and np.any(sig_group_diff):
        sig_mask = sig_group_diff.astype(bool)
        ylim = ax.get_ylim()
        y_marker = ylim[0] + 0.05 * (ylim[1] - ylim[0])  # 5% from bottom
        ax.scatter(
            t_ms[sig_mask], np.full(np.sum(sig_mask), y_marker),
            marker="o", s=14, c="black", zorder=5, label="p<α"
        )
    
    # Labels with matched font sizes (from trial_onset_comprehensive.py)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Net Flow (bits)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc="upper right", frameon=False, fontsize=20)
    
    # Set time range if specified
    if t_min_ms is not None and t_max_ms is not None:
        ax.set_xlim(t_min_ms, t_max_ms)
    
    # No tight_layout() - we use explicit axes positioning
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_panel_d_paper(
    out_path: Path,
    time: np.ndarray,
    mean_AB: np.ndarray,
    se_AB: np.ndarray,
    mean_BA: np.ndarray,
    se_BA: np.ndarray,
    label_A: str,
    label_B: str,
    sig_group_diff: Optional[np.ndarray] = None,
    t_min_ms: Optional[float] = None,
    t_max_ms: Optional[float] = None,
    xlabel: str = "Time (ms)",
) -> None:
    """
    Paper-quality Panel D: df-corrected bits / trial / dim for A→B and B→A.
    
    This shows:
    - bits(t) / (N * K_target)  minus  (L * K_source) / (2 * N * ln2)
    
    Where y=0 means "no more improvement than what's expected from model complexity alone."
    
    Optionally shows significance dots (same as Panel C) for net flow.
    """
    t_ms = time * 1000.0

    plot_width_in = 10.0
    plot_height_in = 5.0
    # Left margin for Panel D
    margin_left_in = 1.4
    margin_right_in = 0.5
    margin_bottom_in = 0.8
    margin_top_in = 0.5

    fig_width = plot_width_in + margin_left_in + margin_right_in
    fig_height = plot_height_in + margin_bottom_in + margin_top_in

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([
        margin_left_in / fig_width,
        margin_bottom_in / fig_height,
        plot_width_in / fig_width,
        plot_height_in / fig_height
    ])

    ax.axvline(0, ls="--", c="k", lw=0.8)
    ax.axhline(0, ls=":", c="k", lw=0.8)

    # No scaling (per trial)
    scale_factor = 1.0
    mean_AB_scaled = mean_AB * scale_factor
    se_AB_scaled = se_AB * scale_factor
    mean_BA_scaled = mean_BA * scale_factor
    se_BA_scaled = se_BA * scale_factor

    ax.plot(t_ms, mean_AB_scaled, color="#0072B2", lw=2.5, label=f"{label_A}→{label_B}")
    ax.fill_between(t_ms, mean_AB_scaled - se_AB_scaled, mean_AB_scaled + se_AB_scaled, color="#0072B2", alpha=0.25, linewidth=0)

    ax.plot(t_ms, mean_BA_scaled, color="#D55E00", lw=2.5, label=f"{label_B}→{label_A}")
    ax.fill_between(t_ms, mean_BA_scaled - se_BA_scaled, mean_BA_scaled + se_BA_scaled, color="#D55E00", alpha=0.25, linewidth=0)

    # Plot significance dots in black (same as Panel C)
    if sig_group_diff is not None and np.any(sig_group_diff):
        sig_mask = sig_group_diff.astype(bool)
        ylim = ax.get_ylim()
        y_marker = ylim[0] + 0.05 * (ylim[1] - ylim[0])  # 5% from bottom
        ax.scatter(
            t_ms[sig_mask], np.full(np.sum(sig_mask), y_marker),
            marker="o", s=14, c="black", zorder=5, label="p<α"
        )

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Directed predictability gain (bits)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc="upper left", frameon=False, fontsize=15)

    if t_min_ms is not None and t_max_ms is not None:
        ax.set_xlim(t_min_ms, t_max_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_panel_d_iv_paper(
    out_path: Path,
    time: np.ndarray,
    mean_AB: np.ndarray,
    se_AB: np.ndarray,
    mean_BA: np.ndarray,
    se_BA: np.ndarray,
    label_A: str,
    label_B: str,
    sig_group_diff: Optional[np.ndarray] = None,
    t_min_ms: Optional[float] = None,
    t_max_ms: Optional[float] = None,
    xlabel: str = "Time (ms)",
) -> None:
    """
    Paper-quality Panel D IV: Inverse-Variance Weighted df-corrected bits / trial / dim.
    
    Same as Panel D but uses inverse-variance weighting across sessions, giving
    more weight to sessions with more trials (higher precision).
    
    Weight_i = N_i (proportional to precision)
    mean_iv = Σ(w_i × x_i) / Σ(w_i)
    SE_iv = sqrt(Σ(w_i × (x_i - mean_iv)²) / ((n-1)/n × Σ(w_i)))
    """
    t_ms = time * 1000.0

    plot_width_in = 10.0
    plot_height_in = 5.0
    margin_left_in = 1.4
    margin_right_in = 0.5
    margin_bottom_in = 0.8
    margin_top_in = 0.5

    fig_width = plot_width_in + margin_left_in + margin_right_in
    fig_height = plot_height_in + margin_bottom_in + margin_top_in

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([
        margin_left_in / fig_width,
        margin_bottom_in / fig_height,
        plot_width_in / fig_width,
        plot_height_in / fig_height
    ])

    ax.axvline(0, ls="--", c="k", lw=0.8)
    ax.axhline(0, ls=":", c="k", lw=0.8)

    ax.plot(t_ms, mean_AB, color="#0072B2", lw=2.5, label=f"{label_A}→{label_B}")
    ax.fill_between(t_ms, mean_AB - se_AB, mean_AB + se_AB, color="#0072B2", alpha=0.25, linewidth=0)

    ax.plot(t_ms, mean_BA, color="#D55E00", lw=2.5, label=f"{label_B}→{label_A}")
    ax.fill_between(t_ms, mean_BA - se_BA, mean_BA + se_BA, color="#D55E00", alpha=0.25, linewidth=0)

    # Plot significance dots in black
    if sig_group_diff is not None and np.any(sig_group_diff):
        sig_mask = sig_group_diff.astype(bool)
        ylim = ax.get_ylim()
        y_marker = ylim[0] + 0.05 * (ylim[1] - ylim[0])
        ax.scatter(
            t_ms[sig_mask], np.full(np.sum(sig_mask), y_marker),
            marker="o", s=14, c="black", zorder=5, label="p<α"
        )

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Directed predictability gain (bits)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc="upper left", frameon=False, fontsize=15)

    if t_min_ms is not None and t_max_ms is not None:
        ax.set_xlim(t_min_ms, t_max_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def inverse_variance_weighted_mean_se(
    data_arr: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse-variance weighted mean and SE across sessions (axis=0).
    
    Parameters
    ----------
    data_arr : (N_sessions, T) array
        Per-session values at each time point.
    weights : (N_sessions,) array
        Weight for each session (e.g., trial count N_i).
    
    Returns
    -------
    mean_iv : (T,) array
        Weighted mean at each time point.
    se_iv : (T,) array
        Standard error of weighted mean.
    """
    N_sess, T = data_arr.shape
    weights = np.asarray(weights, dtype=float)
    
    mean_iv = np.full(T, np.nan)
    se_iv = np.full(T, np.nan)
    
    for t in range(T):
        col = data_arr[:, t]
        valid = np.isfinite(col)
        n_valid = int(np.sum(valid))
        
        if n_valid < 2:
            continue
        
        w = weights[valid]
        x = col[valid]
        
        # Weighted mean
        w_sum = np.sum(w)
        if w_sum <= 0:
            continue
        
        mean_t = np.sum(w * x) / w_sum
        mean_iv[t] = mean_t
        
        # Weighted SE (using reliability weights formula)
        # SE = sqrt(Σw_i(x_i - mean)² / ((n-1)/n × (Σw_i)²) × Σw_i)
        # Simplified: SE = sqrt(Σw_i(x_i - mean)² / (Σw_i)² × n/(n-1))
        residuals_sq = w * (x - mean_t) ** 2
        var_weighted = np.sum(residuals_sq) / w_sum
        
        # Bessel's correction for weighted variance
        # Using the "reliability weights" formula
        w_sum_sq = np.sum(w ** 2)
        if w_sum ** 2 > w_sum_sq:
            var_corrected = var_weighted * w_sum ** 2 / (w_sum ** 2 - w_sum_sq)
        else:
            var_corrected = var_weighted * n_valid / (n_valid - 1)
        
        # SE of weighted mean
        se_iv[t] = np.sqrt(var_corrected / n_valid)
    
    return mean_iv, se_iv


def plot_summary_figure(
    out_path_pdf: Path,
    time: np.ndarray,
    mean_bits_AB: np.ndarray,
    se_bits_AB: np.ndarray,
    mean_bits_BA: np.ndarray,
    se_bits_BA: np.ndarray,
    mean_z_AB: np.ndarray,
    se_z_AB: np.ndarray,
    mean_z_BA: np.ndarray,
    se_z_BA: np.ndarray,
    mean_diff: np.ndarray,
    se_diff: np.ndarray,
    frac_sig_AB: np.ndarray,
    win: Tuple[float, float],
    title: str,
    # NEW arguments for rebinned panel
    rebin_time_arr: Optional[np.ndarray] = None,
    mean_z_AB_rebin: Optional[np.ndarray] = None,
    se_z_AB_rebin: Optional[np.ndarray] = None,
    mean_z_BA_rebin: Optional[np.ndarray] = None,
    se_z_BA_rebin: Optional[np.ndarray] = None,
    # Group DIFF significance
    sig_group_diff: Optional[np.ndarray] = None,
    # Optional labels for reverse perspective
    label_A: str = "A",
    label_B: str = "B",
) -> None:
    """
    Make a summary figure with:
      - Panel 1: bits ± SE (A->B, B->A)
      - Panel 2: z ± SE
      - Panel 3: diff bits ± SE + frac_sig_AB(t)
      - Panel 4 (optional): rebinned z ± SE (if rebin parameters provided)
    """
    t_ms = time * 1000.0
    w_start_ms = win[0] * 1000.0
    w_end_ms = win[1] * 1000.0

    # Decide layout: 3 panels normally, 4 if rebin curves provided
    have_rebin = (
        rebin_time_arr is not None
        and mean_z_AB_rebin is not None
        and se_z_AB_rebin is not None
        and mean_z_BA_rebin is not None
        and se_z_BA_rebin is not None
    )

    n_panels = 4 if have_rebin else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(7.5, 8.5 + (n_panels - 3) * 2.0), sharex=True)
    if n_panels == 3:
        ax1, ax2, ax3 = axes
        ax4 = None
    else:
        ax1, ax2, ax3, ax4 = axes

    # Panel 1: bits
    ax1.axvline(0, ls="--", c="k", lw=0.8)
    ax1.axvspan(w_start_ms, w_end_ms, color="0.9", alpha=0.5, label="window")
    ax1.plot(t_ms, mean_bits_AB, color="C0", lw=2.0, label=f"{label_A}→{label_B} bits")
    ax1.fill_between(
        t_ms,
        mean_bits_AB - se_bits_AB,
        mean_bits_AB + se_bits_AB,
        color="C0",
        alpha=0.25,
        linewidth=0,
    )
    ax1.plot(t_ms, mean_bits_BA, color="C1", lw=2.0, label=f"{label_B}→{label_A} bits")
    ax1.fill_between(
        t_ms,
        mean_bits_BA - se_bits_BA,
        mean_bits_BA + se_bits_BA,
        color="C1",
        alpha=0.25,
        linewidth=0,
    )
    ax1.set_ylabel("ΔLL (bits)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", frameon=False)

    # Panel 2: z-scores
    ax2.axvline(0, ls="--", c="k", lw=0.8)
    ax2.axhline(0, ls=":", c="k", lw=0.8)
    ax2.axvspan(w_start_ms, w_end_ms, color="0.9", alpha=0.5)
    ax2.plot(t_ms, mean_z_AB, color="C0", lw=2.0, label=f"{label_A}→{label_B} z")
    ax2.fill_between(
        t_ms,
        mean_z_AB - se_z_AB,
        mean_z_AB + se_z_AB,
        color="C0",
        alpha=0.25,
        linewidth=0,
    )
    ax2.plot(t_ms, mean_z_BA, color="C1", lw=2.0, label=f"{label_B}→{label_A} z")
    ax2.fill_between(
        t_ms,
        mean_z_BA - se_z_BA,
        mean_z_BA + se_z_BA,
        color="C1",
        alpha=0.25,
        linewidth=0,
    )
    ax2.set_ylabel("Z (σ from null)")
    ax2.legend(loc="upper left", frameon=False)

    # Panel 3: diff + frac sig
    ax3.axvline(0, ls="--", c="k", lw=0.8)
    ax3.axhline(0, ls=":", c="k", lw=0.8)
    ax3.axvspan(w_start_ms, w_end_ms, color="0.9", alpha=0.5)
    ax3.plot(t_ms, mean_diff, color="C3", lw=2.0, label=f"{label_A}→{label_B} − {label_B}→{label_A}")
    ax3.fill_between(
        t_ms,
        mean_diff - se_diff,
        mean_diff + se_diff,
        color="C3",
        alpha=0.25,
        linewidth=0,
    )
    ax3.set_ylabel("ΔΔLL (bits)")

    ax3b = ax3.twinx()
    ax3b.plot(t_ms, frac_sig_AB, color="k", lw=1.5, ls="--", label=f"Frac sig {label_A}→{label_B}")
    ax3b.set_ylabel("Fraction p<α")
    ax3b.set_ylim(0, 1.0)

    # Plot significance dots for group DIFF p(t) at bottom of panel 3
    if sig_group_diff is not None and np.any(sig_group_diff):
        sig_mask = sig_group_diff.astype(bool)
        ylim = ax3.get_ylim()
        y_marker = ylim[0] + 0.05 * (ylim[1] - ylim[0])  # 5% from bottom
        ax3.scatter(
            t_ms[sig_mask], np.full(np.sum(sig_mask), y_marker),
            marker="o", s=12, c="C3", zorder=5, label="p<α (group)"
        )

    # Combined legend for panel 3
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3b.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    # Only set xlabel on ax3 if there's no 4th panel
    if ax4 is None:
        ax3.set_xlabel("Time (ms)")

    # Panel 4: rebinned z-scores (if available)
    if ax4 is not None:
        t_win_ms = rebin_time_arr * 1000.0

        ax4.axvline(0, ls="--", c="k", lw=0.8)
        ax4.axhline(0, ls=":", c="k", lw=0.8)
        ax4.axvspan(w_start_ms, w_end_ms, color="0.9", alpha=0.5)

        # A->B rebinned z
        ax4.plot(t_win_ms, mean_z_AB_rebin, color="C0", lw=2.0, label=f"{label_A}→{label_B} z (rebinned)")
        ax4.fill_between(
            t_win_ms,
            mean_z_AB_rebin - se_z_AB_rebin,
            mean_z_AB_rebin + se_z_AB_rebin,
            color="C0",
            alpha=0.25,
            linewidth=0,
        )

        # B->A rebinned z
        ax4.plot(t_win_ms, mean_z_BA_rebin, color="C1", lw=2.0, label=f"{label_B}→{label_A} z (rebinned)")
        ax4.fill_between(
            t_win_ms,
            mean_z_BA_rebin - se_z_BA_rebin,
            mean_z_BA_rebin + se_z_BA_rebin,
            color="C1",
            alpha=0.25,
            linewidth=0,
        )

        ax4.set_ylabel("Z (rebinned)")
        ax4.set_xlabel("Time (ms)")
        ax4.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    out_dir = out_path_pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


# -------------------- main summarizer --------------------

def summarize_for_tag_align_feature(
    out_root: Path,
    align: str,
    tag: str,
    feature: str,
    sids: List[str],
    alpha: float,
    win: Tuple[float, float],
    rebin_win_s: Optional[float] = None,
    rebin_step_s: Optional[float] = None,
    qc_threshold: Optional[float] = None,
    qc_tag: Optional[str] = None,
    group_diff_p: bool = False,
    group_null_B: int = 4096,
    group_null_seed: int = 12345,
    smooth_ms: float = 0.0,
    bin_combine: int = 1,
) -> None:
    """
    Summarize flows across sessions for one (align, tag, feature).
    Writes one .npz + .pdf/.png per canonical pair (A,B).
    
    Parameters
    ----------
    rebin_win_s : float, optional
        Window size in seconds for rebinned z panel (e.g. 0.05 for 50 ms).
    rebin_step_s : float, optional
        Step size in seconds between rebinned windows (e.g. 0.02 for 20 ms).
    qc_threshold : float, optional
        QC threshold (e.g., 0.75). If None, QC filtering is disabled.
        If provided, SYMMETRIC rejection is applied: for a pair (A, B), if EITHER
        area fails QC for this feature, the entire session is excluded for that pair.
        This ensures both directions (A->B and B->A) have the same session count.
    group_diff_p : bool
        If True, compute old-style group empirical p(t) for DIFF using saved
        null samples. Requires flow_*.npz files to contain null_samps_AtoB.
    group_null_B : int
        Number of group-null replicates for DIFF p(t) (default: 4096).
    group_null_seed : int
        RNG seed for group-null sampling (default: 12345).
    smooth_ms : float
        Smoothing window in milliseconds. If > 0, applies uniform moving average
        to both observed DIFF and group null before computing p-values. Keeps
        original time resolution. Default 0 (no smoothing).
    bin_combine : int
        Number of adjacent time bins to combine (default: 1 = no combining).
        Set to 2 for sacc alignment to match stim's 10ms bins from sacc's 5ms bins.
    """
    for monkey_label in ("M", "S"):
        pairs = canonical_pairs(monkey_label)
        for (A, B) in pairs:
            all_bits_AB = []
            all_bits_BA = []
            all_z_AB = []
            all_z_BA = []
            all_diff = []
            all_sig_AB = []
            all_sig_BA = []
            # Panel D: per-session normalized bits (normalized FIRST, then averaged)
            all_bits_ptd_AB = []
            all_bits_ptd_BA = []
            win_excess_AB = []
            win_excess_BA = []
            win_z_AB = []
            win_z_BA = []
            win_diff = []
            win_sig_AB = []

            # Panel D: collect per-session meta for logging
            Ns = []
            KAs = []
            KBs = []
            lags_ms_list = []
            bin_ms_raw = None

            # For old-style group DIFF p(t)
            dnull_list = []

            session_ids = []
            time = None

            for sid in sids:
                base = out_root / align / sid / "flow" / tag / feature
                p_fwd = base / f"flow_{feature}_{A}to{B}.npz"
                p_rev = base / f"flow_{feature}_{B}to{A}.npz"
                if not (p_fwd.is_file() and p_rev.is_file()):
                    continue

                # QC filtering: check if areas pass QC threshold
                # SYMMETRIC rejection: if EITHER area fails QC, skip the entire session
                # for this pair. This ensures both directions have the same session count.
                # Skip QC filtering if threshold is None, 0, or negative
                if qc_threshold is not None and qc_threshold > 0:
                    qc_pass_A = check_area_qc(out_root, align, sid, tag, A, feature, qc_threshold, qc_tag)
                    qc_pass_B = check_area_qc(out_root, align, sid, tag, B, feature, qc_threshold, qc_tag)
                    
                    # Skip session if EITHER area fails QC (symmetric rejection)
                    if not qc_pass_A or not qc_pass_B:
                        failed_areas = []
                        if not qc_pass_A:
                            failed_areas.append(A)
                        if not qc_pass_B:
                            failed_areas.append(B)
                        print(f"[qc-filter] {sid}: {', '.join(failed_areas)} fails QC "
                              f"(threshold={qc_threshold}), skipping {A}-{B} pair")
                        continue

                Zf = np.load(p_fwd, allow_pickle=True)
                Zr = np.load(p_rev, allow_pickle=True)

                t = np.asarray(Zf["time"], dtype=float)  # seconds
                if time is None:
                    time = t
                else:
                    if time.shape != t.shape or not np.allclose(time, t):
                        raise ValueError(
                            f"Inconsistent time grid for {align}, tag={tag}, feature={feature}, "
                            f"pair {A}->{B}, sid={sid}"
                        )

                # Collect null samples for old-style group DIFF p(t)
                if group_diff_p:
                    if "null_samps_AtoB" not in Zf.files or "null_samps_AtoB" not in Zr.files:
                        # Skip this session for group_diff_p; it was run without --save_null_samples
                        pass  # dnull_list will be shorter; handled gracefully below
                    else:
                        fnull = np.asarray(Zf["null_samps_AtoB"], dtype=float)  # (P, T) from A->B file
                        rnull = np.asarray(Zr["null_samps_AtoB"], dtype=float)  # (P, T) from B->A file
                        if fnull.shape != rnull.shape:
                            raise ValueError(f"Null sample shape mismatch in {sid} for pair {A}-{B}")
                        dnull_list.append(fnull - rnull)  # (P, T)

                bits_AB = np.asarray(Zf["bits_AtoB"], dtype=float)
                bits_BA = np.asarray(Zr["bits_AtoB"], dtype=float)
                mu_AB = np.asarray(Zf["null_mean_AtoB"], dtype=float)
                sd_AB = np.asarray(Zf["null_std_AtoB"], dtype=float)
                mu_BA = np.asarray(Zr["null_mean_AtoB"], dtype=float)
                # BUGFIX: In the reverse file flow_*_{B}to{A}.npz, the B→A direction is stored as "*_AtoB".
                sd_BA = np.asarray(Zr["null_std_AtoB"], dtype=float)
                p_AB = np.asarray(Zf["p_AtoB"], dtype=float)
                p_BA = np.asarray(Zr["p_AtoB"], dtype=float)

                # Panel D: collect per-session meta for per-session normalization
                meta_f = _get_flow_meta(Zf)
                meta_r = _get_flow_meta(Zr)
                N_sess = int(meta_f.get("N", 0))
                K_A_sess = int(meta_f.get("K_A", 1))
                K_B_sess = int(meta_f.get("K_B", 1))
                lags_ms_sess = float(meta_f.get("lags_ms", 0))
                
                Ns.append(N_sess)
                KAs.append(K_A_sess)
                KBs.append(K_B_sess)
                lags_ms_list.append(lags_ms_sess)
                if bin_ms_raw is None and t.size > 1:
                    bin_ms_raw = float((t[1] - t[0]) * 1000.0)

                # At this point, both areas have passed QC (symmetric rejection above)
                z_AB = safe_z(bits_AB, mu_AB, sd_AB)
                z_BA = safe_z(bits_BA, mu_BA, sd_BA)
                diff_bits = bits_AB - bits_BA
                
                # === Panel D: Compute NORMALIZED bits per-session FIRST ===
                # This is the correct order: normalize each session, then average
                # bits_ptd = bits / (N * K_target) - baseline_ptd
                # where baseline_ptd = (L * K_source) / (2 * N * ln2)
                if N_sess > 0 and K_A_sess > 0 and K_B_sess > 0:
                    bin_ms_sess = float((t[1] - t[0]) * 1000.0) if t.size > 1 else 10.0
                    L_sess = max(1, int(round(lags_ms_sess / bin_ms_sess)))
                    
                    # A→B: predicts B, so target_dim = K_B, source_dim = K_A
                    baseline_AB = (L_sess * K_A_sess) / (2.0 * N_sess * LN2)
                    bits_ptd_AB = bits_AB / (N_sess * K_B_sess) - baseline_AB
                    
                    # B→A: predicts A, so target_dim = K_A, source_dim = K_B
                    baseline_BA = (L_sess * K_B_sess) / (2.0 * N_sess * LN2)
                    bits_ptd_BA = bits_BA / (N_sess * K_A_sess) - baseline_BA
                else:
                    bits_ptd_AB = np.full_like(bits_AB, np.nan)
                    bits_ptd_BA = np.full_like(bits_BA, np.nan)

                sig_AB = (p_AB < alpha) & np.isfinite(p_AB)
                sig_BA = (p_BA < alpha) & np.isfinite(p_BA)

                # window mask in seconds
                ws, we = win
                wmask = (time >= ws) & (time <= we)
                if not np.any(wmask):
                    w_excess_AB = np.nan
                    w_excess_BA = np.nan
                    w_z_AB = np.nan
                    w_z_BA = np.nan
                    w_diff = np.nan
                    w_sig = np.nan
                else:
                    excess_AB = bits_AB - mu_AB
                    excess_BA = bits_BA - mu_BA
                    w_excess_AB = float(np.nanmean(excess_AB[wmask]))
                    w_excess_BA = float(np.nanmean(excess_BA[wmask]))
                    w_z_AB = float(np.nanmean(z_AB[wmask]))
                    w_z_BA = float(np.nanmean(z_BA[wmask]))
                    w_diff = float(np.nanmean(diff_bits[wmask]))
                    w_sig = float(np.any(sig_AB[wmask]))

                all_bits_AB.append(bits_AB)
                all_bits_BA.append(bits_BA)
                all_z_AB.append(z_AB)
                all_z_BA.append(z_BA)
                all_diff.append(diff_bits)
                all_sig_AB.append(sig_AB.astype(float))
                all_sig_BA.append(sig_BA.astype(float))
                # Panel D: collect per-session normalized bits
                all_bits_ptd_AB.append(bits_ptd_AB)
                all_bits_ptd_BA.append(bits_ptd_BA)
                win_excess_AB.append(w_excess_AB)
                win_excess_BA.append(w_excess_BA)
                win_z_AB.append(w_z_AB)
                win_z_BA.append(w_z_BA)
                win_diff.append(w_diff)
                win_sig_AB.append(w_sig)
                session_ids.append(sid)

            if not all_bits_AB:
                # no sessions had this pair for this tag/feature/monkey
                continue

            # Compute smooth_bins from smooth_ms and time array
            smooth_bins = 0
            if smooth_ms > 0 and time is not None and len(time) > 1:
                bin_s = float(time[1] - time[0])  # bin size in seconds
                bin_ms = bin_s * 1000.0
                smooth_bins = max(1, int(round(smooth_ms / bin_ms)))
                if smooth_bins > 1:
                    print(f"  [smooth] {smooth_ms:.1f}ms -> {smooth_bins} bins "
                          f"(bin_size={bin_ms:.1f}ms)")

            # stack into (N_sessions, T)
            bits_AB_arr = np.vstack(all_bits_AB)
            bits_BA_arr = np.vstack(all_bits_BA)
            z_AB_arr = np.vstack(all_z_AB)
            z_BA_arr = np.vstack(all_z_BA)
            diff_arr = np.vstack(all_diff)
            sig_AB_arr = np.vstack(all_sig_AB)
            sig_BA_arr = np.vstack(all_sig_BA)

            # Stack Panel D per-session normalized arrays
            bits_ptd_AB_arr = np.vstack(all_bits_ptd_AB)
            bits_ptd_BA_arr = np.vstack(all_bits_ptd_BA)

            # Downsample by combining adjacent bins (e.g., for sacc: 5ms -> 10ms)
            if bin_combine > 1:
                orig_time = time.copy()
                orig_T = time.shape[0]
                time, bits_AB_arr = downsample_bins(orig_time, bits_AB_arr, bin_combine)
                _, bits_BA_arr = downsample_bins(orig_time, bits_BA_arr, bin_combine)
                _, z_AB_arr = downsample_bins(orig_time, z_AB_arr, bin_combine)
                _, z_BA_arr = downsample_bins(orig_time, z_BA_arr, bin_combine)
                _, diff_arr = downsample_bins(orig_time, diff_arr, bin_combine)
                _, sig_AB_arr = downsample_bins(orig_time, sig_AB_arr, bin_combine)
                _, sig_BA_arr = downsample_bins(orig_time, sig_BA_arr, bin_combine)
                # Also downsample Panel D normalized arrays
                _, bits_ptd_AB_arr = downsample_bins(orig_time, bits_ptd_AB_arr, bin_combine)
                _, bits_ptd_BA_arr = downsample_bins(orig_time, bits_ptd_BA_arr, bin_combine)
                # Also downsample dnull_list for group p-value computation
                dnull_list = [downsample_bins(orig_time, dnull, bin_combine)[1] for dnull in dnull_list]
                # Update bin_ms_raw for smoothing computation after downsampling
                bin_ms_raw = bin_ms_raw * bin_combine
                new_T = time.shape[0]
                print(f"  [bin_combine={bin_combine}] {orig_T} bins -> {new_T} bins")

            # per-time summaries
            mean_bits_AB, se_bits_AB, n_AB = nanmean_se(bits_AB_arr, axis=0)
            mean_bits_BA, se_bits_BA, n_BA = nanmean_se(bits_BA_arr, axis=0)
            mean_z_AB,   se_z_AB,   _     = nanmean_se(z_AB_arr,   axis=0)
            mean_z_BA,   se_z_BA,   _     = nanmean_se(z_BA_arr,   axis=0)
            mean_diff,   se_diff,   _     = nanmean_se(diff_arr,   axis=0)

            # ---- Panel D: CORRECT approach - average per-session normalized values ----
            # Each session was already normalized by its own N, K_A, K_B, L before averaging.
            # This is the statistically correct order: normalize first, then average.
            mean_bits_ptd_corr_AB, se_bits_ptd_corr_AB, _ = nanmean_se(bits_ptd_AB_arr, axis=0)
            mean_bits_ptd_corr_BA, se_bits_ptd_corr_BA, _ = nanmean_se(bits_ptd_BA_arr, axis=0)

            # Reference values for logging (median across sessions)
            N_ref = float(np.nanmedian(np.array(Ns, dtype=float)))
            K_A_ref = int(np.nanmedian(np.array(KAs, dtype=float)))
            K_B_ref = int(np.nanmedian(np.array(KBs, dtype=float)))
            lags_ms_ref = float(np.nanmedian(np.array(lags_ms_list, dtype=float)))

            if bin_ms_raw is None or not np.isfinite(bin_ms_raw) or bin_ms_raw <= 0:
                # Fallback: estimate from time array
                if time is not None and len(time) > 1:
                    bin_ms_raw = float((time[1] - time[0]) * 1000.0)
                else:
                    bin_ms_raw = 10.0  # default fallback

            L_ref = max(1, int(round(lags_ms_ref / bin_ms_raw)))
            df_ref = L_ref * K_A_ref * K_B_ref
            baseline_bits_ref = df_ref / (2.0 * LN2)

            # Apply smoothing to Panel D curves (uses same --smooth_ms as group DIFF p-value)
            panel_d_smooth_ms = smooth_ms
            panel_d_smooth_bins = max(1, int(round(panel_d_smooth_ms / bin_ms_raw))) if panel_d_smooth_ms > 0 else 0
            if panel_d_smooth_bins > 1:
                mean_bits_ptd_corr_AB = uniform_smooth_1d(mean_bits_ptd_corr_AB[None, :], panel_d_smooth_bins)[0]
                se_bits_ptd_corr_AB   = uniform_smooth_1d(se_bits_ptd_corr_AB[None, :], panel_d_smooth_bins)[0]
                mean_bits_ptd_corr_BA = uniform_smooth_1d(mean_bits_ptd_corr_BA[None, :], panel_d_smooth_bins)[0]
                se_bits_ptd_corr_BA   = uniform_smooth_1d(se_bits_ptd_corr_BA[None, :], panel_d_smooth_bins)[0]

            print(f"  [Panel D] Per-session normalized, then averaged. Median stats: "
                  f"N={N_ref:.0f}, K_A={K_A_ref}, K_B={K_B_ref}, L={L_ref}, "
                  f"smooth={panel_d_smooth_bins} bins ({panel_d_smooth_ms}ms)")

            # ---- Panel D IV: Inverse-Variance Weighted ----
            # Weight each session by its trial count (proportional to precision)
            N_weights = np.array(Ns, dtype=float)
            mean_bits_ptd_iv_AB, se_bits_ptd_iv_AB = inverse_variance_weighted_mean_se(
                bits_ptd_AB_arr, N_weights
            )
            mean_bits_ptd_iv_BA, se_bits_ptd_iv_BA = inverse_variance_weighted_mean_se(
                bits_ptd_BA_arr, N_weights
            )
            
            # Apply same smoothing to IV-weighted curves (uses --smooth_ms)
            if panel_d_smooth_bins > 1:
                mean_bits_ptd_iv_AB = uniform_smooth_1d(mean_bits_ptd_iv_AB[None, :], panel_d_smooth_bins)[0]
                se_bits_ptd_iv_AB   = uniform_smooth_1d(se_bits_ptd_iv_AB[None, :], panel_d_smooth_bins)[0]
                mean_bits_ptd_iv_BA = uniform_smooth_1d(mean_bits_ptd_iv_BA[None, :], panel_d_smooth_bins)[0]
                se_bits_ptd_iv_BA   = uniform_smooth_1d(se_bits_ptd_iv_BA[None, :], panel_d_smooth_bins)[0]
            
            print(f"  [Panel D IV] Inverse-variance weighted (by N). "
                  f"N range: {int(np.min(N_weights))}-{int(np.max(N_weights))}, "
                  f"sum(N)={int(np.sum(N_weights))}, smooth={panel_d_smooth_bins} bins ({panel_d_smooth_ms}ms)")

            # Compute old-style group empirical p(t) for DIFF if requested
            p_group_diff = None
            sig_group_diff = None
            if group_diff_p:
                if len(dnull_list) == 0:
                    print(f"  [warn] --group_diff_p requested but no sessions have null_samps_AtoB "
                          f"for {tag}/{feature}/{A}-{B}. Skipping group p(t).")
                elif len(dnull_list) < len(session_ids):
                    print(f"  [warn] Only {len(dnull_list)}/{len(session_ids)} sessions have null samples "
                          f"for {tag}/{feature}/{A}-{B}. Group p(t) uses subset.")
            if group_diff_p and len(dnull_list) > 0:
                p_group_diff = group_null_p_for_mean_diff(
                    mean_diff=mean_diff,
                    dnull_list=dnull_list,
                    B=group_null_B,
                    seed=group_null_seed,
                    smooth_bins=smooth_bins,
                )
                sig_group_diff = (p_group_diff < alpha) & np.isfinite(p_group_diff)

            # Optional time rebinning for a 4th panel
            rebin_time_arr = None
            mean_z_AB_rebin = None
            se_z_AB_rebin = None
            mean_z_BA_rebin = None
            se_z_BA_rebin = None

            if rebin_win_s is not None and rebin_step_s is not None:
                # rebin z across time using sliding window
                rebin_time_arr, z_AB_reb = rebin_timeseries(time, z_AB_arr, rebin_win_s, rebin_step_s)
                _,              z_BA_reb = rebin_timeseries(time, z_BA_arr, rebin_win_s, rebin_step_s)

                mean_z_AB_rebin, se_z_AB_rebin, _ = nanmean_se(z_AB_reb, axis=0)
                mean_z_BA_rebin, se_z_BA_rebin, _ = nanmean_se(z_BA_reb, axis=0)

            # fraction of sessions sig at each time
            with np.errstate(invalid="ignore", divide="ignore"):
                frac_sig_AB = np.nanmean(sig_AB_arr, axis=0)
                frac_sig_BA = np.nanmean(sig_BA_arr, axis=0)

            # window-level summaries
            win_excess_AB_arr = np.array(win_excess_AB, dtype=float)
            win_excess_BA_arr = np.array(win_excess_BA, dtype=float)
            win_z_AB_arr = np.array(win_z_AB, dtype=float)
            win_z_BA_arr = np.array(win_z_BA, dtype=float)
            win_diff_arr = np.array(win_diff, dtype=float)
            win_sig_AB_arr = np.array(win_sig_AB, dtype=float)

            w_mean_excess_AB, w_se_excess_AB, w_n = nanmean_se(win_excess_AB_arr, axis=0)
            w_mean_excess_BA, w_se_excess_BA, _   = nanmean_se(win_excess_BA_arr, axis=0)
            w_mean_z_AB,      w_se_z_AB,      _   = nanmean_se(win_z_AB_arr,      axis=0)
            w_mean_z_BA,      w_se_z_BA,      _   = nanmean_se(win_z_BA_arr,      axis=0)
            w_mean_diff,      w_se_diff,      _   = nanmean_se(win_diff_arr,      axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                w_frac_sig_AB = float(np.nanmean(win_sig_AB_arr))

            n_sessions = int(len(session_ids))
            print(f"[summary] align={align}, tag={tag}, feature={feature}, "
                  f"pair={A}-{B}, monkey={monkey_label}, N={n_sessions}")

            # output dir
            summary_dir = out_root / align / "summary" / tag / feature
            summary_dir.mkdir(parents=True, exist_ok=True)
            figs_dir = summary_dir / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{A}_vs_{B}"

            # Save npz
            meta = dict(
                tag=tag,
                align=align,
                feature=feature,
                pair=f"{A}-{B}",
                monkey_label=monkey_label,
                alpha=float(alpha),
                win_start_s=float(win[0]),
                win_end_s=float(win[1]),
                n_sessions=n_sessions,
            )
            if qc_threshold is not None and qc_threshold > 0:
                meta["qc_threshold"] = float(qc_threshold)
            meta_json = json.dumps(meta)

            out_path_npz = summary_dir / f"summary_{pair_name}.npz"
            np.savez_compressed(
                out_path_npz,
                time=time,
                mean_bits_AtoB=mean_bits_AB,
                se_bits_AtoB=se_bits_AB,
                mean_bits_BtoA=mean_bits_BA,
                se_bits_BtoA=se_bits_BA,
                mean_z_AtoB=mean_z_AB,
                se_z_AtoB=se_z_AB,
                mean_z_BtoA=mean_z_BA,
                se_z_BtoA=se_z_BA,
                frac_sig_AtoB=frac_sig_AB,
                frac_sig_BtoA=frac_sig_BA,
                mean_diff_bits=mean_diff,
                se_diff_bits=se_diff,
                p_group_diff=p_group_diff if p_group_diff is not None else np.array([]),
                sig_group_diff=sig_group_diff.astype(int) if sig_group_diff is not None else np.array([], dtype=int),
                # Panel D (shape-preserving): df-corrected bits / trial / dim
                mean_bits_ptd_corr_AtoB=mean_bits_ptd_corr_AB,
                se_bits_ptd_corr_AtoB=se_bits_ptd_corr_AB,
                mean_bits_ptd_corr_BtoA=mean_bits_ptd_corr_BA,
                se_bits_ptd_corr_BtoA=se_bits_ptd_corr_BA,
                # Panel D IV: Inverse-Variance Weighted
                mean_bits_ptd_iv_AtoB=mean_bits_ptd_iv_AB,
                se_bits_ptd_iv_AtoB=se_bits_ptd_iv_AB,
                mean_bits_ptd_iv_BtoA=mean_bits_ptd_iv_BA,
                se_bits_ptd_iv_BtoA=se_bits_ptd_iv_BA,
                # Panel D reference constants (for interpretation)
                panel_d_mode=np.array("shape_preserving"),
                panel_d_N_ref=np.array(N_ref),
                panel_d_L_ref=np.array(L_ref),
                panel_d_K_A_ref=np.array(K_A_ref),
                panel_d_K_B_ref=np.array(K_B_ref),
                panel_d_df_ref=np.array(df_ref),
                panel_d_baseline_bits=np.array(baseline_bits_ref),
                panel_d_iv_N_weights=N_weights,
                win_mean_excess_bits_AtoB=w_mean_excess_AB,
                win_se_excess_bits_AtoB=w_se_excess_AB,
                win_mean_excess_bits_BtoA=w_mean_excess_BA,
                win_se_excess_bits_BtoA=w_se_excess_BA,
                win_mean_z_AtoB=w_mean_z_AB,
                win_se_z_AtoB=w_se_z_AB,
                win_mean_z_BtoA=w_mean_z_BA,
                win_se_z_BtoA=w_se_z_BA,
                win_mean_diff_bits=w_mean_diff,
                win_se_diff_bits=w_se_diff,
                win_frac_sig_AtoB=w_frac_sig_AB,
                session_ids=np.array(session_ids, dtype="U"),
                meta_json=np.array(meta_json),
            )

            # Save figure for A vs B
            # Split title into two lines for better readability
            title = (f"{align.upper()} | {tag} | {feature}\n"
                     f"{A} vs {B} | monkey={monkey_label} | N={n_sessions}")
            fig_path_pdf = figs_dir / f"{pair_name}.pdf"
            plot_summary_figure(
                out_path_pdf=fig_path_pdf,
                time=time,
                mean_bits_AB=mean_bits_AB,
                se_bits_AB=se_bits_AB,
                mean_bits_BA=mean_bits_BA,
                se_bits_BA=se_bits_BA,
                mean_z_AB=mean_z_AB,
                se_z_AB=se_z_AB,
                mean_z_BA=mean_z_BA,
                se_z_BA=se_z_BA,
                mean_diff=mean_diff,
                se_diff=se_diff,
                frac_sig_AB=frac_sig_AB,
                win=win,
                title=title,
                # NEW rebinned z for panel 4
                rebin_time_arr=rebin_time_arr,
                mean_z_AB_rebin=mean_z_AB_rebin,
                se_z_AB_rebin=se_z_AB_rebin,
                mean_z_BA_rebin=mean_z_BA_rebin,
                se_z_BA_rebin=se_z_BA_rebin,
                # Group DIFF significance
                sig_group_diff=sig_group_diff,
            )
            
            # Determine time range for paper-quality figures based on feature
            # Category (C), Direction (R): -100 to 500 ms (stim-aligned)
            # Saccade (S): -290 to 200 ms (sacc-aligned)
            # Target (T): no change (use full time range)
            if feature in ("C", "R"):
                paper_t_min_ms, paper_t_max_ms = -100.0, 500.0
            elif feature == "S":
                paper_t_min_ms, paper_t_max_ms = -290.0, 200.0
            else:
                paper_t_min_ms, paper_t_max_ms = None, None
            
            # Determine x-axis label based on alignment
            if align == "stim":
                paper_xlabel = "Time from Stimulus Onset (ms)"
            elif align == "sacc":
                paper_xlabel = "Time from Saccade Onset (ms)"
            elif align == "targ":
                paper_xlabel = "Time from Target Onset (ms)"
            else:
                paper_xlabel = "Time (ms)"
            
            # Y-axis limits for panel C: different for monkey M vs S, and stim vs sacc
            # Stim (C, R): Monkey M: -10 to 20, Monkey S: -8 to 13
            # Sacc (S): Monkey M: -10 to 25, Monkey S: -8 to 13
            if feature in ("C", "R"):
                if monkey_label.upper() == "M":
                    paper_y_min, paper_y_max = -10.0, 20.0
                else:  # monkey S
                    paper_y_min, paper_y_max = -8.0, 13.0
            elif feature == "S":
                if monkey_label.upper() == "M":
                    paper_y_min, paper_y_max = -10.0, 39.0
                else:  # monkey S
                    paper_y_min, paper_y_max = -8.0, 13.0
            else:
                paper_y_min, paper_y_max = None, None
            
            # Save paper-quality Panel A figure separately
            fig_path_panel_a = figs_dir / f"{pair_name}_panel_a.pdf"
            plot_panel_a_paper(
                out_path=fig_path_panel_a,
                time=time,
                mean_bits_AB=mean_bits_AB,
                se_bits_AB=se_bits_AB,
                mean_bits_BA=mean_bits_BA,
                se_bits_BA=se_bits_BA,
                label_A=A,
                label_B=B,
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                xlabel=paper_xlabel,
            )
            
            # Save paper-quality Panel C figure separately
            fig_path_panel_c = figs_dir / f"{pair_name}_panel_c.pdf"
            plot_panel_c_paper(
                out_path=fig_path_panel_c,
                time=time,
                mean_diff=mean_diff,
                se_diff=se_diff,
                sig_group_diff=sig_group_diff,
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                y_min=paper_y_min,
                y_max=paper_y_max,
                xlabel=paper_xlabel,
            )
            
            # Save paper-quality Panel D figure separately (df-corrected bits / trial / dim)
            fig_path_panel_d = figs_dir / f"{pair_name}_panel_d.pdf"
            plot_panel_d_paper(
                out_path=fig_path_panel_d,
                time=time,
                mean_AB=mean_bits_ptd_corr_AB,
                se_AB=se_bits_ptd_corr_AB,
                mean_BA=mean_bits_ptd_corr_BA,
                se_BA=se_bits_ptd_corr_BA,
                label_A=A,
                label_B=B,
                sig_group_diff=sig_group_diff,
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                xlabel=paper_xlabel,
            )
            
            # Save paper-quality Panel D IV figure (Inverse-Variance Weighted)
            fig_path_panel_d_iv = figs_dir / f"{pair_name}_panel_d_iv.pdf"
            plot_panel_d_iv_paper(
                out_path=fig_path_panel_d_iv,
                time=time,
                mean_AB=mean_bits_ptd_iv_AB,
                se_AB=se_bits_ptd_iv_AB,
                mean_BA=mean_bits_ptd_iv_BA,
                se_BA=se_bits_ptd_iv_BA,
                label_A=A,
                label_B=B,
                sig_group_diff=sig_group_diff,
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                xlabel=paper_xlabel,
            )

            # Also create reverse figure (B vs A) to show significance from the other perspective
            # For reverse: diff = B->A - A->B (negative of original), swap all data
            pair_name_rev = f"{B}_vs_{A}"
            # Split title into two lines for better readability
            title_rev = (f"{align.upper()} | {tag} | {feature}\n"
                         f"{B} vs {A} | monkey={monkey_label} | N={n_sessions}")
            fig_path_pdf_rev = figs_dir / f"{pair_name_rev}.pdf"
            
            # Compute group diff p-value for reverse direction (using negative dnull_list)
            p_group_diff_rev = None
            sig_group_diff_rev = None
            if group_diff_p and len(dnull_list) > 0:
                # For reverse: mean_diff_rev = -mean_diff, dnull_rev = -dnull
                mean_diff_rev = -mean_diff
                dnull_list_rev = [-dnull for dnull in dnull_list]  # negate each dnull
                p_group_diff_rev = group_null_p_for_mean_diff(
                    mean_diff=mean_diff_rev,
                    dnull_list=dnull_list_rev,
                    B=group_null_B,
                    seed=group_null_seed,
                    smooth_bins=smooth_bins,
                )
                sig_group_diff_rev = (p_group_diff_rev < alpha) & np.isfinite(p_group_diff_rev)
            
            plot_summary_figure(
                out_path_pdf=fig_path_pdf_rev,
                time=time,
                mean_bits_AB=mean_bits_BA,  # swapped: B->A becomes first
                se_bits_AB=se_bits_BA,
                mean_bits_BA=mean_bits_AB,  # swapped: A->B becomes second
                se_bits_BA=se_bits_AB,
                mean_z_AB=mean_z_BA,  # swapped
                se_z_AB=se_z_BA,
                mean_z_BA=mean_z_AB,  # swapped
                se_z_BA=se_z_AB,
                mean_diff=-mean_diff,  # reverse diff: B->A - A->B
                se_diff=se_diff,  # SE is same (symmetric)
                frac_sig_AB=frac_sig_BA,  # show frac_sig_BA in panel 3
                win=win,
                title=title_rev,
                # Rebinned z swapped
                rebin_time_arr=rebin_time_arr,
                mean_z_AB_rebin=mean_z_BA_rebin if mean_z_BA_rebin is not None else None,
                se_z_AB_rebin=se_z_BA_rebin if se_z_BA_rebin is not None else None,
                mean_z_BA_rebin=mean_z_AB_rebin if mean_z_AB_rebin is not None else None,
                se_z_BA_rebin=se_z_AB_rebin if se_z_AB_rebin is not None else None,
                # Group DIFF significance for reverse
                sig_group_diff=sig_group_diff_rev,
                # Labels for reverse perspective
                label_A=B,  # swapped: B becomes first area
                label_B=A,  # swapped: A becomes second area
            )
            
            # Save paper-quality Panel A figure separately (reverse perspective)
            fig_path_panel_a_rev = figs_dir / f"{pair_name_rev}_panel_a.pdf"
            plot_panel_a_paper(
                out_path=fig_path_panel_a_rev,
                time=time,
                mean_bits_AB=mean_bits_BA,  # swapped: B->A becomes first
                se_bits_AB=se_bits_BA,
                mean_bits_BA=mean_bits_AB,  # swapped: A->B becomes second
                se_bits_BA=se_bits_AB,
                label_A=B,  # swapped: B becomes first area
                label_B=A,  # swapped: A becomes second area
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                xlabel=paper_xlabel,
            )
            
            # Save paper-quality Panel C figure separately (reverse perspective)
            fig_path_panel_c_rev = figs_dir / f"{pair_name_rev}_panel_c.pdf"
            plot_panel_c_paper(
                out_path=fig_path_panel_c_rev,
                time=time,
                mean_diff=-mean_diff,  # reverse diff: B->A - A->B
                se_diff=se_diff,
                sig_group_diff=sig_group_diff_rev,
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                y_min=paper_y_min,
                y_max=paper_y_max,
                xlabel=paper_xlabel,
            )
            
            # Save paper-quality Panel D figure separately (reverse perspective)
            fig_path_panel_d_rev = figs_dir / f"{pair_name_rev}_panel_d.pdf"
            plot_panel_d_paper(
                out_path=fig_path_panel_d_rev,
                time=time,
                mean_AB=mean_bits_ptd_corr_BA,   # swapped: B→A becomes first
                se_AB=se_bits_ptd_corr_BA,
                mean_BA=mean_bits_ptd_corr_AB,   # swapped: A→B becomes second
                se_BA=se_bits_ptd_corr_AB,
                label_A=B,
                label_B=A,
                sig_group_diff=sig_group_diff_rev,
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                xlabel=paper_xlabel,
            )
            
            # Save paper-quality Panel D IV figure (reverse perspective, IV-weighted)
            fig_path_panel_d_iv_rev = figs_dir / f"{pair_name_rev}_panel_d_iv.pdf"
            plot_panel_d_iv_paper(
                out_path=fig_path_panel_d_iv_rev,
                time=time,
                mean_AB=mean_bits_ptd_iv_BA,   # swapped: B→A becomes first
                se_AB=se_bits_ptd_iv_BA,
                mean_BA=mean_bits_ptd_iv_AB,   # swapped: A→B becomes second
                se_BA=se_bits_ptd_iv_AB,
                label_A=B,
                label_B=A,
                sig_group_diff=sig_group_diff_rev,
                t_min_ms=paper_t_min_ms,
                t_max_ms=paper_t_max_ms,
                xlabel=paper_xlabel,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc/targ live (default: out)")
    ap.add_argument("--align", choices=["stim", "sacc", "targ", "both", "all"], default="both",
                    help="Which alignments to summarize. 'both' = stim+sacc, 'all' = stim+sacc+targ (default: both)")
    ap.add_argument("--tags", nargs="*",
                    help="Flow tags to summarize (e.g. crsweep-stim-vertical-none-trial). "
                         "If omitted, auto-detect per align.")
    ap.add_argument("--tag_prefix", default="evoked",
                    help="Only process tags starting with this prefix (default: 'evoked'). "
                         "Set to empty string '' or 'all' to process all tags.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for p-values (default: 0.05)")
    ap.add_argument("--win_stim", default="0.10:0.30",
                    help="Window [start:end] in seconds for stim-align summary (default: 0.10:0.30)")
    ap.add_argument("--win_sacc", default="-0.20:0.10",
                    help="Window [start:end] in seconds for sacc-align summary (default: -0.20:0.10)")
    ap.add_argument("--win_targ", default="0.10:0.30",
                    help="Window [start:end] in seconds for targ-align summary (default: 0.10:0.30)")
    ap.add_argument("--features", nargs="*",
                    help="Features to include. Default: stim→['C','R'], sacc→['S'], targ→['T']")
    # NEW: time rebin parameters
    ap.add_argument("--rebin_win", type=float, default=None,
                    help="Optional time window size in seconds for rebinned z panel "
                         "(e.g. 0.05 for 50 ms). If None, no rebin panel.")
    ap.add_argument("--rebin_step", type=float, default=None,
                    help="Optional step size in seconds between rebinned windows "
                         "(e.g. 0.02 for 20 ms). Default: equal to rebin_win.")
    ap.add_argument("--qc_threshold", type=float, default=0.6,
                    help="QC threshold for filtering (default: 0.6). SYMMETRIC "
                         "rejection is applied: for pair (A,B), if EITHER area fails QC, "
                         "the session is excluded for that pair. This ensures both directions "
                         "have the same N. Set to 0 or negative to disable QC filtering.")
    ap.add_argument("--qc_tag", type=str, default=None,
                    help="Explicit QC tag to use for filtering (e.g., 'winsearch-stim-vertical'). "
                         "If not provided, script attempts to auto-detect QC tag from flow tag.")
    ap.add_argument("--group_diff_p", action="store_true", default=True,
                    help="Compute group empirical p(t) for DIFF using saved "
                         "null samples (default: True). Requires flow_*.npz files to have null_samps_AtoB.")
    ap.add_argument("--no_group_diff_p", action="store_false", dest="group_diff_p",
                    help="Disable group empirical p(t) computation for DIFF.")
    ap.add_argument("--group_null_B", type=int, default=4096,
                    help="Number of group-null replicates for DIFF p(t) (default: 4096)")
    ap.add_argument("--group_null_seed", type=int, default=12345,
                    help="RNG seed for group-null sampling (default: 12345)")
    ap.add_argument("--smooth_ms", type=float, default=30.0,
                    help="Smoothing window (ms) for group DIFF p(t). If > 0, applies "
                         "uniform moving average to both observed DIFF and group null "
                         "before computing p-values. Keeps original time resolution. "
                         "E.g., --smooth_ms 50 for 50ms window. Default: 30.")
    ap.add_argument("--sacc_bin_combine", type=int, default=1,
                    help="Number of adjacent time bins to combine for sacc alignment "
                         "(default: 1, no combining). Set to 2 if using 5ms bins and want 10ms output.")
    args = ap.parse_args()
    
    # Parse rebin parameters
    rebin_win = args.rebin_win
    rebin_step = args.rebin_step if args.rebin_step is not None else args.rebin_win

    out_root = Path(args.out_root)
    aligns = []
    if args.align in ("stim", "both", "all"):
        aligns.append("stim")
    if args.align in ("sacc", "both", "all"):
        aligns.append("sacc")
    if args.align in ("targ", "all"):
        aligns.append("targ")

    win_stim = parse_window(args.win_stim)
    win_sacc = parse_window(args.win_sacc)
    win_targ = parse_window(args.win_targ)

    for align in aligns:
        sids = find_sessions(out_root, align)
        if not sids:
            print(f"[warn] No sessions found for align={align} under {out_root}")
            continue

        # discover tags if not provided
        if args.tags:
            tags = args.tags
        else:
            tags = discover_tags(out_root, align, sids)
        
        # Filter tags by prefix if specified
        tag_prefix = args.tag_prefix
        if tag_prefix and tag_prefix.lower() != "all":
            tags = [t for t in tags if t.startswith(tag_prefix)]
            if not tags:
                print(f"[warn] No flow tags starting with '{tag_prefix}' found for align={align}")
                continue
        
        if not tags:
            print(f"[warn] No flow tags found for align={align}")
            continue

        print(f"[info] align={align}, sessions={len(sids)}, tags={tags}")

        if align == "stim":
            win = win_stim
        elif align == "sacc":
            win = win_sacc
        else:  # targ
            win = win_targ

        for tag in tags:
            # discover features for this tag if not explicitly provided
            if args.features:
                feats = args.features
            else:
                feats = discover_features(out_root, align, tag, sids)
                if not feats:
                    print(f"[warn] No features found for tag={tag}, align={align}, skipping")
                    continue
            
            print(f"[tag={tag}] features: {feats}")
            
            for feat in feats:
                print(f"\n[tag={tag}] align={align}, feature={feat}")
                # Apply bin combining only for sacc alignment
                bin_combine = args.sacc_bin_combine if align == "sacc" else 1
                summarize_for_tag_align_feature(
                    out_root=out_root,
                    align=align,
                    tag=tag,
                    feature=feat,
                    sids=sids,
                    alpha=args.alpha,
                    win=win,
                    rebin_win_s=rebin_win,
                    rebin_step_s=rebin_step,
                    qc_threshold=args.qc_threshold,
                    qc_tag=args.qc_tag,
                    group_diff_p=args.group_diff_p,
                    group_null_B=args.group_null_B,
                    group_null_seed=args.group_null_seed,
                    smooth_ms=args.smooth_ms,
                    bin_combine=bin_combine,
                )

    print("\n[done] summary + figures completed.")


if __name__ == "__main__":
    main()
