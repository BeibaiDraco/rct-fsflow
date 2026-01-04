# paperflow/norm.py
"""
Normalization utilities for baseline normalization and consistent 
normalization across training, QC, and projection.

Key functions:
- parse_win: Parse "a:b" window string to (float, float)
- baseline_stats: Compute (mu, sd) from baseline window
- apply_z: Apply z-scoring with given (mu, sd)
- get_Z: Get normalized data from cache with proper normalization mode
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import warnings


def parse_win(s: str) -> Tuple[float, float]:
    """
    Parse a window string "a:b" into (float, float).
    
    Examples:
        "-0.20:0.00" -> (-0.20, 0.00)
        "0.10:0.30" -> (0.10, 0.30)
    """
    if not isinstance(s, str) or ":" not in s:
        raise ValueError(f"Expected window string in 'a:b' format, got: {s!r}")
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected window string in 'a:b' format, got: {s!r}")
    return float(parts[0]), float(parts[1])


def baseline_stats(
    X: np.ndarray,
    time_s: np.ndarray,
    baseline_win: Tuple[float, float],
    trial_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute baseline normalization statistics (mu, sd) from raw spike counts.
    
    Math:
        Let Xsel = X[trial_mask][:, baseline_bins, :], flattened over trials×bins.
        mu_u = mean(Xsel[:, u])
        sd_u = std(Xsel[:, u], ddof=1)
        If sd_u is zero or NaN, set sd_u = 1.0.
    
    Parameters
    ----------
    X : np.ndarray
        Raw binned counts, shape (trials, time_bins, units).
    time_s : np.ndarray
        Time array in seconds, shape (time_bins,).
    baseline_win : tuple of (float, float)
        Baseline window (t0, t1) in seconds.
    trial_mask : np.ndarray, optional
        Boolean mask for which trials to use. If None, use all trials.
    
    Returns
    -------
    mu : np.ndarray
        Mean per unit, shape (units,), dtype float64.
    sd : np.ndarray
        Std per unit, shape (units,), dtype float64.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (trials, bins, units), got shape {X.shape}")
    
    # Apply trial mask
    if trial_mask is not None:
        X_masked = X[trial_mask]
    else:
        X_masked = X
    
    if X_masked.shape[0] == 0:
        raise ValueError("No trials after masking")
    
    # Find baseline bins
    t0, t1 = baseline_win
    baseline_mask = (time_s >= t0) & (time_s <= t1)
    
    if not np.any(baseline_mask):
        warnings.warn(f"No bins found in baseline window [{t0:.3f}, {t1:.3f}]. "
                      f"Time range is [{time_s.min():.3f}, {time_s.max():.3f}]. "
                      "Using all bins for normalization.")
        baseline_mask = np.ones(time_s.shape[0], dtype=bool)
    
    # Extract baseline data: (trials, baseline_bins, units)
    X_baseline = X_masked[:, baseline_mask, :]
    
    # Flatten over trials and bins: (trials*baseline_bins, units)
    n_trials, n_baseline_bins, n_units = X_baseline.shape
    X_flat = X_baseline.reshape(-1, n_units).astype(np.float64)
    
    # Compute per-unit statistics
    mu = np.nanmean(X_flat, axis=0)
    sd = np.nanstd(X_flat, axis=0, ddof=1)
    
    # Handle zero/NaN std
    bad_sd = ~np.isfinite(sd) | (sd <= 0)
    sd[bad_sd] = 1.0
    
    # Handle NaN mean (shouldn't happen with valid data, but be safe)
    bad_mu = ~np.isfinite(mu)
    mu[bad_mu] = 0.0
    
    return mu.astype(np.float64), sd.astype(np.float64)


def apply_z(
    X: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """
    Apply z-scoring: Z = (X - mu) / sd, broadcasting over time and trials.
    
    Parameters
    ----------
    X : np.ndarray
        Data to normalize, shape (trials, bins, units) or (bins, units).
    mu : np.ndarray
        Mean per unit, shape (units,).
    sd : np.ndarray
        Std per unit, shape (units,).
    
    Returns
    -------
    Z : np.ndarray
        Normalized data, same shape as X, dtype float64.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    
    # Protect against division by zero
    sd_safe = sd.copy()
    sd_safe[sd_safe <= 0] = 1.0
    
    return ((X.astype(np.float64) - mu) / sd_safe)


def get_Z(
    cache: Dict,
    time_s: np.ndarray,
    trial_mask: np.ndarray,
    norm_mode: str,
    baseline_win: Optional[Tuple[float, float]] = None,
    axes_norm: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Get normalized data from cache with the specified normalization mode.
    
    Parameters
    ----------
    cache : dict
        Cache dictionary with keys "X" (raw counts) and/or "Z" (global z-score).
    time_s : np.ndarray
        Time array in seconds.
    trial_mask : np.ndarray
        Boolean mask for which trials to select.
    norm_mode : str
        Normalization mode: "global", "baseline", or "none".
    baseline_win : tuple of (float, float), optional
        Baseline window in seconds. Required if norm_mode=="baseline" and
        axes_norm does not contain mu/sd.
    axes_norm : dict, optional
        Dictionary with "mu" and "sd" arrays from axes. If provided and non-empty,
        these will be used for baseline normalization instead of recomputing.
    
    Returns
    -------
    Z : np.ndarray
        Normalized data, shape (sum(trial_mask), bins, units), dtype float64.
    norm_meta : dict
        Dictionary describing what was used:
        - "norm_mode": the mode used
        - "baseline_win": the window used (if baseline)
        - "used_axes_mu_sd": whether axes-provided mu/sd were used
        - "computed_mu_sd": whether mu/sd were computed fresh
        - "mu": the mu array used (if baseline)
        - "sd": the sd array used (if baseline)
    """
    norm_meta = {
        "norm_mode": norm_mode,
        "baseline_win": None,
        "used_axes_mu_sd": False,
        "computed_mu_sd": False,
    }
    
    if norm_mode == "global":
        # Use pre-computed global z-score from cache
        if "Z" not in cache:
            raise KeyError("Cache missing 'Z' (global z-score). "
                          "Cannot use norm_mode='global'. Try 'baseline' or 'none'.")
        Z = cache["Z"][trial_mask].astype(np.float64)
        return Z, norm_meta
    
    elif norm_mode == "none":
        # Use raw counts, no normalization
        if "X" not in cache:
            raise KeyError("Cache missing 'X' (raw counts). "
                          "Cannot use norm_mode='none'. "
                          "Consider rebuilding caches or using 'global'.")
        Z = cache["X"][trial_mask].astype(np.float64)
        return Z, norm_meta
    
    elif norm_mode == "baseline":
        # Check if axes provides mu/sd
        axes_mu = None
        axes_sd = None
        if axes_norm is not None:
            axes_mu = axes_norm.get("mu")
            axes_sd = axes_norm.get("sd")
            # Check if they are valid (non-empty arrays)
            if axes_mu is not None and np.asarray(axes_mu).size > 0:
                axes_mu = np.asarray(axes_mu, dtype=np.float64)
            else:
                axes_mu = None
            if axes_sd is not None and np.asarray(axes_sd).size > 0:
                axes_sd = np.asarray(axes_sd, dtype=np.float64)
            else:
                axes_sd = None
        
        # Get raw counts
        if "X" not in cache:
            raise KeyError("Cache missing 'X' (raw counts). "
                          "Cannot use norm_mode='baseline'. "
                          "Either rebuild caches or fall back to 'global'.")
        X = cache["X"]
        
        if axes_mu is not None and axes_sd is not None:
            # Use axes-provided mu/sd
            mu, sd = axes_mu, axes_sd
            norm_meta["used_axes_mu_sd"] = True
            norm_meta["baseline_win"] = baseline_win  # May be None, that's OK
        else:
            # Compute mu/sd from baseline window on masked trials
            if baseline_win is None:
                raise ValueError("norm_mode='baseline' requires baseline_win when "
                               "axes_norm does not provide mu/sd")
            mu, sd = baseline_stats(X, time_s, baseline_win, trial_mask)
            norm_meta["computed_mu_sd"] = True
            norm_meta["baseline_win"] = list(baseline_win)
        
        # Apply normalization to masked trials
        X_masked = X[trial_mask]
        Z = apply_z(X_masked, mu, sd)
        
        # Store mu/sd in meta for potential downstream use
        norm_meta["mu"] = mu
        norm_meta["sd"] = sd
        
        return Z, norm_meta
    
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode!r}. "
                        f"Expected 'global', 'baseline', or 'none'.")


def get_axes_norm(axes: Dict) -> Optional[Dict]:
    """
    Extract normalization parameters (mu, sd) from axes dictionary.
    
    Parameters
    ----------
    axes : dict
        Axes dictionary (loaded from axes_*.npz).
    
    Returns
    -------
    axes_norm : dict or None
        Dictionary with "mu" and "sd" if present in axes, else None.
    """
    norm_mu = axes.get("norm_mu")
    norm_sd = axes.get("norm_sd")
    
    if norm_mu is None or norm_sd is None:
        return None
    
    norm_mu = np.asarray(norm_mu)
    norm_sd = np.asarray(norm_sd)
    
    if norm_mu.size == 0 or norm_sd.size == 0:
        return None
    
    return {"mu": norm_mu, "sd": norm_sd}


def get_axes_norm_mode(axes: Dict) -> str:
    """
    Get the normalization mode used when training axes.
    
    Parameters
    ----------
    axes : dict
        Axes dictionary (loaded from axes_*.npz).
    
    Returns
    -------
    norm_mode : str
        "global", "baseline", or "none".
    """
    meta = axes.get("meta", {})
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    elif hasattr(meta, "item"):
        import json
        try:
            meta = json.loads(meta.item())
        except Exception:
            meta = {}
    
    return meta.get("norm", "global")


def get_axes_baseline_win(axes: Dict) -> Optional[Tuple[float, float]]:
    """
    Get the baseline window used when training axes.
    
    Parameters
    ----------
    axes : dict
        Axes dictionary (loaded from axes_*.npz).
    
    Returns
    -------
    baseline_win : tuple of (float, float) or None
    """
    meta = axes.get("meta", {})
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    elif hasattr(meta, "item"):
        import json
        try:
            meta = json.loads(meta.item())
        except Exception:
            meta = {}
    
    bw = meta.get("baseline_win")
    if bw is None:
        return None
    if isinstance(bw, (list, tuple)) and len(bw) == 2:
        return (float(bw[0]), float(bw[1]))
    return None


# ============== TIME REBINNING ==============

def rebin_cache_data(
    cache: Dict,
    rebin_factor: int,
) -> Tuple[Dict, np.ndarray]:
    """
    Rebin cache data by combining adjacent time bins.
    
    This function modifies the cache in-place and returns the new time array.
    Combines `rebin_factor` adjacent bins by averaging (for X, Z) or taking
    the center time point.
    
    Parameters
    ----------
    cache : dict
        Cache dictionary with keys "X" (raw counts), "Z" (global z-score),
        "time" (time array), and optionally others.
    rebin_factor : int
        Number of adjacent bins to combine. If 1, returns original data unchanged.
    
    Returns
    -------
    cache : dict
        Modified cache with rebinned X, Z, and time arrays.
    new_time : np.ndarray
        New time array after rebinning.
    
    Notes
    -----
    - X and Z are averaged over combined bins (maintains spike rate interpretation)
    - time is averaged over combined bins (gives bin centers)
    - Other cache fields (labels, meta) are preserved unchanged
    - If T % rebin_factor != 0, trailing bins are dropped
    """
    if rebin_factor <= 1:
        return cache, cache["time"].astype(float)
    
    time = cache["time"].astype(float)
    T = time.shape[0]
    T_new = T // rebin_factor
    
    if T_new == 0:
        warnings.warn(f"rebin_factor={rebin_factor} is larger than T={T}. No rebinning applied.")
        return cache, time
    
    # Truncate to multiple of rebin_factor
    T_trunc = T_new * rebin_factor
    
    # Rebin time array
    time_trunc = time[:T_trunc]
    time_reshaped = time_trunc.reshape(T_new, rebin_factor)
    new_time = np.mean(time_reshaped, axis=1)
    cache["time"] = new_time
    
    # Rebin X if present: (trials, time, units) -> (trials, time_new, units)
    if "X" in cache:
        X = cache["X"]
        if X.ndim == 3:
            N, T_x, U = X.shape
            if T_x >= T_trunc:
                X_trunc = X[:, :T_trunc, :]
                X_reshaped = X_trunc.reshape(N, T_new, rebin_factor, U)
                cache["X"] = np.mean(X_reshaped, axis=2)
    
    # Rebin Z if present: (trials, time, units) -> (trials, time_new, units)
    if "Z" in cache:
        Z = cache["Z"]
        if Z.ndim == 3:
            N, T_z, U = Z.shape
            if T_z >= T_trunc:
                Z_trunc = Z[:, :T_trunc, :]
                Z_reshaped = Z_trunc.reshape(N, T_new, rebin_factor, U)
                cache["Z"] = np.mean(Z_reshaped, axis=2)
    
    return cache, new_time


def get_rebin_factor_from_meta(cache: Dict) -> int:
    """
    Get the rebin factor used when creating the cache (if any).
    
    Parameters
    ----------
    cache : dict
        Cache dictionary.
    
    Returns
    -------
    rebin_factor : int
        The rebin factor, or 1 if not rebinned.
    """
    meta = cache.get("meta", {})
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    elif hasattr(meta, "item"):
        import json
        try:
            meta = json.loads(meta.item())
        except Exception:
            meta = {}
    
    return meta.get("rebin_factor", 1)

