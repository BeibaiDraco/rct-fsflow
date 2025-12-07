# paper_project_final/paperflow/standardize.py
from __future__ import annotations
from typing import Tuple
import numpy as np

def compute_regressor_scaling(X: np.ndarray, has_intercept: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std per column for regressors.

    X : (N, p)
    has_intercept : if True, column 0 is treated as an intercept and left unscaled.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0, ddof=1)
    # Avoid zero / NaN std
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    sigma[bad] = 1.0
    if has_intercept and X.shape[1] > 0:
        mu[0] = 0.0
        sigma[0] = 1.0
    return mu, sigma

def apply_regressor_scaling(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Apply precomputed scaling to regressors.

    X : (N, p)
    mu, sigma : (p,)
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != mu.shape[0] or mu.shape != sigma.shape:
        raise ValueError(f"Shape mismatch in scaling: X {X.shape}, mu {mu.shape}, sigma {sigma.shape}")
    return (X - mu) / sigma
