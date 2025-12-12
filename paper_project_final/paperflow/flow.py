# paper_project_final/paperflow/flow.py
from __future__ import annotations
import json, warnings
from typing import Dict, Tuple, Optional
import numpy as np

from paperflow.standardize import compute_regressor_scaling, apply_regressor_scaling

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

# ---------- label helpers (numeric, mask-safe) ----------

def _label_vec(cache: Dict, key: str, mask: np.ndarray) -> Optional[np.ndarray]:
    """Return masked label vector as float (NaN if missing)."""
    v = cache.get(key, None)
    if v is None:
        return None
    v = np.asarray(v).astype(float)
    return v[mask]

def _encode_joint_labels(cache: Dict, mask: np.ndarray, keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    """
    Encode a tuple of label vectors into a single numeric code so we can
    stratify by joint (e.g., C×R). Works even if C is ±1, R is {1,2,3}, etc.
    """
    vecs = []
    for k in keys:
        v = _label_vec(cache, k, mask)
        if v is None:
            return None
        vecs.append(np.asarray(np.round(v), dtype=int))
    base = 1000  # comfortably exceeds #levels
    code = np.zeros_like(vecs[0], dtype=int)
    for i, v in enumerate(reversed(vecs)):
        code += (base**i) * v
    return code.astype(float)  # keep as float to reuse isfinite()

# -------------------- trial masks & axes --------------------

def _trial_mask(cache: Dict, orientation: Optional[str], pt_min_ms: Optional[float]) -> np.ndarray:
    N = cache["Z"].shape[0]
    ok = np.ones(N, dtype=bool)
    if "lab_is_correct" in cache:
        ok &= cache["lab_is_correct"].astype(bool)
    if orientation is not None and "lab_orientation" in cache:
        ok &= (cache["lab_orientation"].astype(str) == orientation)
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        ok &= np.isfinite(PT) & (PT >= float(pt_min_ms))
    return ok

def _axis_matrix(axes_npz: Dict, feature: str) -> Optional[np.ndarray]:
    """
    Return axis as (U,K). Accept both vector and matrix.
    C: sC (U,) or (U,1)
    R: sR (U,Kr)
    S: sS_inv (preferred), else sS_raw (U,) or (U,1)
    T: sT (U,) or (U,1) - target configuration axis
    O: sO (U,) or (U,1) - context / orientation axis
    """
    if feature == "C":
        a = axes_npz.get("sC", np.array([]))
    elif feature == "R":
        a = axes_npz.get("sR", np.array([[]]))
    elif feature == "S":
        a = axes_npz.get("sS_inv", np.array([]))
        if a.size == 0:
            a = axes_npz.get("sS_raw", np.array([]))
    elif feature == "T":
        a = axes_npz.get("sT", np.array([]))
    elif feature == "O":  # NEW: context / orientation axis
        a = axes_npz.get("sO", np.array([]))
    else:
        a = np.array([])
    if a.size == 0:
        return None
    a = np.array(a)
    if a.ndim == 1:
        a = a[:, None]
    return a  # (U,K)

def _project_multi(cache: Dict, A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Z: (N,B,U); A: (U,K) -> Y: (N,B,K)
    """
    Z = cache["Z"][mask].astype(np.float64)
    if A is None or A.shape[0] != Z.shape[2]:
        raise ValueError("Axis size mismatch.")
    return np.tensordot(Z, A, axes=([2],[0]))  # (N,B,K)

def _subtract_global_evoked(Y: np.ndarray, sigma_bins: float = 0.0) -> np.ndarray:
    """
    Subtract global evoked response: mean across trials per time bin.
    Y: (N, B, K)
    """
    if Y.size == 0:
        return Y
    mu = np.nanmean(Y, axis=0, keepdims=True)  # (1,B,K)

    if sigma_bins and sigma_bins > 0 and gaussian_filter1d is not None:
        mu2 = mu.copy()
        for k in range(mu.shape[2]):
            mu2[0, :, k] = gaussian_filter1d(mu[0, :, k], sigma=sigma_bins, mode="nearest")
        mu = mu2

    return Y - mu

# -------------------- induced removal & lags --------------------

def _induce_by_strata_multi(Y: np.ndarray, labels: Optional[np.ndarray]) -> np.ndarray:
    """
    Y: (N,B,K). Subtract per-time per-stratum mean across trials, independently for each dim K.
    """
    if labels is None:
        return Y
    out = Y.copy()
    lab = labels.copy()
    good = np.isfinite(lab)
    uniq = np.unique(lab[good])
    for u in uniq:
        m = good & (lab == u)
        if not np.any(m):
            continue
        mu = out[m].mean(axis=0, keepdims=True)  # (1,B,K)
        out[m] -= mu
    return out

def _build_lag_stack_multi(Y: np.ndarray, L: int) -> np.ndarray:
    """
    Y: (N,B,K) -> (N,B,L*K) with blocks [Y_{t-1},...,Y_{t-L}] concatenated along last axis.
    """
    N,B,K = Y.shape
    if L <= 0:
        return np.zeros((N,B,0), dtype=float)
    Xlags = np.zeros((N,B,L*K), dtype=float)
    for l in range(1, L+1):
        Xlags[:, l:, (l-1)*K : l*K] = Y[:, :-l, :]
    return Xlags

# -------------------- ridge SSE (multi-output) --------------------

def _ridge_sse_multi(Y: np.ndarray, X: np.ndarray, lam: float) -> float:
    """
    Y: (N,Kout), X: (N,p). Ridge regression with shared predictors;
    returns sum of squared residuals over all outputs (scalar SSE).
    """
    XtX = X.T @ X
    p = XtX.shape[0]
    A = XtX + lam * np.eye(p)
    XtY = X.T @ Y
    B = np.linalg.solve(A, XtY)        # (p,Kout)
    R = Y - X @ B                      # (N,Kout)
    return float(np.sum(R*R))

# -------------------- permutation schemes --------------------

def _permute_within_strata(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Permute trial indices within each stratum defined by labels.
    Returns permutation indices (same size as labels).
    """
    N = labels.shape[0]
    perm = np.arange(N)
    good = np.isfinite(labels)
    if not np.any(good):
        return rng.permutation(N)
    for v in np.unique(labels[good]):
        idx = np.where(good & (labels == v))[0]
        if idx.size > 1:
            perm[idx] = rng.permutation(idx)
    return perm

def _circular_shift_time(Y: np.ndarray, rng: np.random.Generator, min_shift: int = 1) -> np.ndarray:
    """
    Circular shift each trial's time series by a random amount.
    Y: (N, B, K) -> Y_shifted: (N, B, K)
    """
    N, B, K = Y.shape
    Y_shifted = np.empty_like(Y)
    for n in range(N):
        shift = rng.integers(min_shift, B)
        Y_shifted[n] = np.roll(Y[n], shift, axis=0)
    return Y_shifted

def _phase_randomize(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Phase-randomize each trial's time series (preserves power spectrum, breaks phase).
    Y: (N, B, K) -> Y_randomized: (N, B, K)
    """
    N, B, K = Y.shape
    Y_rand = np.empty_like(Y)
    for n in range(N):
        for k in range(K):
            fft_vals = np.fft.rfft(Y[n, :, k])
            phases = rng.uniform(0, 2*np.pi, len(fft_vals))
            phases[0] = 0
            if B % 2 == 0:
                phases[-1] = 0
            fft_rand = np.abs(fft_vals) * np.exp(1j * phases)
            Y_rand[n, :, k] = np.fft.irfft(fft_rand, n=B)
    return Y_rand

# -------------------- null descriptions --------------------

def _null_descriptions(null_method: str) -> Tuple[str, str]:
    if null_method == "trial_shuffle":
        H0 = ("Trial identity is exchangeable within strata; "
              "no trial-specific directed coupling from A to B beyond B self-history.")
        H1 = ("The true pairing of A- and B-trials yields larger ΔLL than shuffled pairings, "
              "indicating trial-specific directed coupling.")
    elif null_method == "circular_shift":
        H0 = ("Within each trial, the absolute temporal alignment between A and B is irrelevant; "
              "random circular shifts of time give the same ΔLL.")
        H1 = ("The correct temporal alignment of A and B yields larger ΔLL than circularly shifted "
              "surrogates, indicating meaningful lag structure.")
    elif null_method == "phase_randomize":
        H0 = ("Only the power spectra of A and B matter; temporal phase relationships between A and B "
              "are irrelevant.")
        H1 = ("Temporal phase relationships between A and B yield larger ΔLL than phase-randomized "
              "surrogates.")
    else:
        H0 = ""
        H1 = ""
    return H0, H1

# -------------------- core engine --------------------

def compute_flow_timecourse_for_pair(
    cacheA: Dict, cacheB: Dict,
    axesA: Dict, axesB: Dict,
    feature: str,                   # 'C' | 'R' | 'S' | 'T' | 'O'
    align: str,                     # 'stim' | 'sacc'
    orientation: Optional[str],
    pt_min_ms: Optional[float],
    lags_ms: float,
    ridge: float,
    perms: int = 500,
    induced: bool = True,
    include_B_lags: bool = True,
    seed: int = 0,
    perm_within: str = "CR",        # 'CR', 'other', 'C', 'R', 'none'
    null_method: str = "trial_shuffle",      # 'trial_shuffle', 'circular_shift', 'phase_randomize'
    standardize_mode: str = "none",          # 'none', 'zscore_regressors'
    evoked_subtract: bool = False,
    evoked_sigma_ms: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Compute time-resolved information flow from area A to area B.

    null_method
      'trial_shuffle'   : shuffle trials within strata, preserve within-trial autocorrelation
      'circular_shift'  : circularly shift each trial in time, break alignment
      'phase_randomize' : randomize temporal phase per trial, preserve power spectrum

    standardize_mode
      'none'             : use regressors as-is
      'zscore_regressors': per-bin z-scoring of regressors (except intercept), parameters
                           computed on observed data and reused for null permutations.
    """

    if standardize_mode not in ("none", "zscore_regressors"):
        raise ValueError(f"Unknown standardize_mode='{standardize_mode}'")
    use_std = (standardize_mode == "zscore_regressors")

    # 1) mask trials
    maskA = _trial_mask(cacheA, orientation, pt_min_ms)
    maskB = _trial_mask(cacheB, orientation, pt_min_ms)
    if maskA.shape[0] != maskB.shape[0]:
        raise ValueError("A and B caches have different #trials.")
    mask = maskA & maskB
    if not np.any(mask):
        raise ValueError("No trials remain after masking.")

    # 2) time / binning
    time = cacheA["time"].astype(float)
    if time.shape != cacheB["time"].shape or not np.allclose(time, cacheB["time"]):
        raise ValueError("A and B caches have different time grids.")
    metaA = cacheA["meta"]
    if not isinstance(metaA, dict):
        metaA = json.loads(metaA.item() if hasattr(metaA, "item") else str(metaA))
    bin_s = float(metaA.get("bin_s", (time[1]-time[0] if time.size>1 else 0.01)))

    Bbins = time.size
    L = int(max(1, round(lags_ms / (bin_s*1000.0))))
    start_b = L
    rng = np.random.default_rng(seed)

    # 3) axes (multi-D)
    Aaxis = _axis_matrix(axesA, feature)
    Baxis = _axis_matrix(axesB, feature)
    if Aaxis is None or Baxis is None:
        raise ValueError(f"Missing {feature} axis in one of the areas.")
    K_A = Aaxis.shape[1]
    K_B = Baxis.shape[1]

    # 4) projections
    YA_full = _project_multi(cacheA, Aaxis, mask)  # (N,B,K_A)
    YB_full = _project_multi(cacheB, Baxis, mask)  # (N,B,K_B)

    # Optional old-style evoked subtraction (global PSTH across trials)
    if evoked_subtract:
        sigma_bins = 0.0
        if evoked_sigma_ms and evoked_sigma_ms > 0:
            sigma_bins = float(evoked_sigma_ms) / (bin_s * 1000.0)  # ms -> bins
        YA_full = _subtract_global_evoked(YA_full, sigma_bins=sigma_bins)
        YB_full = _subtract_global_evoked(YB_full, sigma_bins=sigma_bins)

    # 5) induced removal (per-time per-stratum mean)
    labs_induce = None
    if induced:
        if feature == "C":
            labs_induce = _label_vec(cacheA, "lab_R", mask)  # remove direction means
        elif feature == "R":
            labs_induce = _label_vec(cacheA, "lab_C", mask)  # remove category means
        elif feature == "T":
            # For target configuration, default to removing direction means (same choice as C).
            labs_induce = _label_vec(cacheA, "lab_R", mask)
        elif feature in ("S", "O"):
            # For S and O, subtract per-(C,R) means and keep residual fluctuations
            labs_induce = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
    YA = _induce_by_strata_multi(YA_full, labs_induce) if induced else YA_full
    YB = _induce_by_strata_multi(YB_full, labs_induce) if induced else YB_full

    # 6) lag stacks
    XA_lags = _build_lag_stack_multi(YA, L)  # (N,B,L*K_A)
    XB_lags = _build_lag_stack_multi(YB, L)  # (N,B,L*K_B)

    # 7) ΔLL per bin (multi-output)
    N = YA.shape[0]
    ones = np.ones((N,1), dtype=float)
    bits_AtoB = np.full(Bbins, np.nan)
    bits_BtoA = np.full(Bbins, np.nan)

    # Pre-compute reduced model SSE (doesn't depend on A, so same for observed and null)
    sse_red_cache = {}
    sse_red_rev_cache = {}

    # Scaling for full-model regressors (used for null as well)
    scaling_A_full = {}
    scaling_B_full = {}

    for b in range(start_b, Bbins):
        # ---------- A -> B ----------
        Yt = YB[:, b, :]  # (N,K_B)

        # reduced: intercept + B self-lags (optional)
        X_red = ones
        if include_B_lags and XB_lags.shape[2] > 0:
            X_red = np.concatenate([X_red, XB_lags[:, b, :]], axis=1)

        if use_std:
            mu_red, sigma_red = compute_regressor_scaling(X_red, has_intercept=True)
            X_red_used = apply_regressor_scaling(X_red, mu_red, sigma_red)
        else:
            X_red_used = X_red

        sse_red = _ridge_sse_multi(Yt, X_red_used, ridge)
        sse_red_cache[b] = sse_red

        # full: intercept + B self-lags + A lags
        X_full = np.concatenate([X_red, XA_lags[:, b, :]], axis=1)
        if use_std:
            mu_full, sigma_full = compute_regressor_scaling(X_full, has_intercept=True)
            scaling_A_full[b] = (mu_full, sigma_full)
            X_full_used = apply_regressor_scaling(X_full, mu_full, sigma_full)
        else:
            X_full_used = X_full

        sse_full = _ridge_sse_multi(Yt, X_full_used, ridge)
        if sse_full > 0 and sse_red > 0:
            bits_AtoB[b] = 0.5 * N * np.log2(sse_red / sse_full)

        # ---------- B -> A ----------
        Yt_rev = YA[:, b, :]  # (N,K_A)

        X_red_r = ones
        if include_B_lags and XA_lags.shape[2] > 0:
            X_red_r = np.concatenate([X_red_r, XA_lags[:, b, :]], axis=1)

        if use_std:
            mu_red_r, sigma_red_r = compute_regressor_scaling(X_red_r, has_intercept=True)
            X_red_r_used = apply_regressor_scaling(X_red_r, mu_red_r, sigma_red_r)
        else:
            X_red_r_used = X_red_r

        sse_red_r = _ridge_sse_multi(Yt_rev, X_red_r_used, ridge)
        sse_red_rev_cache[b] = sse_red_r

        X_full_r = np.concatenate([X_red_r, XB_lags[:, b, :]], axis=1)
        if use_std:
            mu_full_r, sigma_full_r = compute_regressor_scaling(X_full_r, has_intercept=True)
            scaling_B_full[b] = (mu_full_r, sigma_full_r)
            X_full_r_used = apply_regressor_scaling(X_full_r, mu_full_r, sigma_full_r)
        else:
            X_full_r_used = X_full_r

        sse_full_r = _ridge_sse_multi(Yt_rev, X_full_r_used, ridge)
        if sse_full_r > 0 and sse_red_r > 0:
            bits_BtoA[b] = 0.5 * N * np.log2(sse_red_r / sse_full_r)

    # 8) permutation null (one-sided)
    null_mean_A = np.full(Bbins, np.nan); null_std_A = np.full(Bbins, np.nan); p_A = np.full(Bbins, np.nan)
    null_mean_B = np.full(Bbins, np.nan); null_std_B = np.full(Bbins, np.nan); p_B = np.full(Bbins, np.nan)

    n_actual_shuffles = 0
    strata_counts = {}

    if perms > 0:
        P = int(perms)
        all_A = np.full((P, Bbins-start_b), np.nan)
        all_B = np.full((P, Bbins-start_b), np.nan)

        # Choose permutation strata
        if perm_within == "CR":
            perm_labels = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
        elif perm_within == "other":
            if feature == "C":
                perm_labels = _label_vec(cacheA, "lab_R", mask)
            elif feature == "R":
                perm_labels = _label_vec(cacheA, "lab_C", mask)
            else:
                perm_labels = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
        elif perm_within == "C":
            perm_labels = _label_vec(cacheA, "lab_C", mask)
        elif perm_within == "R":
            perm_labels = _label_vec(cacheA, "lab_R", mask)
        elif perm_within == "none":
            perm_labels = None
        else:
            raise ValueError(f"Unknown perm_within='{perm_within}'")

        if perm_labels is None:
            strata = np.zeros(N, dtype=float)
            warnings.warn("perm_within='none' or labels missing: shuffling across all trials (no stratification)")
        else:
            n_valid = np.sum(np.isfinite(perm_labels))
            if n_valid == 0:
                warnings.warn("All permutation labels are NaN! Shuffling across all trials.")
                strata = np.zeros(N, dtype=float)
            elif n_valid < N:
                warnings.warn(f"Only {n_valid}/{N} trials have valid permutation labels. "
                              f"Trials with NaN labels will be shuffled together.")
                strata = perm_labels
            else:
                strata = perm_labels

        good = np.isfinite(strata)
        for v in np.unique(strata[good]):
            strata_counts[float(v)] = int(np.sum(strata == v))

        for p in range(P):
            # Generate null data based on method
            if null_method == "trial_shuffle":
                perm_idx_A = _permute_within_strata(strata, rng)
                perm_idx_B = _permute_within_strata(strata, rng)
                if not np.array_equal(perm_idx_A, np.arange(N)):
                    n_actual_shuffles += 1
                YA_null = YA[perm_idx_A]
                YB_null = YB[perm_idx_B]
            elif null_method == "circular_shift":
                YA_null = _circular_shift_time(YA, rng, min_shift=L+1)
                YB_null = _circular_shift_time(YB, rng, min_shift=L+1)
                n_actual_shuffles += 1
            elif null_method == "phase_randomize":
                YA_null = _phase_randomize(YA, rng)
                YB_null = _phase_randomize(YB, rng)
                n_actual_shuffles += 1
            else:
                raise ValueError(f"Unknown null_method='{null_method}'")

            XA_lags_null = _build_lag_stack_multi(YA_null, L)
            XB_lags_null = _build_lag_stack_multi(YB_null, L)

            vals_A = np.full(Bbins - start_b, np.nan)
            vals_B = np.full(Bbins - start_b, np.nan)

            for i, b in enumerate(range(start_b, Bbins)):
                # ---------- A -> B null (null source A) ----------
                Yt = YB[:, b, :]
                sse_red = sse_red_cache[b]

                X_red = ones
                if include_B_lags and XB_lags.shape[2] > 0:
                    X_red = np.concatenate([X_red, XB_lags[:, b, :]], axis=1)

                X_full = np.concatenate([X_red, XA_lags_null[:, b, :]], axis=1)
                if use_std:
                    mu_full, sigma_full = scaling_A_full[b]
                    X_full_used = apply_regressor_scaling(X_full, mu_full, sigma_full)
                else:
                    X_full_used = X_full

                sse_full = _ridge_sse_multi(Yt, X_full_used, ridge)
                if sse_full > 0 and sse_red > 0:
                    vals_A[i] = 0.5 * N * np.log2(sse_red / sse_full)

                # ---------- B -> A null (null source B) ----------
                Yt_rev = YA[:, b, :]
                sse_red_r = sse_red_rev_cache[b]

                X_red_r = ones
                if include_B_lags and XA_lags.shape[2] > 0:
                    X_red_r = np.concatenate([X_red_r, XA_lags[:, b, :]], axis=1)

                X_full_r = np.concatenate([X_red_r, XB_lags_null[:, b, :]], axis=1)
                if use_std:
                    mu_full_r, sigma_full_r = scaling_B_full[b]
                    X_full_r_used = apply_regressor_scaling(X_full_r, mu_full_r, sigma_full_r)
                else:
                    X_full_r_used = X_full_r

                sse_full_r = _ridge_sse_multi(Yt_rev, X_full_r_used, ridge)
                if sse_full_r > 0 and sse_red_r > 0:
                    vals_B[i] = 0.5 * N * np.log2(sse_red_r / sse_full_r)

            all_A[p] = vals_A
            all_B[p] = vals_B

        if n_actual_shuffles == 0:
            warnings.warn("NO ACTUAL SHUFFLING OCCURRED IN ANY PERMUTATION! "
                          "Null distribution is identical to observed. Check labels.")

        null_mean = np.nanmean(all_A, axis=0)
        null_std = np.nanstd(all_A, axis=0, ddof=1)
        null_mean_r = np.nanmean(all_B, axis=0)
        null_std_r = np.nanstd(all_B, axis=0, ddof=1)

        obsA = bits_AtoB[start_b:]
        obsB = bits_BtoA[start_b:]

        pA_full = np.full(obsA.shape, np.nan)
        pB_full = np.full(obsB.shape, np.nan)

        for i in range(len(obsA)):
            if np.isfinite(obsA[i]):
                null_vals = all_A[:, i]
                valid_null = np.isfinite(null_vals)
                if np.any(valid_null):
                    counts = np.sum(null_vals[valid_null] >= obsA[i])
                    pA_full[i] = (1 + counts) / (1 + np.sum(valid_null))
            if np.isfinite(obsB[i]):
                null_vals = all_B[:, i]
                valid_null = np.isfinite(null_vals)
                if np.any(valid_null):
                    counts = np.sum(null_vals[valid_null] >= obsB[i])
                    pB_full[i] = (1 + counts) / (1 + np.sum(valid_null))

        null_mean_A[start_b:] = null_mean
        null_std_A[start_b:]  = null_std
        p_A[start_b:]         = pA_full

        null_mean_B[start_b:] = null_mean_r
        null_std_B[start_b:]  = null_std_r
        p_B[start_b:]         = pB_full

    H0_desc, H1_desc = _null_descriptions(null_method)

    return dict(
        time=time,
        bits_AtoB=bits_AtoB, bits_BtoA=bits_BtoA,
        null_mean_AtoB=null_mean_A, null_std_AtoB=null_std_A, p_AtoB=p_A,
        null_mean_BtoA=null_mean_B, null_std_BtoA=null_std_B, p_BtoA=p_B,
        meta=dict(
            feature=feature, align=align, orientation=orientation, pt_min_ms=pt_min_ms,
            lags_ms=lags_ms, ridge=ridge, perms=int(perms), induced=bool(induced),
            include_B_lags=bool(include_B_lags),
            perm_within=str(perm_within),
            null_method=str(null_method),
            standardize_mode=str(standardize_mode),
            evoked_subtract=bool(evoked_subtract),
            evoked_sigma_ms=float(evoked_sigma_ms),
            H0_description=H0_desc,
            H1_description=H1_desc,
            induced_labels=(
                "CR" if (feature in ("S","O") and induced) else
                ("R" if (feature=="C" and induced) else
                 ("C" if (feature=="R" and induced) else
                  ("R" if (feature=="T" and induced) else "none")))
            ),
            N=int(mask.sum()),
            K_A=int(K_A), K_B=int(K_B),
            U_A=int(cacheA["Z"].shape[2]), U_B=int(cacheB["Z"].shape[2]),
            n_strata=len(strata_counts) if perms > 0 else 0,
            strata_sizes=strata_counts if perms > 0 else {},
        )
    )
