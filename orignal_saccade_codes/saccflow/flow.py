# saccflow/flow.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from .io import load_area_cache

# ---------- results containers ----------

@dataclass
class FlowResult:
    time: np.ndarray                 # (T,)
    bits_AtoB: np.ndarray            # (T,)
    bits_BtoA: np.ndarray            # (T,)
    null_mean_AtoB: np.ndarray       # (T,)
    null_std_AtoB:  np.ndarray       # (T,)
    null_mean_BtoA: np.ndarray       # (T,)
    null_std_BtoA:  np.ndarray       # (T,)
    p_AtoB: np.ndarray               # (T,)
    p_BtoA: np.ndarray               # (T,)
    meta: Dict

@dataclass
class PairDiffResult:
    time: np.ndarray                 # (T,)
    diff_bits: np.ndarray            # (T,)  A->B - B->A
    null_mean_diff: np.ndarray       # (T,)
    null_std_diff:  np.ndarray       # (T,)
    p_diff: np.ndarray               # (T,)  two-sided
    z_diff: np.ndarray               # (T,)
    meta: Dict

# ---------- helpers ----------

def _load_axis_vec(axes_dir: str, area: str, prefer="sS_inv") -> np.ndarray:
    path = os.path.join(axes_dir, f"axes_{area}.npz")
    d = np.load(path, allow_pickle=True)
    if prefer in d and d[prefer].size > 0:
        return d[prefer].astype(np.float64)
    key = "sS_raw" if "sS_raw" in d else ("sS" if "sS" in d else None)
    if key is None or d[key].size == 0:
        raise FileNotFoundError(f"No saccade axis found in {path}")
    return d[key].astype(np.float64)

def _project(cache: Dict, axis_vec: np.ndarray, mask_trials: np.ndarray) -> np.ndarray:
    Z = cache["Z"][mask_trials]            # (N,B,U)
    proj = np.tensordot(Z, axis_vec, axes=([2],[0]))  # (N,B)
    return proj.astype(np.float64)

def _induce_by_group(S: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = S.copy()
    for g in (-1.0, 1.0):
        m = (groups == g)
        if np.any(m):
            out[m] = out[m] - out[m].mean(axis=0, dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out

def _build_design(y: np.ndarray, A: np.ndarray, B: np.ndarray, t: int, L: int,
                  covars: Optional[np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    N = y.shape[0]
    cols_B, cols_A = [], []
    for k in range(1, L+1):
        cols_B.append(B[:, t-k]); cols_A.append(A[:, t-k])
    X_B = np.column_stack(cols_B) if cols_B else np.zeros((N,0))
    X_A = np.column_stack(cols_A) if cols_A else np.zeros((N,0))
    X_cov = (covars if covars is not None else np.ones((N,1)))
    X_full = np.column_stack([X_cov, X_B, X_A])   # [cov, autoB, crossA]
    X_red  = np.column_stack([X_cov, X_B])        # [cov, autoB]
    return X_full, X_red, y, np.arange(X_cov.shape[1]+X_B.shape[1], X_cov.shape[1]+X_B.shape[1]+X_A.shape[1])

def _ridge_sse(y: np.ndarray, X: np.ndarray, alpha: float) -> float:
    a = float(alpha) if alpha is not None else 0.0
    XtX = X.T @ X
    if a > 0: XtX = XtX + a * np.eye(X.shape[1], dtype=XtX.dtype)
    w = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ w
    return float(np.dot(r, r))

def _ll_bits_from_sse(sse_full: float, sse_red: float, N: int) -> float:
    sse_full = max(sse_full, 1e-12); sse_red = max(sse_red, 1e-12)
    return 0.5 * N * np.log2(sse_red / sse_full)

def _permute_rows(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return X[idx]

# ---------- core single-direction computation ----------

def compute_sflow_timecourse(
    A_sig: np.ndarray, B_sig: np.ndarray, time_s: np.ndarray,
    C_labels: np.ndarray, PT_ms: Optional[np.ndarray],
    lags_bins: int = 6, alpha: float = 1.0, permutations: int = 500,
    use_induced: bool = True, condition_on_C: bool = True, include_PT_cov: bool = False,
    rng_seed: int = 0
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    N, B = A_sig.shape

    # covariates
    covs = [np.ones((N,1))]
    if condition_on_C: covs.append(C_labels.reshape(-1,1).astype(np.float64))
    if include_PT_cov and PT_ms is not None:
        zPT = (PT_ms - np.nanmean(PT_ms)) / (np.nanstd(PT_ms) + 1e-9)
        covs.append(zPT.reshape(-1,1).astype(np.float64))
    Xcov = np.column_stack(covs) if len(covs) else None

    A0, B0 = ( _induce_by_group(A_sig, C_labels), _induce_by_group(B_sig, C_labels) ) if use_induced else (A_sig, B_sig)

    t_idx = np.arange(lags_bins, B)
    T = len(t_idx)

    bits_AtoB = np.zeros(T); bits_BtoA = np.zeros(T)
    p_AtoB    = np.ones(T);  p_BtoA    = np.ones(T)
    null_m_A  = np.zeros(T); null_s_A  = np.zeros(T)
    null_m_B  = np.zeros(T); null_s_B  = np.zeros(T)

    for j, t in enumerate(t_idx):
        # A->B
        y = B0[:, t]
        Xf, Xr, y_t, _ = _build_design(y, A0, B0, t, lags_bins, Xcov)
        Nobs = y_t.shape[0]
        sse_f = _ridge_sse(y_t, Xf, alpha); sse_r = _ridge_sse(y_t, Xr, alpha)
        obs = _ll_bits_from_sse(sse_f, sse_r, Nobs)
        bits_AtoB[j] = obs

        null_vals = np.zeros(permutations)
        for p in range(permutations):
            idx = rng.permutation(N)
            Xf_p, _, _, _ = _build_design(y, _permute_rows(A0, idx), B0, t, lags_bins, Xcov)
            sse_fp = _ridge_sse(y, Xf_p, alpha)
            null_vals[p] = _ll_bits_from_sse(sse_fp, sse_r, Nobs)
        null_m_A[j] = float(null_vals.mean()); null_s_A[j] = float(null_vals.std(ddof=1))
        p_AtoB[j] = (np.sum(np.abs(null_vals - null_m_A[j]) >= np.abs(obs - null_m_A[j])) + 1.0) / (permutations + 1.0)

        # B->A
        y2 = A0[:, t]
        Xf2, Xr2, y2_t, _ = _build_design(y2, B0, A0, t, lags_bins, Xcov)
        Nobs2 = y2_t.shape[0]
        sse_f2 = _ridge_sse(y2_t, Xf2, alpha); sse_r2 = _ridge_sse(y2_t, Xr2, alpha)
        obs2 = _ll_bits_from_sse(sse_f2, sse_r2, Nobs2)
        bits_BtoA[j] = obs2

        null_vals2 = np.zeros(permutations)
        for p in range(permutations):
            idx = rng.permutation(N)
            Xf2_p, _, _, _ = _build_design(y2, _permute_rows(B0, idx), A0, t, lags_bins, Xcov)
            sse_f2p = _ridge_sse(y2, Xf2_p, alpha)
            null_vals2[p] = _ll_bits_from_sse(sse_f2p, sse_r2, Nobs2)
        null_m_B[j] = float(null_vals2.mean()); null_s_B[j] = float(null_vals2.std(ddof=1))
        p_BtoA[j] = (np.sum(np.abs(null_vals2 - null_m_B[j]) >= np.abs(obs2 - null_m_B[j])) + 1.0) / (permutations + 1.0)

    return dict(
        time=time_s[t_idx],
        bits_AtoB=bits_AtoB, bits_BtoA=bits_BtoA,
        null_mean_AtoB=null_m_A, null_std_AtoB=null_s_A, p_AtoB=p_AtoB,
        null_mean_BtoA=null_m_B, null_std_BtoA=null_s_B, p_BtoA=p_BtoA
    )

# ---------- wrappers (no bands) ----------

def compute_saccade_flow_for_pair(
    out_root: str, sid: str, areaA: str, areaB: str, axes_dir: Optional[str] = None,
    orientation: str = "vertical", lags_ms: float = 30.0, permutations: int = 500, alpha: float = 1.0,
    use_induced: bool = True, condition_on_C: bool = True, include_PT_cov: bool = False,
    rng_seed: int = 0
) -> FlowResult:
    if axes_dir is None: axes_dir = os.path.join(out_root, sid)
    cA = load_area_cache(out_root, sid, areaA); cB = load_area_cache(out_root, sid, areaB)
    oriA = cA["orientation"].astype(str); oriB = cB["orientation"].astype(str)
    corrA = cA["is_correct"];            corrB = cB["is_correct"]
    C_A = cA["C"].astype(np.float64);    C_B = cB["C"].astype(np.float64)

    ok = (~np.isnan(C_A)) & (~np.isnan(C_B)) & (oriA == orientation) & (oriB == orientation)
    if corrA is not None: ok &= corrA
    if corrB is not None: ok &= corrB

    t = cA["time"].astype(np.float64); assert np.allclose(t, cB["time"])
    bin_s = float(np.median(np.diff(t))); lags_bins = max(1, int(round(lags_ms/1000.0 / bin_s)))

    sA = _load_axis_vec(axes_dir, areaA, prefer="sS_inv")
    sB = _load_axis_vec(axes_dir, areaB, prefer="sS_inv")
    A_sig = _project(cA, sA, ok); B_sig = _project(cB, sB, ok)

    C = C_A[ok].astype(np.float64)
    PT = cA["PT_ms"][ok].astype(np.float64) if ("PT_ms" in cA) else None

    out = compute_sflow_timecourse(A_sig, B_sig, t, C, PT, lags_bins, alpha, permutations,
                                   use_induced, condition_on_C, include_PT_cov, rng_seed)

    meta = dict(sid=sid, areaA=areaA, areaB=areaB, orientation=orientation, lags_ms=lags_ms,
                lags_bins=lags_bins, permutations=permutations, alpha=alpha,
                induced=use_induced, condition_on_C=condition_on_C,
                include_PT_cov=include_PT_cov, bin_s=bin_s)
    return FlowResult(
        time=out["time"],
        bits_AtoB=out["bits_AtoB"], bits_BtoA=out["bits_BtoA"],
        null_mean_AtoB=out["null_mean_AtoB"], null_std_AtoB=out["null_std_AtoB"], p_AtoB=out["p_AtoB"],
        null_mean_BtoA=out["null_mean_BtoA"], null_std_BtoA=out["null_std_BtoA"], p_BtoA=out["p_BtoA"],
        meta=meta
    )

def save_flow_npz(fr: FlowResult, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        time=fr.time.astype(np.float32),
        bits_AtoB=fr.bits_AtoB.astype(np.float32),
        bits_BtoA=fr.bits_BtoA.astype(np.float32),
        null_mean_AtoB=fr.null_mean_AtoB.astype(np.float32),
        null_std_AtoB=fr.null_std_AtoB.astype(np.float32),
        null_mean_BtoA=fr.null_mean_BtoA.astype(np.float32),
        null_std_BtoA=fr.null_std_BtoA.astype(np.float32),
        p_AtoB=fr.p_AtoB.astype(np.float32),
        p_BtoA=fr.p_BtoA.astype(np.float32),
        meta=json.dumps(fr.meta)
    )

# ---------- paired-null pair-diff (no bands) ----------

def compute_sflow_pairdiff_for_pair(
    out_root: str, sid: str, areaA: str, areaB: str, axes_dir: Optional[str] = None,
    orientation: str = "vertical", lags_ms: float = 30.0, permutations: int = 500, alpha: float = 1.0,
    use_induced: bool = True, condition_on_C: bool = True, include_PT_cov: bool = False,
    rng_seed: int = 0
) -> PairDiffResult:
    if axes_dir is None: axes_dir = os.path.join(out_root, sid)
    cA = load_area_cache(out_root, sid, areaA); cB = load_area_cache(out_root, sid, areaB)
    oriA = cA["orientation"].astype(str); oriB = cB["orientation"].astype(str)
    corrA = cA["is_correct"];            corrB = cB["is_correct"]
    C_A = cA["C"].astype(np.float64);    C_B = cB["C"].astype(np.float64)

    ok = (~np.isnan(C_A)) & (~np.isnan(C_B)) & (oriA == orientation) & (oriB == orientation)
    if corrA is not None: ok &= corrA
    if corrB is not None: ok &= corrB

    t = cA["time"].astype(np.float64); assert np.allclose(t, cB["time"])
    bin_s = float(np.median(np.diff(t))); lags_bins = max(1, int(round(lags_ms/1000.0 / bin_s)))

    sA = _load_axis_vec(axes_dir, areaA, prefer="sS_inv"); sB = _load_axis_vec(axes_dir, areaB, prefer="sS_inv")
    A_sig = _project(cA, sA, ok); B_sig = _project(cB, sB, ok)

    C = C_A[ok].astype(np.float64)
    PT = cA["PT_ms"][ok].astype(np.float64) if ("PT_ms" in cA) else None

    covs = [np.ones((A_sig.shape[0],1))]
    if condition_on_C: covs.append(C.reshape(-1,1))
    if include_PT_cov and PT is not None:
        zPT = (PT - np.nanmean(PT)) / (np.nanstd(PT) + 1e-9)
        covs.append(zPT.reshape(-1,1))
    Xcov = np.column_stack(covs) if len(covs) else None

    A0, B0 = ( _induce_by_group(A_sig, C), _induce_by_group(B_sig, C) ) if use_induced else (A_sig, B_sig)

    t_idx = np.arange(lags_bins, A0.shape[1])
    T = len(t_idx)

    rng = np.random.default_rng(rng_seed)

    def flow_bits(y, src, dst, t):
        Xf, Xr, y_t, _ = _build_design(y=dst[:,t], A=src, B=dst, t=t, L=lags_bins, covars=Xcov)
        sse_f = _ridge_sse(y_t, Xf, alpha); sse_r = _ridge_sse(y_t, Xr, alpha)
        return _ll_bits_from_sse(sse_f, sse_r, y_t.shape[0])

    # observed DIFF
    diff_obs = np.zeros(T)
    for j, tix in enumerate(t_idx):
        ab = flow_bits(y=B0[:,tix], src=A0, dst=B0, t=tix)
        ba = flow_bits(y=A0[:,tix], src=B0, dst=A0, t=tix)
        diff_obs[j] = ab - ba

    # paired null: same row permutation for BOTH A and B each draw
    diff_null = np.zeros((permutations, T))
    for p in range(permutations):
        idx = rng.permutation(A0.shape[0])
        Aperm = _permute_rows(A0, idx)
        Bperm = _permute_rows(B0, idx)
        for j, tix in enumerate(t_idx):
            ab = flow_bits(y=B0[:,tix], src=Aperm, dst=B0, t=tix)
            ba = flow_bits(y=A0[:,tix], src=Bperm, dst=A0, t=tix)
            diff_null[p, j] = ab - ba

    mu = diff_null.mean(axis=0)
    sd = diff_null.std(axis=0, ddof=1); sd[sd==0] = 1.0
    z  = (diff_obs - mu) / sd
    p  = (np.sum(np.abs(diff_null - mu) >= np.abs(diff_obs - mu), axis=0) + 1.0) / (permutations + 1.0)

    meta = dict(sid=sid, areaA=areaA, areaB=areaB, orientation=orientation, lags_ms=lags_ms,
                permutations=permutations, alpha=alpha, induced=use_induced,
                condition_on_C=condition_on_C, include_PT_cov=include_PT_cov, bin_s=bin_s)

    return PairDiffResult(time=t[t_idx], diff_bits=diff_obs, null_mean_diff=mu, null_std_diff=sd,
                          p_diff=p, z_diff=z, meta=meta)

def save_flow_npz(fr: FlowResult, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        time=fr.time.astype(np.float32),
        bits_AtoB=fr.bits_AtoB.astype(np.float32),
        bits_BtoA=fr.bits_BtoA.astype(np.float32),
        null_mean_AtoB=fr.null_mean_AtoB.astype(np.float32),
        null_std_AtoB=fr.null_std_AtoB.astype(np.float32),
        null_mean_BtoA=fr.null_mean_BtoA.astype(np.float32),
        null_std_BtoA=fr.null_std_BtoA.astype(np.float32),
        p_AtoB=fr.p_AtoB.astype(np.float32),
        p_BtoA=fr.p_BtoA.astype(np.float32),
        meta=json.dumps(fr.meta)
    )

def save_pairdiff_npz(pdres: PairDiffResult, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        time=pdres.time.astype(np.float32),
        diff_bits=pdres.diff_bits.astype(np.float32),
        null_mean_diff=pdres.null_mean_diff.astype(np.float32),
        null_std_diff=pdres.null_std_diff.astype(np.float32),
        p_diff=pdres.p_diff.astype(np.float32),
        z_diff=pdres.z_diff.astype(np.float32),
        meta=json.dumps(pdres.meta)
    )
