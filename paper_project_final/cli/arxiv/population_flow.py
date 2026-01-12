#!/usr/bin/env python
"""
Population-level flow analysis.

Philosophy:
- Compute within-session effect (observed - null_mean) 
- Aggregate effects across sessions
- Test if population effect is consistently > 0

The within-session null doesn't need to be exactly zero.
It just needs to remove bias consistently.
The population-level test asks: "Is there a consistent effect across sessions?"

This is more powerful and scientifically meaningful than session-by-session significance testing.
"""

import numpy as np
import json
import os
import sys
from glob import glob
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------- helpers --------------------

def _label_vec(cache: Dict, key: str, mask: np.ndarray) -> Optional[np.ndarray]:
    v = cache.get(key, None)
    if v is None:
        return None
    return np.asarray(v).astype(float)[mask]

def _encode_joint_labels(cache: Dict, mask: np.ndarray, keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    vecs = []
    for k in keys:
        v = _label_vec(cache, k, mask)
        if v is None:
            return None
        vecs.append(np.asarray(np.round(v), dtype=int))
    base = 1000
    code = np.zeros_like(vecs[0], dtype=int)
    for i, v in enumerate(reversed(vecs)):
        code += (base**i) * v
    return code.astype(float)

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
    else:
        a = np.array([])
    if a.size == 0:
        return None
    a = np.array(a)
    if a.ndim == 1:
        a = a[:, None]
    return a

def _project_multi(cache: Dict, A: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Z = cache["Z"][mask].astype(np.float64)
    return np.tensordot(Z, A, axes=([2], [0]))

def _induce_by_strata_multi(Y: np.ndarray, labels: Optional[np.ndarray]) -> np.ndarray:
    if labels is None:
        return Y
    out = Y.copy()
    lab = labels.copy()
    good = np.isfinite(lab)
    for u in np.unique(lab[good]):
        m = good & (lab == u)
        if np.any(m):
            out[m] -= out[m].mean(axis=0, keepdims=True)
    return out

def _build_lag_stack_multi(Y: np.ndarray, L: int) -> np.ndarray:
    N, B, K = Y.shape
    if L <= 0:
        return np.zeros((N, B, 0), dtype=float)
    Xlags = np.zeros((N, B, L * K), dtype=float)
    for l in range(1, L + 1):
        Xlags[:, l:, (l - 1) * K: l * K] = Y[:, :-l, :]
    return Xlags

def _circular_shift_time(Y: np.ndarray, rng: np.random.Generator, min_shift: int = 1) -> np.ndarray:
    N, B, K = Y.shape
    Y_shifted = np.empty_like(Y)
    for n in range(N):
        shift = rng.integers(min_shift, max(min_shift + 1, B))
        Y_shifted[n] = np.roll(Y[n], shift, axis=0)
    return Y_shifted

def _ridge_sse(Y: np.ndarray, X: np.ndarray, ridge: float) -> float:
    """Simple training SSE with ridge regression."""
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    XtY = X.T @ Y
    try:
        B = np.linalg.solve(XtX, XtY)
        R = Y - X @ B
        return float(np.sum(R * R))
    except:
        return np.nan


# -------------------- single session flow --------------------

def compute_session_flow(
    cacheA: Dict, cacheB: Dict,
    axesA: Dict, axesB: Dict,
    feature: str,
    orientation: Optional[str],
    pt_min_ms: Optional[float],
    lags_ms: float,
    ridge: float,
    n_null: int = 100,  # Fewer perms needed for effect estimation
    induced: bool = True,
    include_B_lags: bool = True,
    seed: int = 0,
) -> Dict:
    """
    Compute flow for a single session.
    
    Returns:
        observed: array of observed flow per time bin
        null_mean: array of null mean per time bin  
        effect: observed - null_mean (the key quantity!)
        time: time array
    """
    
    # Mask trials
    maskA = _trial_mask(cacheA, orientation, pt_min_ms)
    maskB = _trial_mask(cacheB, orientation, pt_min_ms)
    mask = maskA & maskB
    if mask.sum() < 30:
        return None  # Not enough trials
    
    # Time grid
    time = cacheA["time"].astype(float)
    metaA = cacheA["meta"]
    if not isinstance(metaA, dict):
        metaA = json.loads(metaA.item() if hasattr(metaA, "item") else str(metaA))
    bin_s = float(metaA.get("bin_s", (time[1] - time[0] if time.size > 1 else 0.01)))
    
    Bbins = time.size
    L = int(max(1, round(lags_ms / (bin_s * 1000.0))))
    start_b = L
    rng = np.random.default_rng(seed)
    
    # Axes
    Aaxis = _axis_matrix(axesA, feature)
    Baxis = _axis_matrix(axesB, feature)
    if Aaxis is None or Baxis is None:
        return None
    
    # Project
    YA_full = _project_multi(cacheA, Aaxis, mask)
    YB_full = _project_multi(cacheB, Baxis, mask)
    
    # Induced removal
    labs_induce = None
    if induced:
        if feature == "C":
            labs_induce = _label_vec(cacheA, "lab_R", mask)
        elif feature == "R":
            labs_induce = _label_vec(cacheA, "lab_C", mask)
        elif feature == "S":
            labs_induce = _encode_joint_labels(cacheA, mask, ("lab_C", "lab_R"))
    
    YA = _induce_by_strata_multi(YA_full, labs_induce) if induced else YA_full
    YB = _induce_by_strata_multi(YB_full, labs_induce) if induced else YB_full
    
    # Lag stacks
    XA_lags = _build_lag_stack_multi(YA, L)
    XB_lags = _build_lag_stack_multi(YB, L)
    
    N = YA.shape[0]
    ones = np.ones((N, 1), dtype=float)
    
    # Compute observed flow
    obs_AtoB = np.full(Bbins, np.nan)
    obs_BtoA = np.full(Bbins, np.nan)
    null_AtoB = np.full((n_null, Bbins), np.nan)
    null_BtoA = np.full((n_null, Bbins), np.nan)
    
    for b in range(start_b, Bbins):
        # A → B
        Yt = YB[:, b, :]
        X_red = ones.copy()
        if include_B_lags and XB_lags.shape[2] > 0:
            X_red = np.concatenate([X_red, XB_lags[:, b, :]], axis=1)
        X_full = np.concatenate([X_red, XA_lags[:, b, :]], axis=1)
        
        sse_red = _ridge_sse(Yt, X_red, ridge)
        sse_full = _ridge_sse(Yt, X_full, ridge)
        if sse_red > 0 and sse_full > 0:
            obs_AtoB[b] = 0.5 * N * np.log2(sse_red / sse_full)
        
        # B → A
        Yt_rev = YA[:, b, :]
        X_red_r = ones.copy()
        if include_B_lags and XA_lags.shape[2] > 0:
            X_red_r = np.concatenate([X_red_r, XA_lags[:, b, :]], axis=1)
        X_full_r = np.concatenate([X_red_r, XB_lags[:, b, :]], axis=1)
        
        sse_red_r = _ridge_sse(Yt_rev, X_red_r, ridge)
        sse_full_r = _ridge_sse(Yt_rev, X_full_r, ridge)
        if sse_red_r > 0 and sse_full_r > 0:
            obs_BtoA[b] = 0.5 * N * np.log2(sse_red_r / sse_full_r)
        
        # Null (circular shift)
        for p in range(n_null):
            YA_null = _circular_shift_time(YA, rng, min_shift=L+1)
            YB_null = _circular_shift_time(YB, rng, min_shift=L+1)
            
            XA_lags_null = _build_lag_stack_multi(YA_null, L)
            XB_lags_null = _build_lag_stack_multi(YB_null, L)
            
            # A → B null
            X_full_null = np.concatenate([X_red, XA_lags_null[:, b, :]], axis=1)
            sse_full_null = _ridge_sse(Yt, X_full_null, ridge)
            if sse_red > 0 and sse_full_null > 0:
                null_AtoB[p, b] = 0.5 * N * np.log2(sse_red / sse_full_null)
            
            # B → A null
            X_full_null_r = np.concatenate([X_red_r, XB_lags_null[:, b, :]], axis=1)
            sse_full_null_r = _ridge_sse(Yt_rev, X_full_null_r, ridge)
            if sse_red_r > 0 and sse_full_null_r > 0:
                null_BtoA[p, b] = 0.5 * N * np.log2(sse_red_r / sse_full_null_r)
    
    # Compute effect = observed - null_mean
    null_mean_AtoB = np.nanmean(null_AtoB, axis=0)
    null_mean_BtoA = np.nanmean(null_BtoA, axis=0)
    
    effect_AtoB = obs_AtoB - null_mean_AtoB
    effect_BtoA = obs_BtoA - null_mean_BtoA
    
    return dict(
        time=time,
        observed_AtoB=obs_AtoB,
        observed_BtoA=obs_BtoA,
        null_mean_AtoB=null_mean_AtoB,
        null_mean_BtoA=null_mean_BtoA,
        effect_AtoB=effect_AtoB,  # THE KEY QUANTITY
        effect_BtoA=effect_BtoA,
        n_trials=int(mask.sum()),
    )


# -------------------- population aggregation --------------------

def aggregate_sessions(session_results: List[Dict], direction: str = "AtoB") -> Dict:
    """
    Aggregate effects across sessions.
    
    For each time point, compute:
    - Mean effect across sessions
    - SEM across sessions
    - t-statistic and p-value (is effect > 0?)
    """
    
    if not session_results:
        return None
    
    # Get common time grid (assume all sessions have same time)
    time = session_results[0]['time']
    n_bins = len(time)
    n_sessions = len(session_results)
    
    # Stack effects: (n_sessions, n_bins)
    effects = np.full((n_sessions, n_bins), np.nan)
    for i, res in enumerate(session_results):
        key = f'effect_{direction}'
        effects[i, :] = res[key]
    
    # Compute population statistics
    mean_effect = np.nanmean(effects, axis=0)
    std_effect = np.nanstd(effects, axis=0, ddof=1)
    n_valid = np.sum(np.isfinite(effects), axis=0)
    sem_effect = std_effect / np.sqrt(n_valid)
    
    # T-test at each time point: is effect > 0?
    t_values = np.full(n_bins, np.nan)
    p_values = np.full(n_bins, np.nan)
    
    for b in range(n_bins):
        eff_b = effects[:, b]
        valid = np.isfinite(eff_b)
        if valid.sum() >= 3:
            t, p = stats.ttest_1samp(eff_b[valid], 0)
            t_values[b] = t
            p_values[b] = p / 2  # One-sided (effect > 0)
            if t < 0:
                p_values[b] = 1 - p_values[b]
    
    return dict(
        time=time,
        mean_effect=mean_effect,
        sem_effect=sem_effect,
        std_effect=std_effect,
        t_values=t_values,
        p_values=p_values,
        n_sessions=n_sessions,
        effects_all=effects,  # Keep individual session effects
    )


# -------------------- main --------------------

def load_cache(out_root: str, align: str, sid: str, area: str):
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    return cache

def load_axes(out_root: str, align: str, sid: str, area: str, tag: str = None):
    if tag:
        path = os.path.join(out_root, align, sid, "axes", tag, f"axes_{area}.npz")
    else:
        path = os.path.join(out_root, align, sid, "axes", f"axes_{area}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)

def find_sessions(out_root: str, align: str, areaA: str, areaB: str, tag: str = None) -> List[str]:
    """Find sessions that have both areas."""
    base = os.path.join(out_root, align)
    if not os.path.isdir(base):
        return []
    
    sessions = []
    for sid in sorted(os.listdir(base)):
        if not sid.isdigit():
            continue
        
        cacheA = os.path.join(base, sid, "caches", f"area_{areaA}.npz")
        cacheB = os.path.join(base, sid, "caches", f"area_{areaB}.npz")
        
        if tag:
            axesA = os.path.join(base, sid, "axes", tag, f"axes_{areaA}.npz")
            axesB = os.path.join(base, sid, "axes", tag, f"axes_{areaB}.npz")
        else:
            axesA = os.path.join(base, sid, "axes", f"axes_{areaA}.npz")
            axesB = os.path.join(base, sid, "axes", f"axes_{areaB}.npz")
        
        if all(os.path.exists(p) for p in [cacheA, cacheB, axesA, axesB]):
            sessions.append(sid)
    
    return sessions


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Population-level flow analysis")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--align", default="stim")
    ap.add_argument("--areas", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--feature", default="C", choices=["C", "R", "S", "T"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--orientation", default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--lags_ms", type=float, default=50.0)
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--n_null", type=int, default=100, help="Null permutations per session")
    ap.add_argument("--sessions", nargs="*", default=None, help="Specific sessions (default: all)")
    args = ap.parse_args()
    
    areaA, areaB = args.areas
    
    # Find sessions
    if args.sessions:
        sessions = args.sessions
    else:
        sessions = find_sessions(args.out_root, args.align, areaA, areaB, args.tag)
    
    print(f"Found {len(sessions)} sessions with {areaA} and {areaB}")
    if not sessions:
        print("No sessions found!")
        return
    
    # Process each session
    session_results = []
    for sid in sessions:
        print(f"  Processing {sid}...", end=" ")
        
        cacheA = load_cache(args.out_root, args.align, sid, areaA)
        cacheB = load_cache(args.out_root, args.align, sid, areaB)
        axesA = load_axes(args.out_root, args.align, sid, areaA, args.tag)
        axesB = load_axes(args.out_root, args.align, sid, areaB, args.tag)
        
        if any(x is None for x in [cacheA, cacheB, axesA, axesB]):
            print("skipped (missing data)")
            continue
        
        result = compute_session_flow(
            cacheA, cacheB, axesA, axesB,
            feature=args.feature,
            orientation=args.orientation,
            pt_min_ms=args.pt_min_ms,
            lags_ms=args.lags_ms,
            ridge=args.ridge,
            n_null=args.n_null,
            seed=hash(sid) % 10000,
        )
        
        if result is None:
            print("skipped (not enough trials)")
            continue
        
        result['sid'] = sid
        session_results.append(result)
        print(f"done (N={result['n_trials']})")
    
    print(f"\nSuccessfully processed {len(session_results)} sessions")
    
    if len(session_results) < 2:
        print("Need at least 2 sessions for population analysis!")
        return
    
    # Aggregate across sessions
    pop_AtoB = aggregate_sessions(session_results, "AtoB")
    pop_BtoA = aggregate_sessions(session_results, "BtoA")
    
    time = pop_AtoB['time'] * 1000  # ms
    
    # Plot
    fig, axes_arr = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Population mean effect with SEM (A → B)
    ax = axes_arr[0, 0]
    mean = pop_AtoB['mean_effect']
    sem = pop_AtoB['sem_effect']
    ax.fill_between(time, mean - 2*sem, mean + 2*sem, alpha=0.3, color='blue')
    ax.plot(time, mean, 'b-', lw=2, label=f'{areaA} → {areaB}')
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('Effect (bits above null)')
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{areaA} → {areaB}: Population Effect ± 2 SEM (N={pop_AtoB["n_sessions"]} sessions)')
    ax.legend()
    
    # Panel 2: Individual session effects (A → B)
    ax = axes_arr[0, 1]
    effects = pop_AtoB['effects_all']
    for i in range(effects.shape[0]):
        ax.plot(time, effects[i, :], 'b-', alpha=0.3, lw=0.5)
    ax.plot(time, mean, 'b-', lw=2, label='Mean')
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('Effect (bits)')
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{areaA} → {areaB}: Individual Sessions')
    
    # Panel 3: Both directions
    ax = axes_arr[1, 0]
    ax.fill_between(time, pop_AtoB['mean_effect'] - pop_AtoB['sem_effect'], 
                    pop_AtoB['mean_effect'] + pop_AtoB['sem_effect'], alpha=0.3, color='blue')
    ax.fill_between(time, pop_BtoA['mean_effect'] - pop_BtoA['sem_effect'], 
                    pop_BtoA['mean_effect'] + pop_BtoA['sem_effect'], alpha=0.3, color='red')
    ax.plot(time, pop_AtoB['mean_effect'], 'b-', lw=2, label=f'{areaA} → {areaB}')
    ax.plot(time, pop_BtoA['mean_effect'], 'r-', lw=2, label=f'{areaB} → {areaA}')
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('Effect (bits)')
    ax.set_xlabel('Time (ms)')
    ax.legend()
    ax.set_title('Bidirectional Flow')
    
    # Panel 4: P-values
    ax = axes_arr[1, 1]
    ax.semilogy(time, pop_AtoB['p_values'], 'b-', lw=1.5, label=f'{areaA}→{areaB}')
    ax.semilogy(time, pop_BtoA['p_values'], 'r-', lw=1.5, label=f'{areaB}→{areaA}')
    ax.axhline(0.05, color='gray', ls='--', lw=1, label='p=0.05')
    ax.axhline(0.01, color='gray', ls=':', lw=1, label='p=0.01')
    ax.axvline(0, color='k', ls=':', lw=0.5)
    ax.set_ylabel('P-value (t-test, one-sided)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylim([0.0001, 1])
    ax.legend(fontsize=8)
    ax.set_title('Population-Level Significance')
    
    plt.tight_layout()
    
    out_dir = os.path.join(args.out_root, "population_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pop_flow_{areaA}_{areaB}_{args.feature}_{args.align}.png")
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace('.png', '.pdf'))
    print(f"\nSaved figure to {out_path}")
    
    # Summary
    print("\n" + "="*70)
    print(f"POPULATION SUMMARY: {areaA} ↔ {areaB} ({args.feature})")
    print("="*70)
    
    # Find peak
    t_max_idx = np.nanargmax(pop_AtoB['mean_effect'])
    t_max = time[t_max_idx]
    
    print(f"\n{areaA} → {areaB}:")
    print(f"  Peak at t = {t_max:.0f} ms")
    print(f"  Mean effect: {pop_AtoB['mean_effect'][t_max_idx]:.2f} ± {pop_AtoB['sem_effect'][t_max_idx]:.2f} bits")
    print(f"  T-statistic: {pop_AtoB['t_values'][t_max_idx]:.2f}")
    print(f"  P-value: {pop_AtoB['p_values'][t_max_idx]:.4f}")
    
    t_max_idx_r = np.nanargmax(pop_BtoA['mean_effect'])
    t_max_r = time[t_max_idx_r]
    
    print(f"\n{areaB} → {areaA}:")
    print(f"  Peak at t = {t_max_r:.0f} ms")
    print(f"  Mean effect: {pop_BtoA['mean_effect'][t_max_idx_r]:.2f} ± {pop_BtoA['sem_effect'][t_max_idx_r]:.2f} bits")
    print(f"  T-statistic: {pop_BtoA['t_values'][t_max_idx_r]:.2f}")
    print(f"  P-value: {pop_BtoA['p_values'][t_max_idx_r]:.4f}")
    
    # Count significant time points
    n_bins = len(time)
    for direction, pop, name in [("AtoB", pop_AtoB, f"{areaA}→{areaB}"), 
                                  ("BtoA", pop_BtoA, f"{areaB}→{areaA}")]:
        p = pop['p_values']
        n_05 = np.sum(p < 0.05)
        n_01 = np.sum(p < 0.01)
        n_001 = np.sum(p < 0.001)
        print(f"\n{name} significant time bins:")
        print(f"  p < 0.05:  {n_05}/{n_bins}")
        print(f"  p < 0.01:  {n_01}/{n_bins}")
        print(f"  p < 0.001: {n_001}/{n_bins}")
    
    # Save results
    np.savez_compressed(
        out_path.replace('.png', '.npz'),
        time=time,
        pop_AtoB_mean=pop_AtoB['mean_effect'],
        pop_AtoB_sem=pop_AtoB['sem_effect'],
        pop_AtoB_p=pop_AtoB['p_values'],
        pop_AtoB_t=pop_AtoB['t_values'],
        pop_BtoA_mean=pop_BtoA['mean_effect'],
        pop_BtoA_sem=pop_BtoA['sem_effect'],
        pop_BtoA_p=pop_BtoA['p_values'],
        pop_BtoA_t=pop_BtoA['t_values'],
        sessions=sessions,
        effects_AtoB=pop_AtoB['effects_all'],
        effects_BtoA=pop_BtoA['effects_all'],
    )
    print(f"\nSaved data to {out_path.replace('.png', '.npz')}")


if __name__ == "__main__":
    main()
