# paper_project/paperflow/pseudo.py
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple, Optional
import numpy as np

def load_area_cache(out_root: str, align: str, sid: str, area: str) -> Dict:
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    return cache

def list_sessions_with_area(out_root: str, align: str, area: str) -> List[str]:
    root = os.path.join(out_root, align)
    if not os.path.isdir(root): return []
    sids = [s for s in os.listdir(root) if s.isdigit() and os.path.isdir(os.path.join(root, s))]
    keep = []
    for sid in sorted(sids):
        p = os.path.join(root, sid, "caches", f"area_{area}.npz")
        if os.path.exists(p): keep.append(sid)
    return keep

def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0: return np.array([1.0], dtype=float)
    half = int(np.ceil(3.0 * sigma_bins))
    x = np.arange(-half, half+1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins)**2)
    k /= k.sum()
    return k

def _smooth_time(X: np.ndarray, sigma_bins: float) -> np.ndarray:
    """
    X: (trials, bins, units), smooth along 'bins' with Gaussian kernel (same for all trials/units).
    """
    if sigma_bins <= 0: return X
    k = _gaussian_kernel(sigma_bins)  # (L,)
    T, B, U = X.shape
    Y = np.empty_like(X)
    # vectorized convolution along axis=1 using FFT would be faster; here keep it simple and robust
    for u in range(U):
        Y[:,:,u] = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, X[:,:,u])
    return Y

def _mask_trials(cache: Dict, orientation: Optional[str], pt_min_ms: Optional[float],
                 saccade_match: Optional[int]) -> np.ndarray:
    """
    Return a boolean mask of trials satisfying:
      - correct,
      - orientation (vertical/horizontal) if requested,
      - PT > pt_min_ms if provided,
      - saccade_location_sign == saccade_match if provided (+1 or -1).
    """
    N = cache["X"].shape[0]
    ok = np.ones(N, dtype=bool)
    if "lab_is_correct" in cache:
        ok &= cache["lab_is_correct"].astype(bool)
    if orientation is not None and "lab_orientation" in cache:
        ok &= (cache["lab_orientation"].astype(str) == orientation)
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        ok &= np.isfinite(PT) & (PT > float(pt_min_ms))
    if (saccade_match is not None) and ("lab_S" in cache or "lab_saccade_location_sign" in cache):
        S = cache.get("lab_S", cache.get("lab_saccade_location_sign")).astype(float)
        ok &= (np.sign(S) == (1 if saccade_match > 0 else -1))
    return ok

def _within_cat_label_map(R: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, Dict[float,int]]:
    rvals = np.unique(R[m])
    if rvals.size < 3:
        return np.array([], dtype=int), {}
    rvals = np.sort(rvals)[:3]
    rmap = {float(v): i for i, v in enumerate(rvals)}
    y = np.array([rmap.get(float(v), -1) for v in R[m]], dtype=int)
    good = (y >= 0)
    return y[good], rmap

def _unit_pools_for_category(C: np.ndarray, R: np.ndarray, ok: np.ndarray, Csign: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    m = ok & np.isfinite(C) & np.isfinite(R) & (np.sign(C) == (1 if Csign > 0 else -1))
    if not np.any(m): return np.array([], dtype=int), [np.array([], dtype=int)]*3
    y, rmap = _within_cat_label_map(R, m)
    if y.size == 0: return np.array([], dtype=int), [np.array([], dtype=int)]*3
    idx_all = np.where(m)[0]
    idx_good = idx_all[y >= 0]
    y = y[y >= 0]
    pools = [idx_good[y == k] for k in (0,1,2)]
    if any(p.size == 0 for p in pools):
        return np.array([], dtype=int), [np.array([], dtype=int)]*3
    return idx_good, pools

def _min_rate_hz(X_u: np.ndarray, bin_s: float, win: Tuple[float,float]) -> float:
    """Return max( trial-avg rate ) over time within window (sec)."""
    B = X_u.shape[1]
    t0, t1 = win
    centers = None  # not needed; assume caller set win in bin indices logic if necessary
    # approximate by averaging counts in window and dividing by bin_s
    return float(np.nanmax(X_u[:, int(np.ceil((t0)/bin_s)): int(np.floor((t1)/bin_s))].mean(axis=0) / bin_s))

def _dir_selective_score(X_u: np.ndarray, C: np.ndarray, R: np.ndarray,
                         ok: np.ndarray, time_bins: np.ndarray,
                         min_per_label: int = 8) -> float:
    """
    Simple direction selectivity score within category:
      For each category, one-vs-rest linear decode (3-way folded into 3 one-vs-rest) at this time bin,
      take mean of AUCs (here approximated with balanced accuracy).
    Returns max score over categories at that time (0..1).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    best = 0.0
    b = int(time_bins)  # a bin index
    Xu = X_u[:, b]  # (trials,)
    for cs in (+1, -1):
        m = ok & (np.sign(C) == (1 if cs>0 else -1)) & np.isfinite(R)
        if m.sum() < (3*min_per_label): continue
        rvals = np.unique(R[m])
        if rvals.size < 3: continue
        rvals = np.sort(rvals)[:3]
        rmap = {float(v): i for i, v in enumerate(rvals)}
        y = np.array([rmap.get(float(v), -1) for v in R[m]], dtype=int)
        good = (y >= 0); y = y[good]; x = Xu[m][good].reshape(-1,1)
        # fold 3-way into three one-vs-rest, compute mean balanced accuracy
        accs=[]
        for k in (0,1,2):
            yy = (y==k).astype(int)
            if yy.sum() < min_per_label or (1-yy).sum() < min_per_label: continue
            try:
                clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                         class_weight="balanced", max_iter=1000)
                clf.fit(x, yy)
                pred = clf.predict(x)
                accs.append(balanced_accuracy_score(yy, pred))
            except Exception:
                pass
        if accs:
            best = max(best, float(np.mean(accs)))
    return best

def gather_units_for_area(out_root: str,
                          align: str,
                          area: str,
                          orientation: str = "vertical",
                          pt_min_ms: float = 200.0,
                          max_units: Optional[int] = None,
                          selective_only: bool = False,
                          select_win: Tuple[float,float] = (0.08, 0.20),
                          min_rate_hz: float = 10.0,
                          smooth_sigma_ms: float = 20.0,
                          match_saccade: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load all sessions with this area, filter trials, optionally smooth in time,
    and build per-unit pools. If selective_only, keep only units passing a simple
    direction selectivity + min-rate test within select_win.
    """
    sids = list_sessions_with_area(out_root, align, area)
    if not sids: raise RuntimeError(f"No sessions with area {area} under {out_root}/{align}")
    time_ref = None
    units: List[Dict] = []

    for sid in sids:
        cache = load_area_cache(out_root, align, sid, area)
        X = cache["X"].astype(np.float32)   # (trials, bins, units)
        C = cache["lab_C"].astype(float)
        R = cache["lab_R"].astype(float)
        bin_s = float(cache["meta"].get("bin_s", 0.010))
        # optional smoothing and convert to rates (Hz)
        sigma_bins = float(smooth_sigma_ms)/1000.0 / bin_s if smooth_sigma_ms>0 else 0.0
        Xs = _smooth_time(X, sigma_bins) / bin_s

        ok = _mask_trials(cache, orientation=orientation, pt_min_ms=pt_min_ms, saccade_match=match_saccade)

        if time_ref is None:
            time_ref = cache["time"].astype(float)
        else:
            t = cache["time"].astype(float)
            if t.shape != time_ref.shape or not np.allclose(t, time_ref):
                continue

        T, B, U = Xs.shape
        # convert selection window into a representative bin index
        # pick the center of the window
        sel_center = (select_win[0] + select_win[1]) / 2.0
        sel_bin = int(np.clip(round(sel_center / bin_s), 0, B-1))

        for u in range(U):
            Xu = Xs[:,:,u]  # (trials, bins)
            # direction-selective filter (approximate)
            if selective_only:
                # min rate threshold (peak over time within select_win)
                if _min_rate_hz(Xu, bin_s, select_win) < min_rate_hz:
                    continue
                scr = _dir_selective_score(Xu, C, R, ok, sel_bin)
                if scr < 0.6:  # rough threshold
                    continue

            # pools per category sign (+1/-1)
            pools = {}
            for cs in (+1, -1):
                idx_cat, pools3 = _unit_pools_for_category(C, R, ok, cs)
                pools[cs] = pools3
            if not selective_only:
                # accept if any category has all 3 pools non-empty
                accept = any(all(p.size > 0 for p in pools[cs]) for cs in (+1,-1))
            else:
                # for selected units, require both categories when possible
                accept = any(all(p.size > 0 for p in pools[cs]) for cs in (+1,-1))
            if not accept: continue

            units.append(dict(sid=sid, X_u=Xu, pools=pools))

    if not units:
        raise RuntimeError(f"No units passed filters for {area} ({orientation}, PT>{pt_min_ms} ms, selective={selective_only}).")

    if max_units is not None and len(units) > max_units:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(units), size=max_units, replace=False)
        units = [units[i] for i in pick]

    return time_ref, units

def _sample_indices(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if pool.size >= n:
        return rng.choice(pool, size=n, replace=False)
    return rng.choice(pool, size=n, replace=True)

def decode_direction_within_cat_pseudopop(time_s: np.ndarray,
                                          units: List[Dict],
                                          Csign: int,
                                          n_pops: int,
                                          n_train: int,
                                          n_test: int,
                                          seed: int = 0,
                                          progress: bool = True) -> np.ndarray:
    from sklearn.svm import LinearSVC
    rng = np.random.default_rng(seed)
    B = time_s.size
    accs = np.full((n_pops, B), np.nan, dtype=float)

    # keep only units that have all 3 label pools non-empty for this category
    use_units = [u for u in units if all(p.size > 0 for p in u["pools"][Csign])]
    Usel = len(use_units)
    if Usel == 0: return accs

    for p in range(n_pops):
        if progress and (p % 10 == 0):
            print(f"  [decode] pop {p+1}/{n_pops} (Csign={'+1' if Csign>0 else '-1'})")
        clf = LinearSVC(C=1.0, class_weight="balanced", dual=True, max_iter=2000)

        # draw per-unit, per-label train/test index pools
        idx_tr_per_unit = {k: [None]*Usel for k in (0,1,2)}
        idx_te_per_unit = {k: [None]*Usel for k in (0,1,2)}
        for ui, u in enumerate(use_units):
            pools = u["pools"][Csign]
            for k in (0,1,2):
                all_idx = pools[k]
                tr = _sample_indices(all_idx, n_train, rng)
                remain = np.setdiff1d(all_idx, tr)
                te = _sample_indices(remain if remain.size>0 else all_idx, n_test, rng)
                idx_tr_per_unit[k][ui] = tr
                idx_te_per_unit[k][ui] = te

        for b in range(B):
            Xtr=[]; ytr=[]; Xte=[]; yte=[]
            for k in (0,1,2):
                # training samples
                for _ in range(n_train):
                    x = np.empty(Usel, dtype=float)
                    for ui, u in enumerate(use_units):
                        ti = int(rng.choice(idx_tr_per_unit[k][ui]))
                        x[ui] = u["X_u"][ti, b]
                    Xtr.append(x); ytr.append(k)
                # testing samples
                for _ in range(n_test):
                    x = np.empty(Usel, dtype=float)
                    for ui, u in enumerate(use_units):
                        ti = int(rng.choice(idx_te_per_unit[k][ui]))
                        x[ui] = u["X_u"][ti, b]
                    Xte.append(x); yte.append(k)
            if not Xtr: continue
            Xtr = np.vstack(Xtr); ytr = np.array(ytr, dtype=int)
            Xte = np.vstack(Xte); yte = np.array(yte, dtype=int)
            try:
                clf.fit(Xtr, ytr)
                accs[p, b] = float((clf.predict(Xte) == yte).mean())
            except Exception:
                accs[p, b] = np.nan

    return accs

def pseudo_decode_direction_within_category(
    out_root: str,
    align: str,
    area: str,
    orientation: str = "vertical",
    n_pops: int = 200,
    n_train: int = 10,
    n_test: int = 2,
    pt_min_ms: float = 200.0,
    max_units: Optional[int] = None,
    seed: int = 0,
    selective_only: bool = False,
    select_win: Tuple[float,float] = (0.08, 0.20),
    min_rate_hz: float = 10.0,
    smooth_sigma_ms: float = 20.0,
    match_saccade: Optional[int] = None   # +1 or -1 to fix saccade direction; None to ignore
) -> Dict[str, np.ndarray]:
    """
    Paper-style pseudopop direction decoder with options to replicate Methods:
      - correct, vertical, PT>pt_min_ms
      - selective_only: keep only direction-selective units (approx),
      - smooth_sigma_ms: time smoothing (20 ms recommended),
      - match_saccade: fix saccade direction (+1 or -1) within category,
      - area unit cap via max_units,
      - choose better category (C=+1 or C=-1), report mean ± SD.
    """
    assert align == "stim", "direction pseudopop should be run in stim alignment"

    time_s, units = gather_units_for_area(
        out_root, align, area, orientation=orientation, pt_min_ms=pt_min_ms,
        max_units=max_units, selective_only=selective_only, select_win=select_win,
        min_rate_hz=min_rate_hz, smooth_sigma_ms=smooth_sigma_ms,
        match_saccade=match_saccade
    )

    pos = decode_direction_within_cat_pseudopop(time_s, units, +1, n_pops, n_train, n_test, seed=seed,   progress=True)
    neg = decode_direction_within_cat_pseudopop(time_s, units, -1, n_pops, n_train, n_test, seed=seed+1, progress=True)

    mean_pos = np.nanmean(pos, axis=0); std_pos = np.nanstd(pos, axis=0, ddof=1)
    mean_neg = np.nanmean(neg, axis=0); std_neg = np.nanstd(neg, axis=0, ddof=1)
    choose_pos = (np.nanmean(mean_pos) >= np.nanmean(mean_neg))
    acc_mean = mean_pos if choose_pos else mean_neg
    acc_std  = std_pos  if choose_pos else std_neg

    return dict(
        time=time_s, acc_mean=acc_mean, acc_std=acc_std,
        meta=dict(align=align, area=area, orientation=orientation,
                  n_pops=int(n_pops), n_train=int(n_train), n_test=int(n_test),
                  pt_min_ms=float(pt_min_ms), max_units=(int(max_units) if max_units else None),
                  choose_category=("C=+1" if choose_pos else "C=-1"),
                  n_units=int(len(units)),
                  selective_only=bool(selective_only),
                  select_win=list(select_win),
                  min_rate_hz=float(min_rate_hz),
                  smooth_sigma_ms=float(smooth_sigma_ms),
                  match_saccade=(int(match_saccade) if match_saccade is not None else None))
    )
