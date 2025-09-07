#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Trial-wise Threshold Crossing (TTC) Latency Analysis
import argparse, os, json, numpy as np

def moving_average_1d(x, win):
    if win <= 1: return x
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode='same')

def softmax(logits, axis=-1):
    m = np.max(logits, axis=axis, keepdims=True)
    z = np.exp(logits - m)
    return z / np.sum(z, axis=axis, keepdims=True)

def fit_gaussian_lda_diag(X, y, n_classes=None, lam=1e-3):
    if n_classes is None:
        classes = np.unique(y); K = int(classes.max()) + 1
    else:
        K = n_classes; classes = np.arange(K)
    d = X.shape[1]
    mus = np.zeros((K, d), dtype=float)
    pri = np.zeros(K, dtype=float)
    var = np.zeros(d, dtype=float)
    n_total = float(X.shape[0])
    for k in classes:
        idx = np.where(y == k)[0]
        pri[k] = len(idx) / n_total if n_total > 0 else 0.0
        if len(idx) > 0: mus[k] = X[idx].mean(axis=0)
    for k in classes:
        idx = np.where(y == k)[0]
        if len(idx) > 1: var += pri[k] * X[idx].var(axis=0, ddof=1)
        elif len(idx) == 1: var += pri[k] * (np.ones(d) * 1e-2)
    var = var + lam
    return {"mu": mus, "var": var, "pri": pri}

def predict_proba_gaussian_lda_diag(model, X):
    mu = model["mu"]; var = model["var"]; pri = model["pri"]
    K, d = mu.shape
    const = -0.5 * (np.log(2.0 * np.pi) + np.log(var)).sum()
    inv_var = 1.0 / var
    X = np.asarray(X); n = X.shape[0]
    ll = np.zeros((n, K), dtype=float)
    for k in range(K):
        dx = X - mu[k]
        ll[:, k] = const - 0.5 * ((dx * dx) * inv_var).sum(axis=1) + np.log(max(pri[k], 1e-12))
    return softmax(ll, axis=1)

def earliest_crossing(p, t, theta, tmin, tmax, min_consecutive=1, smooth_bins=1):
    ps = moving_average_1d(p, smooth_bins)
    mask = (t >= tmin) & (t <= tmax)
    idxs = np.where(mask)[0]
    if idxs.size == 0: return np.nan
    above = ps[idxs] >= theta
    if not np.any(above): return np.nan
    count = 0
    for j, flag in enumerate(above):
        if flag: count += 1
        else: count = 0
        if count >= min_consecutive:
            start_idx = j - min_consecutive + 1
            return t[idxs[start_idx]]
    return np.nan

def find_key(npz, candidates):
    for k in candidates:
        if k in npz: return k
    return None

def load_axes(session_dir, area):
    path = os.path.join(session_dir, f"axes_{area}.npz")
    if not os.path.exists(path): raise FileNotFoundError(f"Missing {path}")
    npz = np.load(path, allow_pickle=True)
    key_t = find_key(npz, ["t","times","time"])
    t = npz[key_t] if key_t else None
    ZC = npz["ZC"] if "ZC" in npz else None
    ZR = npz["ZR"] if "ZR" in npz else None
    C = npz["C"] if "C" in npz else None
    R = npz["R"] if "R" in npz else None
    DIR = None
    if "R_full" in npz: DIR = npz["R_full"]
    elif "DIR" in npz: DIR = npz["DIR"]
    return {"path": path, "t": t, "ZC": ZC, "ZR": ZR, "C": C, "R": R, "DIR": DIR}

def try_load_labels_from_cache(results_root, sid, area_hint):
    cache_path = os.path.join(os.path.dirname(results_root), "caches", f"{sid}_{area_hint}.npz")
    if os.path.exists(cache_path):
        npz = np.load(cache_path, allow_pickle=True)
        C = npz["C"] if "C" in npz else None
        R = npz["R"] if "R" in npz else None
        DIR = npz["DIR"] if "DIR" in npz else (npz["direction"] if "direction" in npz else None)
        return C, R, DIR
    return None, None, None

def calibrate_threshold(p_true_matrix, t, alpha=0.01, min_theta=0.55):
    baseline_mask = t < 0.0
    if not np.any(baseline_mask):
        cutoff = float(t.min() + 0.05)
        baseline_mask = t <= cutoff
    if p_true_matrix.shape[0] == 0: return min_theta
    base = p_true_matrix[:, baseline_mask]
    base = np.where(np.isfinite(base), base, -np.inf)
    max_baseline = base.max(axis=1)
    max_baseline = max_baseline[np.isfinite(max_baseline)]
    if max_baseline.size == 0: return min_theta
    theta = np.quantile(max_baseline, 1.0 - alpha)
    theta = max(theta, min_theta)
    return float(min(theta, 0.999))

def main():
    ap = argparse.ArgumentParser(description="Trial-wise decodability threshold crossing (TTC) latencies")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="+", required=True)
    ap.add_argument("--root", default="results/session")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--smoothing_bins", type=int, default=3)
    ap.add_argument("--min_consecutive", type=int, default=2)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=0.5)
    ap.add_argument("--exclude_test_direction", action="store_true")
    ap.add_argument("--tag", default="ttc_v1")
    ap.add_argument("--min_theta_C", type=float, default=0.55)
    ap.add_argument("--min_theta_R", type=float, default=0.60)
    args = ap.parse_args()

    session_dir = os.path.join(args.root, args.sid)
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    area_data = {}; t = None
    for area in args.areas:
        info = load_axes(session_dir, area)
        if info["t"] is None: raise RuntimeError(f"{info['path']} missing time vector")
        if t is None: t = info["t"]
        else:
            if info["t"].shape != t.shape or not np.allclose(info["t"], t):
                raise RuntimeError("Time vectors differ across areas; ensure alignment.")
        area_data[area] = info

    n_trials = None
    for area in args.areas:
        ZC = area_data[area]["ZC"]
        if ZC is None: raise RuntimeError(f"{area_data[area]['path']} missing ZC")
        if n_trials is None: n_trials = ZC.shape[0]
        else:
            if ZC.shape[0] != n_trials: raise RuntimeError("Number of trials differs across areas.")
    n_bins = t.size

    ref_area = args.areas[0]
    C = area_data[ref_area]["C"]; R = area_data[ref_area]["R"]; DIR = area_data[ref_area]["DIR"]
    if (C is None) or (R is None):
        C2, R2, DIR2 = try_load_labels_from_cache(session_dir, args.sid, ref_area)
        C = C if C is not None else C2
        R = R if R is not None else R2
        DIR = DIR if DIR is not None else DIR2
    if (C is None) or (R is None):
        raise RuntimeError("Could not find trial labels C and R in axes npz or caches.")

    C = np.asarray(C).astype(int).reshape(-1)
    R = np.asarray(R).astype(int).reshape(-1)
    if C.shape[0] != n_trials or R.shape[0] != n_trials:
        raise RuntimeError("Label vectors C/R length mismatch with ZC trials.")
    if DIR is None:
        Rw = R - 1; DIR = np.where(C < 0, Rw, Rw + 3).astype(int)

    post_true_C = {area: np.full((n_trials, n_bins), np.nan, dtype=float) for area in args.areas}
    post_true_R = {area: np.full((n_trials, n_bins), np.nan, dtype=float) for area in args.areas}
    unique_DIR = np.unique(DIR)

    for area in args.areas:
        ZC = area_data[area]["ZC"]; ZR = area_data[area]["ZR"]
        if ZR is None: raise RuntimeError(f"{area_data[area]['path']} missing ZR")

        models_C = {}
        for r_excl in unique_DIR:
            tr_mask = DIR != r_excl if args.exclude_test_direction else np.ones(n_trials, dtype=bool)
            X_tr = ZC[tr_mask]; y_tr = ((C[tr_mask] > 0).astype(int))
            models_bins = []
            for j in range(n_bins):
                Xj = X_tr[:, j, :]
                model = fit_gaussian_lda_diag(Xj, y_tr, n_classes=2, lam=1e-3)
                models_bins.append(model)
            models_C[int(r_excl)] = models_bins

        for i in range(n_trials):
            fold = int(DIR[i]); models_bins = models_C[fold]
            y_true = 1 if C[i] > 0 else 0
            pis = np.zeros(n_bins, dtype=float)
            for j in range(n_bins):
                Xj = ZC[i, j, :][None, :]
                proba = predict_proba_gaussian_lda_diag(models_bins[j], Xj)[0]
                pis[j] = proba[y_true]
            post_true_C[area][i] = pis

        idx_c_neg = np.where(C < 0)[0]; idx_c_pos = np.where(C > 0)[0]
        R_within = R - 1
        for i in range(n_trials):
            cpos = C[i] > 0
            idx_pool = (idx_c_pos if cpos else idx_c_neg)
            idx_pool = idx_pool[idx_pool != i]
            if idx_pool.size < 5: continue
            y_pool = R_within[idx_pool]
            models_bins_R = []
            for j in range(n_bins):
                Xj = ZR[idx_pool, j, :]
                model = fit_gaussian_lda_diag(Xj, y_pool, n_classes=3, lam=1e-3)
                models_bins_R.append(model)
            y_true = int(R_within[i]); pis = np.zeros(n_bins, dtype=float)
            for j in range(n_bins):
                Xj = ZR[i, j, :][None, :]
                proba = predict_proba_gaussian_lda_diag(models_bins_R[j], Xj)[0]
                pis[j] = proba[y_true]
            post_true_R[area][i] = pis

    theta_C = {area: calibrate_threshold(post_true_C[area], t, alpha=args.alpha, min_theta=args.min_theta_C)
               for area in args.areas}
    theta_R = {area: calibrate_threshold(post_true_R[area], t, alpha=args.alpha, min_theta=args.min_theta_R)
               for area in args.areas}

    ttcC = {}; ttcR = {}
    for area in args.areas:
        arr = post_true_C[area]; tt = np.full(n_trials, np.nan, dtype=float)
        for i in range(n_trials):
            tt[i] = earliest_crossing(arr[i], t, theta_C[area], args.tmin, args.tmax,
                                      min_consecutive=args.min_consecutive, smooth_bins=args.smoothing_bins)
        ttcC[area] = tt
        arr = post_true_R[area]; tt = np.full(n_trials, np.nan, dtype=float)
        for i in range(n_trials):
            tt[i] = earliest_crossing(arr[i], t, theta_R[area], args.tmin, args.tmax,
                                      min_consecutive=args.min_consecutive, smooth_bins=args.smoothing_bins)
        ttcR[area] = tt

    out_dir = session_dir; out_path = os.path.join(out_dir, f"ttc_{args.tag}.npz")
    params = vars(args).copy()
    np.savez_compressed(out_path,
                        t=t, areas=np.array(args.areas, dtype=object),
                        C=C, R=R, DIR=DIR,
                        theta_C=json.dumps(theta_C), theta_R=json.dumps(theta_R),
                        ttcC=json.dumps({a: ttcC[a].tolist() for a in args.areas}),
                        ttcR=json.dumps({a: ttcR[a].tolist() for a in args.areas}),
                        params=json.dumps(params))
    print(f"[OK] Saved {out_path}")
    quick = {"sid": args.sid, "areas": args.areas, "theta_C": theta_C, "theta_R": theta_R,
             "n_trials": int(n_trials), "tmin": args.tmin, "tmax": args.tmax, "alpha": args.alpha}
    with open(os.path.join(out_dir, f"ttc_{args.tag}.json"), "w") as f:
        json.dump(quick, f, indent=2)

if __name__ == "__main__":
    main()
