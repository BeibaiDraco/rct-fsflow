# paper_project/paperflow/axes.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# ============ small utils ============

def window_mask(time_s: np.ndarray, win: Tuple[float,float]) -> np.ndarray:
    return (time_s >= win[0]) & (time_s <= win[1])

def avg_over_window(Z: np.ndarray, m: np.ndarray) -> np.ndarray:
    if not np.any(m):  # (N,B,U) -> (N,U)
        return np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
    return Z[:, m, :].mean(axis=1).astype(np.float64)

def unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0: n = 1.0
    return (v / n).astype(np.float64)

def orthogonalize(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray,float]:
    if u is None or v is None: return u, np.nan
    v = unit_vec(v)
    denom = (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    cosang = float(np.dot(u, v) / denom)
    r = u - np.dot(u, v) * v
    n = np.linalg.norm(r)
    if not np.isfinite(n) or n == 0: return u, cosang
    return (r / n).astype(np.float64), cosang

def weights_by_other(y_pm1: np.ndarray, other_pm1: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    ok = ~np.isnan(y_pm1) & ~np.isnan(other_pm1)
    y = (y_pm1[ok] > 0).astype(int)
    o = (other_pm1[ok] > 0).astype(int)
    w = np.ones(ok.sum(), dtype=float)
    for yi in (0,1):
        for oi in (0,1):
            m = (y==yi) & (o==oi)
            n = int(m.sum())
            if n>0: w[m] = 1.0 / max(n,1)
    return ok, w

def cv_select_C_binary(X: np.ndarray, y_pm1: np.ndarray, C_grid: List[float], sw: Optional[np.ndarray]=None) -> tuple[LogisticRegression,float,float,float]:
    y = (y_pm1 > 0).astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best_auc, best_C = -np.inf, None
    for Cval in C_grid:
        aucs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(penalty="l2", C=Cval, solver="liblinear",
                                     class_weight="balanced", max_iter=2000)
            clf.fit(X[tr], y[tr], sample_weight=(sw[tr] if sw is not None else None))
            s = clf.decision_function(X[te])
            try: aucs.append(roc_auc_score(y[te], s))
            except ValueError: pass
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_C = float(np.mean(aucs)), float(Cval)
    if best_C is None: best_C = 1.0
    clf = LogisticRegression(penalty="l2", C=best_C, solver="liblinear",
                             class_weight="balanced", max_iter=2000)
    clf.fit(X, y, sample_weight=sw)
    return clf, float(best_auc if np.isfinite(best_auc) else np.nan), np.nan, float(best_C)

def multinomial_cv_acc(X: np.ndarray, y_int: np.ndarray, C_grid: List[float]) -> tuple[LogisticRegression,float,float,float]:
    y = y_int.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best_acc, best_C = -np.inf, None
    for Cval in C_grid:
        accs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(penalty="l2", C=Cval, solver="lbfgs",
                                     max_iter=2000, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pr = clf.predict(X[te])
            accs.append(accuracy_score(y[te], pr))
        if accs and np.mean(accs) > best_acc:
            best_acc, best_C = float(np.mean(accs)), float(Cval)
    if best_C is None: best_C = 1.0
    clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs",
                             max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf, float(best_acc if np.isfinite(best_acc) else np.nan), np.nan, float(best_C)

def learn_category_invariant(Xmean: np.ndarray, C: np.ndarray, R: np.ndarray,
                             C_reg: float, C_dim: int = 1) -> np.ndarray:
    """Hold-one-R-out stacking for direction-invariant sC (stim-align)."""
    Rvals = np.unique(R[~np.isnan(R)])
    Ws = []
    if Rvals.size < 2:
        lr = LogisticRegression(penalty="l2", C=C_reg, solver="liblinear",
                                class_weight="balanced", max_iter=2000)
        y = (C > 0).astype(int)
        lr.fit(Xmean, y)
        return unit_vec(lr.coef_.ravel()).reshape(-1,1)
    for rv in Rvals:
        m_te = (R == rv);  m_tr = ~m_te & (~np.isnan(C))
        if m_tr.sum() < 30 or np.unique(np.sign(C[m_tr])).size < 2: continue
        lr = LogisticRegression(penalty="l2", C=C_reg, solver="liblinear",
                                class_weight="balanced", max_iter=2000)
        lr.fit(Xmean[m_tr], (C[m_tr] > 0).astype(int))
        Ws.append(lr.coef_.ravel())
    if not Ws:
        return np.zeros((Xmean.shape[1], 0))
    W = np.vstack(Ws)  # (folds x units)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_C = Vt[:max(1, C_dim)].T
    Q, _ = np.linalg.qr(W_C[:, :C_dim])
    return Q

# ============ core training ============

@dataclass
class AxisPack:
    sC: Optional[np.ndarray]
    sR: Optional[np.ndarray]         # (U, R_dim)
    sS_raw: Optional[np.ndarray]
    sS_inv: Optional[np.ndarray]
    metrics: Dict

def train_axes_for_area(
    cache: Dict,
    feature_set: List[str],              # ["C","R","S"]
    time_s: np.ndarray,
    winC: Optional[Tuple[float,float]] = None,
    winR: Optional[Tuple[float,float]] = None,
    winS: Optional[Tuple[float,float]] = None,
    orientation: Optional[str] = None,   # "vertical"|"horizontal"|None
    C_dim: int = 1, R_dim: int = 2, S_dim: int = 1,
    make_S_invariant: bool = True,
    c_invariance: str = "holdoutR",      # "none"|"holdoutR"
    C_grid: Optional[List[float]] = None,
    r_orthC: bool = False,
) -> AxisPack:
    if C_grid is None: C_grid = [0.1, 0.3, 1.0, 3.0, 10.0]

    Z = cache["Z"].astype(np.float64)                   # (N,B,U)
    C = cache.get("lab_C", np.full(Z.shape[0], np.nan)).astype(np.float64)
    R = cache.get("lab_R", np.full(Z.shape[0], np.nan)).astype(np.float64)
    S = cache.get("lab_S", np.full(Z.shape[0], np.nan)).astype(np.float64)
    OR = cache.get("lab_orientation", np.array(["pooled"]*Z.shape[0], dtype=object))
    IC = cache.get("lab_is_correct", np.ones(Z.shape[0], dtype=bool))
    align_event = cache.get("meta",{}).get("align_event","stim")

    keep = np.ones(Z.shape[0], dtype=bool)
    keep &= IC
    if orientation is not None and "lab_orientation" in cache:
        keep &= (OR.astype(str) == orientation)

    Z = Z[keep]; C = C[keep]; R = R[keep]; S = S[keep]
    metrics = {
        "n_trials": int(Z.shape[0]), "n_bins": int(Z.shape[1]), "n_units": int(Z.shape[2]),
        "winC": winC, "winR": winR, "winS": winS,
        "C_dim": int(C_dim), "R_dim": int(R_dim), "S_dim": int(S_dim),
        "c_invariance": c_invariance, "align_event": align_event,
        "r_orthC": bool(r_orthC),
    }

    sC_vec = None; sR_mat = None; sS_raw_vec = None; sS_inv_vec = None
    mu_R = None; std_R = None

    # ----- sC -----
    if "C" in feature_set and winC is not None:
        if C_dim != 1:
            metrics["warn_dimC"] = f"Requested dimC={C_dim} but only 1-D binary axis is implemented; producing 1-D."
        mC = window_mask(time_s, winC)
        Xw = avg_over_window(Z, mC)
        m = ~np.isnan(C)
        Xc, yc, Rc = Xw[m], C[m], R[m]
        if Xc.shape[0] >= 20 and np.unique(np.sign(yc)).size >= 2:
            if c_invariance == "holdoutR" and align_event == "stim" and not np.all(np.isnan(Rc)):
                Wc = learn_category_invariant(Xc, yc, Rc, C_reg=C_grid[2], C_dim=1)
                sC_vec = unit_vec(Wc[:,0])
                metrics.update({"sC_auc_mean": np.nan, "sC_C": np.nan, "sC_n": int(Xc.shape[0]),
                                "sC_invariance": "holdoutR"})
            else:
                clfC, aucC, _, Cbest = cv_select_C_binary(Xc, yc, C_grid)
                sC_vec = unit_vec(clfC.coef_.ravel().astype(np.float64))
                metrics.update({"sC_auc_mean": float(aucC), "sC_C": float(Cbest), "sC_n": int(Xc.shape[0]),
                                "sC_invariance": "none"})
        # canonicalize sign at training window
        if sC_vec is not None:
            yy = (yc > 0).astype(int)
            if np.unique(yy).size >= 2:
                sc = Xc @ sC_vec
                try:
                    auc_train = roc_auc_score(yy, sc)
                    if auc_train < 0.5:
                        sC_vec = -sC_vec
                        auc_train = 1.0 - auc_train
                    metrics["sC_auc_train"] = float(auc_train)
                except Exception:
                    pass

    # ----- sR (multi-D; stim-align only) -----
    if "R" in feature_set and winR is not None:
        mR = window_mask(time_s, winR)
        Xw = avg_over_window(Z, mR)
        # per-category μ/σ on training window
        cats = (-1.0, 1.0)
        mu_R = np.zeros((2, Z.shape[2]), float)
        std_R = np.ones((2, Z.shape[2]), float)
        for idx, cval in enumerate(cats):
            mc = (np.sign(C) == (1 if cval>0 else -1))
            if mc.any():
                mu = Xw[mc].mean(axis=0)
                sd = Xw[mc].std(axis=0, ddof=1)
                sd[sd < 1e-8] = 1.0
                mu_R[idx] = mu; std_R[idx] = sd

        # learn R within each category; residualize vs sC if present
        W_blocks = []; accs = []
        for cval in (-1.0, 1.0):
            mc = (np.sign(C) == (1 if cval>0 else -1))
            if not np.any(mc): continue
            Xi = Xw[mc].copy()
            # center/whiten
            idx = 0 if cval < 0 else 1
            Xi = (Xi - mu_R[idx]) / std_R[idx]
            # residualize against sC
            if sC_vec is not None:
                q = sC_vec.reshape(-1,1) / (np.linalg.norm(sC_vec) + 1e-12)
                Xi = Xi - (Xi @ q) @ q.T
            # map R labels to 0..K-1
            ydir = R[mc].copy()
            uniq = np.unique(ydir[~np.isnan(ydir)])
            if len(uniq) < 3: continue
            mapv = {float(u): i for i,u in enumerate(sorted(uniq))}
            yi = np.array([mapv.get(float(v), np.nan) for v in ydir], dtype=float)
            mm = ~np.isnan(yi)
            Xi, yi = Xi[mm], yi[mm].astype(int)
            if Xi.shape[0] < 30: continue
            clfR, accR, _, CbestR = multinomial_cv_acc(Xi, yi, C_grid)
            accs.append(accR)
            Wc = clfR.coef_.astype(np.float64)  # (K, U)
            W_blocks.append(Wc)
        if W_blocks:
            W = np.vstack(W_blocks)  # (K*C, U)
            U_, S_, Vt_ = np.linalg.svd(W, full_matrices=False)
            rdim = max(1, int(R_dim))
            sR_mat = Vt_[:rdim].T  # (U, rdim)
            # disentangle vs sC
            if r_orthC and sC_vec is not None:
                for d in range(sR_mat.shape[1]):
                    sR_mat[:,d], _ = orthogonalize(sR_mat[:,d], sC_vec)
            for d in range(sR_mat.shape[1]): sR_mat[:,d] = unit_vec(sR_mat[:,d])
            metrics.update({"sR_cv_acc_mean": float(np.nanmean(accs)) if accs else np.nan, "sR_dim": int(rdim),
                            "mu_R": mu_R.tolist(), "std_R": std_R.tolist()})

    # ----- sS (raw + invariant; sacc-align only) -----
    if "S" in feature_set and winS is not None:
        if S_dim != 1: metrics["warn_dimS"] = f"Requested dimS={S_dim} but only 1-D S is implemented; producing 1-D."
        mS = window_mask(time_s, winS)
        Xw = avg_over_window(Z, mS)
        ok, sw = weights_by_other(S, C)
        Xs, ys = Xw[ok], S[ok]
        if Xs.shape[0] >= 20 and np.unique(np.sign(ys)).size >= 2:
            clfS, aucS, _, CbestS = cv_select_C_binary(Xs, ys, C_grid, sw)
            sS_raw_vec = unit_vec(clfS.coef_.ravel().astype(np.float64))
            # canonicalize S sign
            try:
                yy = (ys > 0).astype(int)
                sr = (Xs @ sS_raw_vec).ravel()
                aucSwin = roc_auc_score(yy, sr)
                if aucSwin < 0.5:
                    sS_raw_vec = -sS_raw_vec
                    aucSwin = 1.0 - aucSwin
                metrics["sSraw_auc_train"] = float(aucSwin)
            except Exception:
                pass
            sS_inv_vec = sS_raw_vec
            cos_sSraw_sC = None
            if sC_vec is not None and make_S_invariant:
                sS_inv_vec, cos_sSraw_sC = orthogonalize(sS_raw_vec, sC_vec)
            metrics.update({"sSraw_auc_mean": float(aucS), "sS_C": float(CbestS), "sS_n": int(Xs.shape[0]),
                            "cos_sSraw_sC": float(cos_sSraw_sC) if cos_sSraw_sC is not None else None})

    # ----- leak tests -----
    def auc_from_axis(Xwin, axis, y_pm1) -> float:
        if axis is None or axis.size == 0: return np.nan
        y = (y_pm1 > 0).astype(int)
        if np.unique(y).size < 2: return np.nan
        s = Xwin @ axis
        try: return float(roc_auc_score(y, s))
        except Exception: return np.nan

    if sC_vec is not None and winS is not None and "S" in feature_set:
        Xs = avg_over_window(Z, window_mask(time_s, winS))
        metrics["auc_C_from_sS_inv"] = auc_from_axis(Xs, sS_inv_vec, C)

    if sS_inv_vec is not None and winC is not None and "C" in feature_set:
        Xc = avg_over_window(Z, window_mask(time_s, winC))
        metrics["auc_S_from_sC"] = auc_from_axis(Xc, sC_vec, S)

    return AxisPack(sC=sC_vec, sR=sR_mat, sS_raw=sS_raw_vec, sS_inv=sS_inv_vec, metrics=metrics)

def save_axes(out_dir: str, area: str, pack: AxisPack, meta_extra: Dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"axes_{area}.npz")
    meta = dict(meta_extra); meta.update(pack.metrics)
    npz = dict(
        meta=json.dumps(meta),
        sC=(pack.sC if pack.sC is not None else np.array([])),
        sR=(pack.sR if pack.sR is not None else np.array([[]])),
        sS_raw=(pack.sS_raw if pack.sS_raw is not None else np.array([])),
        sS_inv=(pack.sS_inv if pack.sS_inv is not None else np.array([])),
    )
    if "mu_R" in pack.metrics: npz["mu_R"] = np.array(pack.metrics["mu_R"], dtype=np.float32)
    if "std_R" in pack.metrics: npz["std_R"] = np.array(pack.metrics["std_R"], dtype=np.float32)
    np.savez_compressed(path, **npz)
    return path
