# paper_project/paperflow/axes.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------- small helpers ----------

def window_mask(time_s: np.ndarray, win: Tuple[float,float]) -> np.ndarray:
    return (time_s >= win[0]) & (time_s <= win[1])

# 2) fix avg_over_window fallback
def avg_over_window(Z: np.ndarray, mask_t: np.ndarray) -> np.ndarray:
    # (trials, bins, units) -> (trials, units)
    if not np.any(mask_t):
        return np.zeros((Z.shape[0], Z.shape[2]), dtype=np.float64)
    return Z[:, mask_t, :].mean(axis=1).astype(np.float64)


def unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v); 
    if not np.isfinite(n) or n == 0: 
        return np.zeros_like(v)
    return v / n

def orthogonalize(u: np.ndarray, v: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[float]]:
    if v is None or u.size == 0: 
        return u, None
    v = unit_vec(v)
    # protect against degenerate v
    vn = np.linalg.norm(v)
    if not np.isfinite(vn) or vn == 0:
        return u, None
    dot = float(np.dot(u, v) / (np.linalg.norm(u)*vn + 1e-12))
    u2 = u - dot * v
    un = np.linalg.norm(u2)
    if not np.isfinite(un) or un == 0:
        return u, dot
    return u2 / un, dot

def auc_binary_scores(scores: np.ndarray, y_pm1: np.ndarray) -> float:
    """AUC for binary ±1 labels from 1D scores."""
    y = (y_pm1 > 0).astype(int)
    if np.unique(y).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return np.nan

def cv_logreg_binary(X: np.ndarray, y_pm1: np.ndarray, C_grid: List[float], sample_weight: Optional[np.ndarray]) -> tuple[LogisticRegression, float, float]:
    y = (y_pm1 > 0).astype(int)
    best, bestC = -np.inf, None
    skf = np.random.RandomState(0)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for Cval in C_grid:
        vals = []
        for tr, te in splitter.split(X, y):
            clf = LogisticRegression(penalty="l2", C=Cval, solver="liblinear", class_weight="balanced", max_iter=2000)
            clf.fit(X[tr], y[tr], sample_weight=(None if sample_weight is None else sample_weight[tr]))
            s = clf.decision_function(X[te])
            try: vals.append(roc_auc_score(y[te], s))
            except Exception: pass
        if vals:
            m = float(np.mean(vals))
            if m > best: best, bestC = m, float(Cval)
    if bestC is None: bestC = 1.0
    clf = LogisticRegression(penalty="l2", C=bestC, solver="liblinear", class_weight="balanced", max_iter=2000)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf, best, bestC

def multinomial_cv_acc(X: np.ndarray, y_int: np.ndarray, C_grid: List[float]) -> tuple[LogisticRegression, float, float]:
    y = y_int.astype(int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best, bestC = -np.inf, None
    for Cval in C_grid:
        vals = []
        for tr, te in splitter.split(X, y):
            clf = LogisticRegression(penalty="l2", C=Cval, solver="lbfgs", max_iter=2000, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pr = clf.predict(X[te])
            vals.append(accuracy_score(y[te], pr))
        m = float(np.mean(vals)) if vals else np.nan
        if m > best: best, bestC = m, float(Cval)
    if bestC is None: bestC = 1.0
    clf = LogisticRegression(penalty="l2", C=bestC, solver="lbfgs", max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf, best, bestC

def learn_category_invariant(Xmean: np.ndarray, C: np.ndarray, R: np.ndarray, C_reg: float, C_dim: int = 1) -> np.ndarray:
    """Hold-one-R-out stacking to suppress direction content in sC."""
    Rvals = np.unique(R[~np.isnan(R)])
    Ws = []
    for rv in Rvals:
        mtr = (~np.isnan(C)) & (R != rv)
        if mtr.sum() < 30 or np.unique(np.sign(C[mtr])).size < 2: 
            continue
        clf = LogisticRegression(penalty="l2", C=C_reg, solver="liblinear", class_weight="balanced", max_iter=2000)
        clf.fit(Xmean[mtr], (C[mtr] > 0).astype(int))
        Ws.append(clf.coef_.ravel())
    if not Ws:
        return np.zeros((Xmean.shape[1], 0))
    W = np.vstack(Ws)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)  # Vt: (k x U)
    Wc = Vt[:max(1, C_dim)].T
    # orthonormalize
    Q, _ = np.linalg.qr(Wc)
    return Q[:, :C_dim]

def select_topk_units_by_auc(Xmean: np.ndarray, y_pm1: np.ndarray, frac: float) -> np.ndarray:
    """Return boolean mask of top-k fraction units by |AUC-0.5| using univariate projections X[:,j]."""
    y = (y_pm1 > 0).astype(int)
    nU = Xmean.shape[1]
    if frac >= 0.999 or nU <= 1:
        return np.ones(nU, dtype=bool)
    scores = np.zeros(nU, dtype=float)
    for j in range(nU):
        try:
            scores[j] = abs(roc_auc_score(y, Xmean[:,j]))
        except Exception:
            scores[j] = 0.0
    k = max(1, int(np.ceil(frac * nU)))
    idx = np.argsort(scores)[-k:]
    mask = np.zeros(nU, dtype=bool); mask[idx] = True
    return mask

# ---------- main trainer ----------

@dataclass
class AxisPack:
    sC: Optional[np.ndarray]
    sR: Optional[np.ndarray]         # (U, R_dim) or None
    sS_raw: Optional[np.ndarray]
    sS_inv: Optional[np.ndarray]
    sT: Optional[np.ndarray] = None  # target configuration axis (binary)
    meta: Dict = None
    sO: Optional[np.ndarray] = None  # NEW: context / orientation axis

def train_axes_for_area(
    cache: Dict,
    feature_set: List[str],                # any of ["C","R","S","T","O"]
    time_s: np.array,
    winC: Optional[Tuple[float,float]] = None,
    winR: Optional[Tuple[float,float]] = None,
    winS: Optional[Tuple[float,float]] = None,
    winT: Optional[Tuple[float,float]] = None,  # window for T axis (default: winC if None)
    orientation: Optional[str] = None,     # 'vertical'|'horizontal'|None
    C_dim: int = 1,
    R_dim: int = 2,
    S_dim: int = 1,                        # kept 1 by design (binary S)
    make_S_invariant: bool = True,
    C_grid: Optional[List[float]] = None,
    # selection knobs (for training only)
    select_mode: Optional[str] = None,     # None|'C'|'R'|'S'
    select_frac: float = 1.0,              # 0<..≤1.0
    # PT gating
    pt_min_ms: Optional[float] = None,     # if not None: filter to PT >= this (ms)
) -> AxisPack:

    if C_grid is None:
        Cworth = [0.1, 0.3, 1.0, 3.0, 10.0]
    else:
        Cworth = C_grid

    Z = cache["Z"].astype(np.float64)
    C = cache.get("lab_C", np.full(Z.shape[0], np.nan)).astype(np.float64)
    R = cache.get("lab_R", np.full(Z.shape[0], np.nan)).astype(np.float64)
    S = cache.get("lab_S", np.full(Z.shape[0], np.nan)).astype(np.float64)
    # Target configuration (T): category × saccade_location_sign
    # Prefer stored lab_T if present; otherwise derive it lazily (lets us run on older caches).
    if "lab_T" in cache:
        Tcfg = cache.get("lab_T", np.full(Z.shape[0], np.nan)).astype(np.float64)
    else:
        Tcfg = np.sign(C) * np.sign(S)
        Tcfg[~(np.isfinite(C) & np.isfinite(S))] = np.nan
    OR = cache.get("lab_orientation", np.array(["pooled"] * Z.shape[0], dtype=object))
    PT = cache.get("lab_PT_ms", None)  # ms
    IC = cache.get("lab_is_correct", np.ones(Z.shape[0], dtype=bool))

    keep = np.ones(Z.shape[0], dtype=bool)
    keep &= IC
    if orientation is not None and "lab_orientation" in cache:
        keep &= (OR.astype(str) == orientation)
    if pt_min_ms is not None and (PT is not None):
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))

    # apply trial mask
    Z = Z[keep]; C = C[keep]; R = R[keep]; S = S[keep]; Tcfg = Tcfg[keep]; OR = OR[keep]
    # Use explicit winT if provided, else fall back to winC
    winT_used = winT if winT is not None else winC
    meta = dict(
        n_trials=int(Z.shape[0]), n_bins=int(Z.shape[1]), n_units=int(Z.shape[2]),
        orientation=orientation, pt_min_ms=(float(pt_min_ms) if pt_min_ms is not None else None),
        feature_set=feature_set, winC=winC, winR=winR, winS=winS, winT=winT_used,
        C_dim=int(C_dim), R_dim=int(R_dim), S_dim=int(S_dim),
        select_mode=(select_mode or "none"), select_frac=float(select_mode and select_frac or 1.0),
        sC_invariance=None,  # filled below if trained
    )

    sC_vec = None
    sR_mat = None
    sS_raw = None
    sS_inv = None
    sT_vec = None
    sO_vec = None  # NEW: context / orientation axis

    # ---------- Train sC ----------
    if "C" in feature_set and winC is not None:
        mC = window_mask(time_s, winC)
        Xc_full = avg_over_window(Z, mC)  # (N x U)
        yc = C.copy()
        okc = ~np.isnan(yc)
        Xc = Xc_full[okc]; yC = yc[okc]
        # optional unit selection for C training
        Xc_used = Xc
        maskC = np.ones(Xc.shape[1], dtype=bool)
        if select_mode == "C" and select_frac < 0.999:
            maskC = select_topk_units_by_auc(Xc, yC, select_frac)
            Xc_used = Xc[:, maskC]
            meta["select_C_units"] = int(maskC.sum())
        if Xc_used.shape[0] >= 20 and np.unique(np.sign(yC)).size >= 2:
            if cache["meta"].get("align_event","stim") == "stim":
                # direction-invariant via holdout R
                # pass only the selected columns for C stage
                if select_mode == "C":
                    # map R for the retained rows
                    Rc = R[okc]
                    Wc = learn_category_invariant(Xc_used, yC, Rc, C_reg=Cworth[2], C_dim=C_dim)
                    if Wc.shape[1] >= 1:
                        tmp = np.zeros((Xc.shape[1], Wc.shape[1]))
                        tmp[maskC,:] = Wc
                        sC_vec = unit_vec(tmp[:,0])
                        meta["sC_invariance"] = "holdoutR"
                        meta["sC_C"] = float(Cworth[2]); meta["sC_n"] = int(Xc_used.shape[0])
                else:
                    # simple CV logistic (direction suppression happens in R stage)
                    clfC, aucC, Cbest = cv_logreg_binary(Xc_used, yC, Cworth, None)
                    wC = clfC.coef_.ravel().astype(np.float64)
                    # expand back to full U as needed
                    if Xc_used.shape[1] != Xc.shape[1]:
                        tmp = np.zeros(Xc.shape[1], dtype=float)
                        tmp[maskC] = wC
                        wC = tmp
                    sC_vec = unit_vec(wC)
                    meta["sC_auc_mean"] = float(aucC); meta["sC_C"] = float(Cbest); meta["sC_n"] = int(Xc_used.shape[0]); meta["sC_invariance"] = "none"
            else:
                # saccade-aligned C (for S invariance) — simple binary logistic
                clfC, aucC, Cbest = cv_logreg_binary(Xc_used, yC, Cworth, None)
                wC = clfC.coef_.ravel().astype(np.float64)
                if select_mode == "C" and Xc_used.shape[1] != Xc.shape[1]:
                    tmp = np.zeros(Xc.shape[1], dtype=float); tmp[maskC] = wC; wC = tmp
                sC_vec = unit_vec(wC)
                meta["sC_auc_mean"] = float(aucC); meta["sC_C"] = float(Cbest); meta["sC_n"] = int(Xc.shape[0]); meta["sC_invariance"] = "sacc"
        # rectify sC orientation if trained
        if sC_vec is not None and Xc.shape[1] == sC_vec.shape[0]:
            sC_scores = (Xc @ sC_vec)
            aucC0 = auc_binary_scores(sC_scores, yC)
            # orient sC to positive AUC
            if np.isfinite(aucC0) and aucC0 < 0.5:
                sC_vec = -sC_vec
                aucC0 = 1.0 - aucC0
            meta["sC_auc_proj"] = float(aucC0)

    # ---------- Train sR (stim only) ----------
    mu_R = None; std_R = None
    if "R" in feature_set and winR is not None:
        mR = window_mask(time_s, winR)
        Xr = avg_over_window(Z, mR)                   # (N,U)
        # per-category center/whiten stats
        cats = np.array([-1.0, +1.0], dtype=float)
        mu_R = np.zeros((2, Xr.shape[1]), dtype=float)
        std_R = np.ones((2, Xr.shape[1]), dtype=float)
        for i,cval in enumerate(cats):
            m = (np.sign(C) == cval)
            if m.any():
                mu_R[i]  = Xr[m].mean(axis=0)
                std = Xr[m].std(axis=0, ddof=1)
                std[std < 1e-8] = 1.0
                std_R[i] = std
        # center/whiten per category
        Xcw = np.empty_like(Xr)
        for i,cval in enumerate(cats):
            m = (np.sign(C) == cval)
            if m.any():
                Xcw[m] = (Xr[m] - mu_R[i]) / std_R[i]
        # restrict to top-variance units if desired
        # reuse your earlier 'resp_frac_R' semantics
        # We'll pass in select_frac for R through 'select_mode=="R"'
        maskR = np.ones(Xcw.shape[1], dtype=bool)
        if select_mode == "R" and select_frac < 0.999:
            # score by average |X| over training window (proxy for responsiveness)
            var = np.mean(np.abs(Xcw), axis=0)
            k = max(1, int(np.ceil(select_frac * Xcw.shape[1])))
            idx = np.argsort(var)[-k:]
            maskR[:] = False; maskR[idx] = True
            meta["select_R_units"] = int(maskR.sum())
        Xr_use = Xcw[:, maskR]
        if Xr_use.shape[0] >= 30 and Xr_use.shape[1] >= 1:
            # train multinomial per category inside cv folds
            # > we build pooled W by stacking per-category softmax weights then SVD
            blocks = []
            accs = []
            for cval in (-1.0, +1.0):
                m = (np.sign(C) == cval)
                if not m.any(): 
                    continue
                ydir = R[m]
                cats_present = np.unique(ydir[~np.isnan(ydir)])
                if cats_present.size < 2: 
                    continue
                # map to [0..K-1]
                yint = np.array([np.where(cats_present==v)[0][0] if np.isfinite(v) else -1 for v in ydir], dtype=int)
                mm = (yint >= 0)
                Xi, yi = Xr_use[m][mm], yint[mm]
                if Xi.shape[0] < 30: 
                    continue
                clfR, accm, Cbest = multinomial_cv_acc(Xi, yi, Cworth)
                accs.append(accm)
                blocks.append(clfR.coef_.astype(np.float64))  # (K x U_sel)
            if blocks:
                W = np.vstack(blocks)  # (sum K) x U_sel
                U_, S_, Vt = np.linalg.svd(W, full_matrices=False)
                rd = max(1, int(R_dim))
                sR_small = Vt[:rd].T  # (U_sel x rd)
                # expand back to full U
                sR_full = np.zeros((Xr.shape[1], rd), dtype=float)
                sR_full[maskR, :] = sR_small
                # make sR ⟂ sC if available
                if sC_vec is not None:
                    for j in range(rd):
                        sR_full[:,j], _ = orthogonalize(sR_full[:,j], sC_vec)
                        sR_full[:,j] = unit_vec(sR_full[:,j])
                sR_mat = sR_full
                meta["sR_dim"] = int(rd)
                meta["sR_cv_acc_mean"] = float(np.nanmean(accs)) if accs else np.nan
            else:
                sR_mat = None

    # ---------- Train sS (sacc only) ----------
    if "S" in feature_set and winS is not None:
        mS = window_mask(time_s, winS)
        Xs_full = avg_over_window(Z, mS)
        ys = S.copy()
        ok = ~np.isnan(ys)
        Xs = Xs_full[ok]; ys2 = ys[ok]
        # option to preselect units for S training
        maskS = np.ones(Xs.shape[1], dtype=bool)
        if select_mode == "S" and select_frac < 0.999:
            scores = np.array([abs(auc_binary_scores(Xs[:,j], ys2) - 0.5) for j in range(Xs.shape[1])])
            k = max(1, int(np.ceil(select_frac * Xs.shape[1])))
            idx = np.argsort(scores)[-k:]
            maskS[:] = False; maskS[idx] = True
            Xs = Xs[:, maskS]
            meta["select_S_units"] = int(maskS.sum())
        if Xs.shape[0] >= 20 and np.unique(np.sign(ys2)).size >= 2:
            # Balance via sample weights (within C)
            yC_used = C[ok]
            ok2 = ~np.isnan(yC_used) & ~np.isnan(ys2)
            yb = ys2[ok2]; X_use = Xs[ok2]; w = np.ones_like(yb, dtype=float)
            # stratified by C and S
            for sign_s in (0,1):
                for sign_c in (0,1):
                    m = (yb > 0) == bool(sign_s)
                    mc = (yC_used[ok2] > 0) == bool(sign_c)
                    mask = m & mc
                    cnt = mask.sum()
                    if cnt > 0:
                        w[mask] = 1.0 / cnt
            clfS, aucS, CbestS = cv_logreg_binary(X_use, yb, Cworth, w)
            wS_small = clfS.coef_.ravel().astype(np.float64)
            # expand to full U if selection applied
            if not maskS.all():
                wfull = np.zeros(Xs_full.shape[1], dtype=float); wfull[maskS] = wS_small; wS = wfull
            else:
                wS = wS_small
            sS_raw = unit_vec(wS)
            sS_inv, cos = orthogonalize(sS_raw, sC_vec) if make_S_invariant else (sS_raw, None)
            meta["sSraw_auc_mean"] = float(aucS); meta["sS_C"] = float(CbestS); meta["sS_n"] = int(Xs.shape[0])
            meta["cos_sSraw_sC"] = (None if cos is None else float(cos))

    # ---------- Train sT (target configuration) ----------
    # Use winT if provided, else fall back to winC.
    winT_actual = winT if winT is not None else winC
    if "T" in feature_set and winT_actual is not None:
        mT = window_mask(time_s, winT_actual)
        Xt_full = avg_over_window(Z, mT)  # (N x U)
        yt = Tcfg.copy()
        okt = ~np.isnan(yt)
        Xt = Xt_full[okt]; yT = yt[okt]
        if Xt.shape[0] >= 20 and np.unique(np.sign(yT)).size >= 2:
            clfT, aucT, CbestT = cv_logreg_binary(Xt, yT, Cworth, sample_weight=None)
            wT = clfT.coef_.ravel().astype(np.float64)
            sT_vec = unit_vec(wT)
            # orient to positive AUC
            t_scores = Xt @ sT_vec
            aucT0 = auc_binary_scores(t_scores, yT)
            if np.isfinite(aucT0) and aucT0 < 0.5:
                sT_vec = -sT_vec
                aucT0 = 1.0 - aucT0
            meta["sT_auc_mean"] = float(aucT)
            meta["sT_auc_proj"] = float(aucT0) if np.isfinite(aucT0) else np.nan
            meta["sT_C"] = float(CbestT)
            meta["sT_n"] = int(Xt.shape[0])

    # ---------- Train sO (context / orientation axis) ----------
    if "O" in feature_set and winC is not None:
        # Use the same window as C by default
        mO = window_mask(time_s, winC)
        Xo_full = avg_over_window(Z, mO)  # (N, U)

        # Orientation labels: +1 vertical, -1 horizontal
        # Note: OR is already filtered by `keep` along with Z, C, R, S
        OR_str = np.asarray(OR).astype(str)
        yO = np.full(Z.shape[0], np.nan, dtype=float)
        yO[OR_str == "vertical"] = +1.0
        yO[OR_str == "horizontal"] = -1.0

        ok = np.isfinite(yO)
        Xo = Xo_full[ok]
        yO2 = yO[ok]

        if Xo.shape[0] >= 20 and np.unique(yO2).size == 2:
            # Balance across (C,O) to avoid just reusing C
            Ck = C[ok]
            w = np.ones_like(yO2, dtype=float)
            for c_sign in (-1.0, +1.0):
                for o_sign in (-1.0, +1.0):
                    m = (Ck == c_sign) & (yO2 == o_sign)
                    cnt = m.sum()
                    if cnt > 0:
                        w[m] = 1.0 / cnt

            clfO, aucO, CbestO = cv_logreg_binary(Xo, yO2, Cworth, sample_weight=w)
            wO = clfO.coef_.ravel().astype(np.float64)

            # Optionally orthogonalize sO to sC to get "pure context"
            if sC_vec is not None:
                wO, cos_sO_sC = orthogonalize(wO, sC_vec)
                meta["cos_sO_sC"] = float(cos_sO_sC) if cos_sO_sC is not None else None

            sO_vec = unit_vec(wO)
            meta["sO_auc_mean"] = float(aucO)
            meta["sO_C"] = float(CbestO)
            meta["sO_n"] = int(Xo.shape[0])

    # finalize meta with simple AUC checks for trained axes
    res = dict(meta)
    return AxisPack(sC=sC_vec, sR=sR_mat, sS_raw=sS_raw, sS_inv=sS_inv, sT=sT_vec, meta=res, sO=sO_vec)

def save_axes(out_dir: str, area: str, pack: AxisPack) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"axes_{area}.npz")
    np.savez_compressed(
        path,
        sC=(pack.sC if pack.sC is not None else np.array([])),
        sR=(pack.sR if pack.sR is not None else np.array([[]])),
        sS_raw=(pack.sS_raw if pack.sS_raw is not None else np.array([])),
        sS_inv=(pack.sS_inv if pack.sS_inv is not None else np.array([])),
        sT=(pack.sT if pack.sT is not None else np.array([])),
        sO=(pack.sO if pack.sO is not None else np.array([])),  # NEW
        meta=json.dumps(pack.meta),
    )
    return path
