#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L01_trial_latency.py

Per-trial time-to-decodability latencies for feature-specific codes:

  • Category (C): direction-invariant decoding (hold-one-R-out across both categories)
  • Within-category Direction (R): multinomial decoding within each category

For each area & trial:
  1) build time-resolved decoder probabilities p_true(t) via CV per time-bin
  2) smooth lightly across time (optional)
  3) find latency = first t ≥ tmin where p_true(t..t+m-1) ≥ threshold

AUTO threshold (optional):
  --auto_thr picks a threshold per feature to hit a target fraction of trials
             with valid latency (highest threshold whose resolved fraction ≥ target).
  Strategy:
    • session  (default): one threshold per feature shared by all areas in this session
    • area             : a separate threshold per area per feature

Outputs:
  results/session/<sid>/<tag>/latency_C_<AREA>.npy
  results/session/<sid>/<tag>/latency_R_<AREA>.npy
  results/session/<sid>/<tag>/latency_pairs_<A>to<B>.png (ΔT distributions)
  results/session/<sid>/<tag>/latency_summary.csv (per pair stats)
  results/session/<sid>/<tag>/latency_thresholds.json  (actual thresholds used)

Example:
  python L01_trial_latency.py --sid 20200926 --axes_tag win160_k5_perm500 \
    --tmin 0.0 --hold_m 3 --smooth_bins 3 \
    --auto_thr --auto_targetC 0.60 --auto_targetR 0.60 --auto_strategy session
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import uniform_filter1d

# Silence the deprecation spam if any downstream libs still pass multi_class explicitly
warnings.filterwarnings("ignore", message=".*'multi_class' was deprecated.*")

VALID = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

# ---------------- IO ----------------
def load_axes(sid: int, area: str, axes_dir: Path, tag: str=""):
    base = axes_dir / str(sid)
    tagged = base / tag if tag else base
    f = tagged / f"axes_{area}.npz"
    if not f.exists():
        raise FileNotFoundError(f"axes_{area}.npz not found in {tagged}")
    z = np.load(f, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    ZC = z["ZC"] if "ZC" in z.files else z["sC"][..., None]
    ZR = z["ZR"] if "ZR" in z.files else z["sR"][..., None]
    return ZC.astype(float), ZR.astype(float), meta

def load_trials(cache_dir: Path, sid: int, prefer_area: str) -> pd.DataFrame:
    p = cache_dir / f"{sid}_{prefer_area}.npz"
    if not p.exists():
        anyp = sorted(cache_dir.glob(f"{sid}_*.npz"))
        if not anyp: raise FileNotFoundError(f"No caches for sid {sid}")
        p = anyp[0]
    z = np.load(p, allow_pickle=True)
    return pd.read_json(StringIO(z["trials"].item()))

def detect_areas(cache_dir: Path, sid: int):
    hits = sorted(cache_dir.glob(f"{sid}_*.npz"))
    areas=[]
    for p in hits:
        a = p.stem.split("_",1)[1]
        if a in VALID: areas.append(a)
    return sorted(set(areas))

def time_axis(meta: dict):
    bs = float(meta.get("bin_size_s", 0.010))
    t0, t1 = meta["window_s"]
    return np.arange(t0 + bs/2, t1 + bs/2, bs)

# ---------------- smoothing / latency ----------------
def smooth_prob(p: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1: return p
    return uniform_filter1d(p, size=bins, mode="nearest")

def find_latency(t: np.ndarray, p: np.ndarray, thr: float, m: int, tmin: float) -> float:
    """Return earliest t >= tmin with m-consecutive bins p>=thr; NaN if none."""
    mask = (t >= tmin)
    idx_start = int(np.argmax(mask)) if mask.any() else 0
    if idx_start >= len(t): 
        return np.nan
    good = (p >= thr).astype(np.int32)
    cs = np.cumsum(good)
    # window sum length m at i..i+m-1 is cs[i+m-1] - cs[i-1]
    last_start = len(t) - m
    if last_start < idx_start:
        return np.nan
    for i in range(idx_start, last_start+1):
        if (cs[i + m - 1] - (cs[i - 1] if i > 0 else 0)) == m:
            return float(t[i])
    return np.nan

def latencies_from_probs(t: np.ndarray, P: np.ndarray, thr: float, hold_m: int, tmin: float) -> np.ndarray:
    """Vectorized over trials (loop trials to stay simple)."""
    return np.array([find_latency(t, P[i], thr, hold_m, tmin) for i in range(P.shape[0])], dtype=float)

# ---------------- decoders ----------------
def decode_category_probs(ZC: np.ndarray, C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Direction-invariant category decoding (hold-one-R-out across both categories) per time-bin.
    Returns p_true of shape (nTrials, nBins): probability of the true C label for each trial/time.
    """
    nT, nB, d = ZC.shape
    y = (C > 0).astype(int)           # 0/1 labels
    Rs = np.asarray(R, int)

    p = np.full((nT, nB), np.nan, dtype=float)

    for t in range(nB):
        Xt = ZC[:, t, :]              # (nT, d) at this time bin

        for r_hold in (1, 2, 3):
            test_idx = (Rs == r_hold)
            train_idx = ~test_idx

            # sanity checks
            if train_idx.sum() < 20 or test_idx.sum() < 8:
                continue
            if len(np.unique(y[train_idx])) < 2:
                continue

            clf = LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=2000, class_weight="balanced"
            )
            clf.fit(Xt[train_idx], y[train_idx])

            pro_pos = clf.predict_proba(Xt[test_idx])[:, 1]  # P(C=+1)
            yt = y[test_idx]                                 # true labels ∈ {0,1}

            # probability of the true label
            p_true = pro_pos.copy()
            p_true[yt == 0] = 1.0 - pro_pos[yt == 0]

            p[test_idx, t] = p_true

    return p

def decode_direction_probs(ZR: np.ndarray, C: np.ndarray, R: np.ndarray, nfold: int = 5) -> np.ndarray:
    """
    Within-category 3-way direction decoding per time-bin.
    Returns p_true shape (nTrials, nBins): probability of the TRUE direction per trial/time.
    """
    nT, nB, d = ZR.shape
    p = np.full((nT, nB), np.nan, dtype=float)
    Cint = (C > 0).astype(int)
    R01 = (np.asarray(R, int) - 1)  # {0,1,2}

    for t in range(nB):
        Xt = ZR[:, t, :]

        for cbin in (0, 1):
            sel = (Cint == cbin)
            if sel.sum() < 30:
                continue
            yc = R01[sel]
            # need all 3 classes present
            if len(np.unique(yc)) < 3:
                continue

            # safe n_splits
            counts = np.bincount(yc, minlength=3)
            min_class = counts.min()
            n_splits = max(2, min(nfold, int(min_class)))
            if n_splits < 2:
                continue

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
            Xc = Xt[sel]

            for tr_idx, te_idx in skf.split(Xc, yc):
                clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)
                clf.fit(Xc[tr_idx], yc[tr_idx])
                pro = clf.predict_proba(Xc[te_idx])  # (n,3)

                # probability of the true class
                true = yc[te_idx]
                p_true = pro[np.arange(len(te_idx)), true]

                glob = np.where(sel)[0][te_idx]
                p[glob, t] = p_true

    return p

# ---------------- plotting ----------------
def plot_pair_latencies(sid: int, A: str, B: str,
                        tC_A: np.ndarray, tC_B: np.ndarray,
                        tR_A: np.ndarray, tR_B: np.ndarray,
                        out_png: Path):
    dC = tC_B - tC_A
    dR = tR_B - tR_A
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for ax, d, ttl in zip(axes, (dC, dR), ("ΔT_C (B - A)", "ΔT_R (B - A)")):
        ok = np.isfinite(d)
        if ok.sum() < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(ttl)
            ax.axvline(0, color="k", lw=1)
            continue
        ax.hist(d[ok], bins=30, color="steelblue", alpha=0.8)
        med = np.nanmedian(d)
        prop_lead = float(np.mean(d[ok] > 0))  # B later than A if >0
        ax.axvline(0, color="k", lw=1)
        ax.axvline(med, color="crimson", lw=2, ls="--", label=f"median={med:.3f}s")
        ax.set_title(ttl); ax.set_xlabel("seconds"); ax.legend(frameon=False)
        ax.text(0.02, 0.95, f"P(B>A)={prop_lead:.2f}", transform=ax.transAxes, va="top")
    fig.suptitle(f"Trial-wise latency differences — sid={sid}  A={A}  B={B}")
    fig.tight_layout(rect=[0,0.03,1,0.95]); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------------- auto-threshold helpers ----------------
def parse_grid(spec: str) -> np.ndarray:
    """
    Parse grid string:
      '0.55:0.95:0.01'  -> np.arange(0.55, 0.95+1e-12, 0.01)
      '0.6,0.65,0.7'    -> np.array([...])
    """
    spec = spec.strip()
    if ":" in spec:
        a, b, c = map(float, spec.split(":"))
        # inclusive of the end
        n = max(1, int(round((b - a) / c)) + 1)
        return np.linspace(a, b, n)
    else:
        return np.array([float(x) for x in spec.split(",")])

def choose_threshold_for_session(p_list: list[tuple[np.ndarray,np.ndarray]],
                                 target: float, hold_m: int, tmin: float, grid: np.ndarray) -> float:
    """
    p_list: list of (P, t) over areas, where P is (nTrials x nBins) probabilities.
    Returns the HIGHEST threshold in grid with resolved fraction ≥ target (aggregated across areas).
    If none meets target, returns grid.min().
    """
    best_thr = grid.min()
    # evaluate from high to low to pick highest feasible
    for thr in sorted(grid, reverse=True):
        total = 0
        resolved = 0
        for P, t in p_list:
            lat = latencies_from_probs(t, P, thr, hold_m, tmin)
            m = np.isfinite(lat).sum()
            resolved += m
            total += len(lat)
        frac = (resolved / max(1, total))
        if frac >= target:
            best_thr = thr
            break
    return float(best_thr)

def choose_threshold_per_area(p: np.ndarray, t: np.ndarray,
                              target: float, hold_m: int, tmin: float, grid: np.ndarray) -> float:
    best_thr = grid.min()
    for thr in sorted(grid, reverse=True):
        lat = latencies_from_probs(t, p, thr, hold_m, tmin)
        if np.isfinite(lat).mean() >= target:
            best_thr = thr
            break
    return float(best_thr)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--axes_dir", type=Path, default=Path("results/session"))
    ap.add_argument("--cache_dir", type=Path, default=Path("results/caches"))
    ap.add_argument("--axes_tag", type=str, default="", help="Read axes under results/session/<sid>/<tag>/")
    # thresholds & smoothing
    ap.add_argument("--thrC", type=float, default=0.75)
    ap.add_argument("--thrR", type=float, default=0.60)
    ap.add_argument("--hold_m", type=int, default=3, help="Consecutive bins ≥ threshold to accept crossing")
    ap.add_argument("--tmin", type=float, default=0.0, help="Earliest time (s) to allow crossings")
    ap.add_argument("--smooth_bins", type=int, default=3, help="Temporal smoothing of p(t) with boxcar")
    # AUTO threshold
    ap.add_argument("--auto_thr", action="store_true", help="Auto-pick thresholds to hit target resolved fraction")
    ap.add_argument("--auto_strategy", type=str, choices=["session","area"], default="session",
                    help="Auto threshold per 'session' (shared across areas) or per 'area'")
    ap.add_argument("--auto_targetC", type=float, default=0.60, help="Target resolved fraction for category")
    ap.add_argument("--auto_targetR", type=float, default=0.60, help="Target resolved fraction for direction")
    ap.add_argument("--auto_grid_C", type=str, default="0.55:0.95:0.01",
                    help="Grid for category threshold (start:stop:step or comma list)")
    ap.add_argument("--auto_grid_R", type=str, default="0.40:0.95:0.01",
                    help="Grid for direction threshold (start:stop:step or comma list)")
    # plotting / output
    ap.add_argument("--pairs_plots", action="store_true", default=True)
    ap.add_argument("--out_tag", type=str, default="latency", help="Write under results/session/<sid>/<tag>/")
    args = ap.parse_args()

    sid = args.sid
    areas = detect_areas(args.cache_dir, sid)
    if not areas:
        print(f"[info] No areas for sid={sid}"); return

    # choose a prefer_area for trials (any present)
    prefer = areas[0]
    trials = load_trials(args.cache_dir, sid, prefer_area=prefer)
    C = np.asarray(trials["C"], int)
    R = np.asarray(trials["R"], int)

    out_dir = args.axes_dir / f"{sid}" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- decode probabilities per area ----
    P_C = {}  # area -> (nT x nB)
    P_R = {}
    t_by_area = {}
    for area in areas:
        print(f"[info] sid={sid} area={area}: decoding per time-bin...")
        ZC, ZR, meta = load_axes(sid, area, args.axes_dir, args.axes_tag)
        t = time_axis(meta)
        t_by_area[area] = t

        # Category (hold-one-R-out)
        pC = decode_category_probs(ZC, C, R)
        if args.smooth_bins > 1:
            pC = np.apply_along_axis(smooth_prob, 1, pC, args.smooth_bins)
        P_C[area] = pC

        # Direction (within-category)
        pR = decode_direction_probs(ZR, C, R, nfold=5)
        if args.smooth_bins > 1:
            pR = np.apply_along_axis(smooth_prob, 1, pR, args.smooth_bins)
        P_R[area] = pR

    # ---- auto threshold selection (optional) ----
    thr_log = {"strategy": args.auto_strategy}

    if args.auto_thr:
        gridC = parse_grid(args.auto_grid_C)
        gridR = parse_grid(args.auto_grid_R)

        if args.auto_strategy == "session":
            # one threshold per feature across areas
            p_list_C = [(P_C[a], t_by_area[a]) for a in areas]
            p_list_R = [(P_R[a], t_by_area[a]) for a in areas]
            thrC_used = choose_threshold_for_session(p_list_C, args.auto_targetC, args.hold_m, args.tmin, gridC)
            thrR_used = choose_threshold_for_session(p_list_R, args.auto_targetR, args.hold_m, args.tmin, gridR)
            thrC_map = {a: thrC_used for a in areas}
            thrR_map = {a: thrR_used for a in areas}
            print(f"[auto] sid={sid} session-level thresholds → C={thrC_used:.3f}, R={thrR_used:.3f} "
                  f"(targets C={args.auto_targetC}, R={args.auto_targetR})")
            thr_log.update({"thrC_session": thrC_used, "thrR_session": thrR_used})
        else:
            # per-area thresholds
            thrC_map, thrR_map = {}, {}
            for a in areas:
                thrC_map[a] = choose_threshold_per_area(P_C[a], t_by_area[a], args.auto_targetC,
                                                        args.hold_m, args.tmin, gridC)
                thrR_map[a] = choose_threshold_per_area(P_R[a], t_by_area[a], args.auto_targetR,
                                                        args.hold_m, args.tmin, gridR)
                print(f"[auto] sid={sid} area={a} thresholds → C={thrC_map[a]:.3f}, R={thrR_map[a]:.3f} "
                      f"(targets C={args.auto_targetC}, R={args.auto_targetR})")
            thr_log.update({"thrC_per_area": thrC_map, "thrR_per_area": thrR_map})
    else:
        # fixed thresholds from CLI
        thrC_map = {a: float(args.thrC) for a in areas}
        thrR_map = {a: float(args.thrR) for a in areas}
        print(f"[fixed] sid={sid} thresholds → C={args.thrC:.3f}, R={args.thrR:.3f}")
        thr_log.update({"thrC_fixed": args.thrC, "thrR_fixed": args.thrR})

    # ---- latencies per area with chosen thresholds ----
    latC = {}
    latR = {}
    for area in areas:
        t = t_by_area[area]
        pC = P_C[area]; pR = P_R[area]
        thrC = thrC_map[area]; thrR = thrR_map[area]

        lat_C = latencies_from_probs(t, pC, thrC, args.hold_m, args.tmin)
        lat_R = latencies_from_probs(t, pR, thrR, args.hold_m, args.tmin)

        # save per-area vectors
        np.save(out_dir / f"latency_C_{area}.npy", lat_C)
        np.save(out_dir / f"latency_R_{area}.npy", lat_R)
        latC[area] = lat_C
        latR[area] = lat_R

        # diagnostics
        fracC = float(np.isfinite(lat_C).mean()) if lat_C.size else 0.0
        fracR = float(np.isfinite(lat_R).mean()) if lat_R.size else 0.0
        if np.any(np.isfinite(lat_C)):
            medC = float(np.nanmedian(lat_C))
            print(f"[ok] sid={sid} area={area}: C thr={thrC:.2f}  med={medC:.3f}s  resolved={fracC*100:.1f}%")
        else:
            print(f"[warn] sid={sid} area={area}: no C latencies  thr={thrC:.2f}  resolved={fracC*100:.1f}%")
        if np.any(np.isfinite(lat_R)):
            medR = float(np.nanmedian(lat_R))
            print(f"[ok] sid={sid} area={area}: R thr={thrR:.2f}  med={medR:.3f}s  resolved={fracR*100:.1f}%")
        else:
            print(f"[warn] sid={sid} area={area}: no R latencies  thr={thrR:.2f}  resolved={fracR*100:.1f}%")

    # ---- pairwise lead/lag plots ----
    if args.pairs_plots:
        for A in areas:
            for B in areas:
                if A == B: continue
                out_png = out_dir / f"latency_pairs_{A}to{B}.png"
                plot_pair_latencies(sid, A, B, latC[A], latC[B], latR[A], latR[B], out_png)

    # ---- summary CSV ----
    rows = []
    for A in areas:
        for B in areas:
            if A==B: continue
            dC = latC[B] - latC[A]
            dR = latR[B] - latR[A]
            rows.append({
                "sid": sid, "A": A, "B": B,
                "median_dC": float(np.nanmedian(dC)) if np.any(np.isfinite(dC)) else np.nan,
                "prop_B_after_A_C": float(np.nanmean(dC > 0)) if np.any(np.isfinite(dC)) else np.nan,
                "median_dR": float(np.nanmedian(dR)) if np.any(np.isfinite(dR)) else np.nan,
                "prop_B_after_A_R": float(np.nanmean(dR > 0)) if np.any(np.isfinite(dR)) else np.nan,
                "thrC_used_A": float(thrC_map[A]), "thrC_used_B": float(thrC_map[B]),
                "thrR_used_A": float(thrR_map[A]), "thrR_used_B": float(thrR_map[B]),
                "hold_m": args.hold_m, "tmin": args.tmin, "smooth_bins": args.smooth_bins,
                "auto_thr": bool(args.auto_thr),
                "auto_strategy": args.auto_strategy if args.auto_thr else "",
                "auto_targetC": args.auto_targetC if args.auto_thr else np.nan,
                "auto_targetR": args.auto_targetR if args.auto_thr else np.nan
            })
    pd.DataFrame(rows).to_csv(out_dir / "latency_summary.csv", index=False)

    # Save thresholds used
    (out_dir / "latency_thresholds.json").write_text(json.dumps(thr_log, indent=2))
    print(f"[done] sid={sid} -> {out_dir}")

if __name__ == "__main__":
    main()
