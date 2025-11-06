#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- helpers ----------

def list_sessions_with_min_areas(rct_root: str, min_areas: int) -> List[str]:
    sids = [s for s in os.listdir(rct_root) if s.isdigit() and os.path.isdir(os.path.join(rct_root, s))]
    keep = []
    for sid in sorted(sids):
        areas_dir = os.path.join(rct_root, sid, "areas")
        if not os.path.isdir(areas_dir):
            continue
        n_areas = sum(os.path.isdir(os.path.join(areas_dir, a)) for a in os.listdir(areas_dir))
        if n_areas >= min_areas:
            keep.append(sid)
    return keep

def gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0: return np.array([1.0], dtype=float)
    half = int(np.ceil(3.0*sigma_bins))
    x = np.arange(-half, half+1, dtype=float)
    k = np.exp(-0.5*(x/sigma_bins)**2)
    k /= k.sum()
    return k

def smooth_time(X: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Convolve along time (axis=1) with a Gaussian kernel for each unit, trial-wise."""
    if sigma_bins <= 0: return X
    k = gaussian_kernel(sigma_bins)
    T,B,U = X.shape
    Y = np.empty_like(X, dtype=float)
    for u in range(U):
        # apply_along_axis: convolve each trial's timecourse
        Y[:,:,u] = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, X[:,:,u])
    return Y

def binary_mod_fraction(ru: np.ndarray, m_pos: np.ndarray, m_neg: np.ndarray, win_idx: np.ndarray) -> float:
    """% modulation between two trial groups, max over time within window."""
    if win_idx.size == 0 or (not m_pos.any()) or (not m_neg.any()): return np.nan
    # mean across trials for each time in window
    mu_pos = np.nanmean(ru[m_pos][:, win_idx], axis=0)
    mu_neg = np.nanmean(ru[m_neg][:, win_idx], axis=0)
    diff = np.abs(mu_pos - mu_neg)  # (T_win,)
    return float(100.0*np.nanmax(diff)) if diff.size else np.nan

def dir_mod_fraction(ru: np.ndarray, C: np.ndarray, R: np.ndarray, ok: np.ndarray, win_idx: np.ndarray) -> float:
    """
    Direction modulation within category: for each category and each time in window,
    compute max(mean_dir) - min(mean_dir) across the directions present; return
    the maximum span (in Hz) across categories/time, as a percentage value (x100).
    """
    if win_idx.size == 0: return np.nan
    best = np.nan
    for cs in (+1.0, -1.0):
        m_c = ok & np.isfinite(C) & (np.sign(C) == cs)
        if not m_c.any(): continue
        dirs = np.unique(R[m_c & np.isfinite(R)])
        if dirs.size < 2: continue
        # build mean per direction across time
        vals = []
        for dcode in dirs:
            md = m_c & (R == dcode)
            if not md.any(): continue
            vals.append(np.nanmean(ru[md][:, win_idx], axis=0))  # (T_win,)
        if not vals: continue
        M = np.vstack(vals)  # (n_dir, T_win)
        span = np.nanmax(M, axis=0) - np.nanmin(M, axis=0)      # (T_win,)
        m = float(np.nanmax(span))
        if (not np.isfinite(best)) or (m > best): best = m
    return best*100.0 if np.isfinite(best) else np.nan

# ---------- main work ----------

def run_selectivity(rct_root: str,
                    out_root: str,
                    align: str,
                    min_areas: int,
                    orientation: str,
                    win: Tuple[float,float],
                    smooth_sigma_ms: float,
                    min_rate_hz: float):
    """
    Aggregate single-unit modulation histograms across sessions and areas.
    Uses caches in out_root/<align>/<sid>/caches/area_*.npz
    """
    sids = list_sessions_with_min_areas(rct_root, min_areas)
    if not sids:
        raise SystemExit(f"No sessions with ≥{min_areas} areas under {rct_root}")

    out_dir = os.path.join(out_root, "qc", "selectivity", align)
    os.makedirs(out_dir, exist_ok=True)

    mod_cat, mod_dir, mod_sacc, mod_or = [], [], [], []
    units_kept = 0

    for sid in sids:
        caches_dir = os.path.join(out_root, align, sid, "caches")
        if not os.path.isdir(caches_dir):
            continue  # no caches for this align/session
        for fname in sorted(os.listdir(caches_dir)):
            if not (fname.startswith("area_") and fname.endswith(".npz")):
                continue
            d = np.load(os.path.join(caches_dir, fname), allow_pickle=True)
            time = d["time"].astype(float)
            X = d["X"].astype(float)  # (trials,bins,units)
            C = d.get("lab_C", np.full(X.shape[0], np.nan)).astype(float)
            R = d.get("lab_R", np.full(X.shape[0], np.nan)).astype(float)
            S = d.get("lab_S", np.full(X.shape[0], np.nan)).astype(float)
            OR = d.get("lab_orientation", np.array(["unknown"]*X.shape[0], dtype=object))

            # choose window bins
            idx = np.where((time >= win[0]) & (time <= win[1]))[0]
            if idx.size == 0:
                continue

            # correct-only already encoded by caches; still guard if present
            ok = np.ones(X.shape[0], dtype=bool)
            if "lab_is_correct" in d:
                ok &= d["lab_is_correct"].astype(bool)
            if orientation in ("vertical","horizontal") and "lab_orientation" in d:
                ok &= (OR.astype(str) == orientation)

            # smoothing and Hz conversion
            bin_s = float(json.loads(d["meta"].item()).get("bin_s", (time[1]-time[0] if time.size>1 else 0.010)))
            sigma_bins = (smooth_sigma_ms/1000.0) / (bin_s if bin_s>0 else 1.0)
            rate = smooth_time(X, sigma_bins) / (bin_s if bin_s>0 else 1.0)  # Hz

            U = rate.shape[2]
            for u in range(U):
                ru = rate[:,:,u]  # (trials,bins)

                # min peak rate in window
                peak = float(np.nanmax(np.nanmean(ru[:, idx], axis=0)))
                if (not np.isfinite(peak)) or (peak < min_rate_hz):
                    continue

                # masks
                mCpos = ok & np.isfinite(C) & (np.sign(C) == +1)
                mCneg = ok & np.isfinite(C) & (np.sign(C) == -1)
                mSpos = ok & np.isfinite(S) & (np.sign(S) == +1)
                mSneg = ok & np.isfinite(S) & (np.sign(S) == -1)

                mf_cat = binary_mod_fraction(ru, mCpos, mCneg, idx)
                mf_sacc = binary_mod_fraction(ru, mSpos, mSneg, idx) if mSpos.any() and mSneg.any() else np.nan
                mf_dir  = dir_mod_fraction(ru, C, R, ok, idx)

                if np.isfinite(mf_cat):  mod_cat.append(mf_cat)
                if np.isfinite(mf_dir):  mod_dir.append(mf_dir)
                if np.isfinite(mf_sacc): mod_sacc.append(mf_sacc)

                # orientation modulation only if both present in this session
                if "lab_orientation" in d:
                    mV = ok & (OR.astype(str) == "vertical")
                    mH = ok & (OR.astype(str) == "horizontal")
                    if mV.any() and mH.any():
                        mf_or = binary_mod_fraction(ru, mV, mH, idx)
                        if np.isfinite(mf_or): mod_or.append(mf_or)

                units_kept += 1

    # to arrays
    def clean(a): 
        a = np.asarray(a, dtype=float)
        return a[np.isfinite(a)]
    Cvals = clean(mod_cat); Rvals = clean(mod_dir)
    Svals = clean(mod_sacc); Ovals = clean(mod_or)

    # plot
    fig, axes = plt.subplots(2,2, figsize=(10,6))
    panels = [(Cvals,"Category"), (Rvals,"Direction (within C)"),
              (Svals,"Saccade direction"), (Ovals,"Target orientation")]
    bins = np.linspace(0,100,21)
    for ax,(vals,title) in zip(axes.ravel(), panels):
        ax.hist(vals, bins=bins, color="C0", alpha=0.85, edgecolor="none")
        ax.set_title(f"{title} modulation [%]  (N={len(vals)})")
        ax.set_xlabel("Modulation [%]"); ax.set_ylabel("Units")
    fig.suptitle(f"Single-unit modulation in window {win[0]:.2f}–{win[1]:.2f}s  (align={align}, orientation={orientation}); units kept={units_kept}")
    fig.tight_layout(rect=[0,0,1,0.95])

    out_path = os.path.join(out_dir, f"selectivity_hist_{orientation}_minA{min_areas}.pdf")
    fig.savefig(out_path); fig.savefig(out_path.replace(".pdf",".png"), dpi=300)
    plt.close(fig)
    print(f"[ok] wrote {out_path} (+ PNG).")

# ---------- CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single-unit modulation histograms (stim/sacc windows).")
    ap.add_argument("--root", default=os.environ.get("PAPER_DATA",""),
                    help="RCT root (with <sid>/areas)")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["sacc","stim"], default="sacc")
    ap.add_argument("--min_areas", type=int, default=2)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="pooled")
    ap.add_argument("--sacc_win", nargs=2, type=float, metavar=("START","END"),
                    default=[-0.40, 0.10], help="Saccade-aligned window (s)")
    ap.add_argument("--stim_win", nargs=2, type=float, metavar=("START","END"),
                    default=[-0.10, 0.10], help="Stimulus-aligned window (s)")
    ap.add_argument("--smooth_sigma_ms", type=float, default=20.0)
    ap.add_argument("--min_rate_hz", type=float, default=10.0)
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("Provide --root or set $PAPER_DATA")

    win = tuple(args.sacc_win) if args.align == "sacc" else tuple(args.stim_win)
    run_selectivity(
        rct_root=args.root,
        out_root=args.out_root,
        align=args.align,
        min_areas=args.min_areas,
        orientation=args.orientation,
        win=win,
        smooth_sigma_ms=args.smooth_sigma_ms,
        min_rate_hz=args.min_rate_hz,
    )
