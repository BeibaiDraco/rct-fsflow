#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_window(s: str) -> Tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def load_npz(p: Path) -> Dict:
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta" in out and not isinstance(out["meta"], dict):
        try:
            out["meta"] = json.loads(out["meta"].item())
        except Exception:
            pass
    return out


def list_areas(cache_dir: Path):
    return sorted([p.name[5:-4] for p in cache_dir.glob("area_*.npz")])


def pick_area(areas, key: str) -> str:
    hits = [a for a in areas if key.upper() in a.upper()]
    if not hits:
        raise SystemExit(f"No area containing '{key}' found. Areas={areas}")
    return hits[0]


def cache_path(out_root: Path, align: str, sid: str, area: str) -> Path:
    return out_root / align / sid / "caches" / f"area_{area}.npz"


def axis_path(out_root: Path, align: str, sid: str, axes_tag: str, area: str) -> Path:
    return out_root / align / sid / "axes" / axes_tag / f"axes_{area}.npz"


def trial_mask(cache: Dict, orientation: str, pt_min_ms: float | None) -> np.ndarray:
    N = cache["Z"].shape[0]
    keep = np.ones(N, dtype=bool)

    keep &= cache.get("lab_is_correct", np.ones(N, dtype=bool)).astype(bool)

    if orientation != "pooled" and "lab_orientation" in cache:
        keep &= (cache["lab_orientation"].astype(str) == orientation)

    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))

    C = cache.get("lab_C", np.full(N, np.nan)).astype(float)
    keep &= np.isfinite(C)

    return keep


def project_1d(Z: np.ndarray, s: np.ndarray) -> np.ndarray:
    # Z: (N,B,U), s: (U,)
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size != Z.shape[2]:
        raise ValueError(f"Axis dim {s.size} != n_units {Z.shape[2]}")
    return np.tensordot(Z, s, axes=([2], [0]))  # (N,B)


def gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    half = int(np.ceil(3 * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k


def smooth_timewise(Y: np.ndarray, sigma_bins: float) -> np.ndarray:
    # Y: (N,B)
    if sigma_bins <= 0:
        return Y
    k = gaussian_kernel(sigma_bins)
    out = np.empty_like(Y, dtype=float)
    for n in range(Y.shape[0]):
        out[n] = np.convolve(Y[n], k, mode="same")
    return out


def calculate_threshold(
    e: np.ndarray,              # (B,)
    time: np.ndarray,           # (B,)
    baseline: Tuple[float,float],
    k_sigma: float,
) -> float:
    """Calculate threshold for a trial: mu + k_sigma * sd"""
    bmask = (time >= baseline[0]) & (time <= baseline[1])
    if not np.any(bmask):
        return np.nan
    mu = float(np.nanmean(e[bmask]))
    sd = float(np.nanstd(e[bmask], ddof=1))
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1e-9
    return mu + k_sigma * sd


def onset_time_per_trial(
    e: np.ndarray,              # (B,)
    time: np.ndarray,           # (B,)
    baseline: Tuple[float,float],
    search: Tuple[float,float],
    k_sigma: float,
    runlen: int,
) -> float:
    # baseline stats per trial
    bmask = (time >= baseline[0]) & (time <= baseline[1])
    if not np.any(bmask):
        return np.nan
    mu = float(np.nanmean(e[bmask]))
    sd = float(np.nanstd(e[bmask], ddof=1))
    if not np.isfinite(sd) or sd <= 1e-9:
        sd = 1e-9
    thr = mu + k_sigma * sd

    smask = (time >= search[0]) & (time <= search[1])
    idx = np.where(smask)[0]
    if idx.size == 0:
        return np.nan

    for i in idx:
        j = i + runlen
        if j > e.size:
            break
        seg = e[i:j]
        if np.all(np.isfinite(seg)) and np.all(seg > thr):
            return float(time[i])
    return np.nan


def main():
    ap = argparse.ArgumentParser(description="Per-trial onset time of category evidence in FEF vs LIP (scatter).")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--align", choices=["stim"], default="stim")
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--axes_tag", default="axes_sweep-stim-pooled",
                    help="Axes tag to read sC from (default: axes_sweep-stim-pooled)")
    ap.add_argument("--baseline", default="-0.20:0.00",
                    help="Baseline window (sec) for per-trial threshold (default: -0.20:0.00)")
    ap.add_argument("--search", default="0.00:0.50",
                    help="Search window (sec) for onset (default: 0.00:0.50)")
    ap.add_argument("--k_sigma", type=float, default=2.0,
                    help="Threshold = baseline mean + k_sigma*baseline std (default: 2.0)")
    ap.add_argument("--runlen", type=int, default=3,
                    help="Consecutive bins above threshold (default: 3)")
    ap.add_argument("--smooth_ms", type=float, default=20.0,
                    help="Gaussian smoothing sigma in ms applied to evidence traces before onset (default: 20)")
    ap.add_argument("--tag", default="trialonset_v1",
                    help="Output subfolder name (default: trialonset_v1)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    sid = args.sid
    align = args.align

    baseline = parse_window(args.baseline)
    search = parse_window(args.search)

    cache_dir = out_root / align / sid / "caches"
    areas = list_areas(cache_dir)
    fef = pick_area(areas, "FEF")
    lip = pick_area(areas, "LIP")

    cache_fef = load_npz(cache_path(out_root, align, sid, fef))
    cache_lip = load_npz(cache_path(out_root, align, sid, lip))

    keep = trial_mask(cache_fef, args.orientation, args.pt_min_ms) & trial_mask(cache_lip, args.orientation, args.pt_min_ms)
    if keep.sum() < 60:
        raise SystemExit(f"Too few trials after filtering: N={keep.sum()}")

    # load axes
    axes_fef = load_npz(axis_path(out_root, align, sid, args.axes_tag, fef))
    axes_lip = load_npz(axis_path(out_root, align, sid, args.axes_tag, lip))
    sC_fef = axes_fef.get("sC", np.array([])).ravel()
    sC_lip = axes_lip.get("sC", np.array([])).ravel()
    if sC_fef.size == 0 or sC_lip.size == 0:
        raise SystemExit("Missing sC in FEF or LIP axes.")

    time = cache_fef["time"].astype(float)
    Zf = cache_fef["Z"][keep].astype(float)  # (N,B,U)
    Zl = cache_lip["Z"][keep].astype(float)
    C = cache_fef["lab_C"][keep].astype(float)
    C = np.sign(C)
    if np.unique(C).size < 2:
        raise SystemExit("Need both categories in filtered trials.")

    # projections
    yf = project_1d(Zf, sC_fef)  # (N,B)
    yl = project_1d(Zl, sC_lip)  # (N,B)

    # signed evidence (positive = correct category direction)
    ef = (C[:, None] * yf)
    el = (C[:, None] * yl)

    # smooth
    dt = float(np.nanmedian(np.diff(time)))
    sigma_bins = (args.smooth_ms / 1000.0) / dt if dt > 0 else 0.0
    ef = smooth_timewise(ef, sigma_bins)
    el = smooth_timewise(el, sigma_bins)

    # calculate thresholds
    thr_fef = np.array([calculate_threshold(ef[i], time, baseline, args.k_sigma)
                        for i in range(ef.shape[0])], dtype=float)
    thr_lip = np.array([calculate_threshold(el[i], time, baseline, args.k_sigma)
                        for i in range(el.shape[0])], dtype=float)

    # onsets
    t_fef = np.array([onset_time_per_trial(ef[i], time, baseline, search, args.k_sigma, args.runlen)
                      for i in range(ef.shape[0])], dtype=float)
    t_lip = np.array([onset_time_per_trial(el[i], time, baseline, search, args.k_sigma, args.runlen)
                      for i in range(el.shape[0])], dtype=float)

    # keep trials where both exist
    good = np.isfinite(t_fef) & np.isfinite(t_lip)
    n_good = int(good.sum())
    print(f"[info] kept {n_good}/{keep.sum()} trials with defined onsets")
    
    # print threshold statistics
    print(f"[threshold] FEF: median={np.nanmedian(thr_fef):.4f}, mean={np.nanmean(thr_fef):.4f}, std={np.nanstd(thr_fef):.4f}")
    print(f"[threshold] LIP: median={np.nanmedian(thr_lip):.4f}, mean={np.nanmean(thr_lip):.4f}, std={np.nanstd(thr_lip):.4f}")

    out_dir = out_root / align / sid / "trialtiming" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{fef}_vs_{lip}_onset_{args.orientation}.npz"
    np.savez_compressed(
        out_npz,
        time=time,
        t_fef=t_fef,
        t_lip=t_lip,
        good_mask=good.astype(int),
        meta=dict(
            sid=sid, align=align, orientation=args.orientation,
            axes_tag=args.axes_tag,
            baseline=baseline,
            search=search,
            k_sigma=float(args.k_sigma),
            runlen=int(args.runlen),
            smooth_ms=float(args.smooth_ms),
            pt_min_ms=args.pt_min_ms,
            n_trials=int(keep.sum()),
            n_good=n_good,
            fef=fef, lip=lip,
        )
    )
    print(f"[ok] wrote {out_npz}")

    # scatter plot
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    x = t_fef[good] * 1000.0
    y = t_lip[good] * 1000.0
    ax.plot(x, y, "k.", ms=3, alpha=0.6)
    lo = np.nanmin(np.r_[x, y])
    hi = np.nanmax(np.r_[x, y])
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, label="y=x")
    ax.set_xlabel("FEF onset time (ms)")
    ax.set_ylabel("LIP onset time (ms)")
    ax.set_title(f"{sid} {args.orientation} | onset (k={args.k_sigma}, runlen={args.runlen}, smooth={args.smooth_ms}ms)\nN={n_good}")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_npz.with_suffix(".png"), dpi=300)
    fig.savefig(out_npz.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] wrote {out_npz.with_suffix('.png')} and .pdf")

    # quick summary stats
    if n_good >= 10:
        dt_ms = (t_lip[good] - t_fef[good]) * 1000.0
        print(f"[summary] median(LIP-FEF) = {np.nanmedian(dt_ms):.1f} ms | mean = {np.nanmean(dt_ms):.1f} ms")


if __name__ == "__main__":
    main()
