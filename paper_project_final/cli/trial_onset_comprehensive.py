#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
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


def pick_area(areas, key: str) -> Optional[str]:
    hits = [a for a in areas if key.upper() in a.upper()]
    if not hits:
        return None
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
    # Z: (N,B,U), s: (U,) or (U,K)
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1)
    else:
        # For multi-dimensional (e.g., sR), use first dimension
        s = s[:, 0].reshape(-1)
    
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


def get_monkey(sid: str) -> str:
    """Return 'M' for sessions starting with 2020, 'S' for 2023."""
    if sid.startswith("2020"):
        return "M"
    elif sid.startswith("2023"):
        return "S"
    else:
        return "Unknown"


def process_one_session(
    out_root: Path,
    align: str,
    sid: str,
    orientation: str,
    pt_min_ms: float,
    axes_tag: str,
    baseline: Tuple[float, float],
    search: Tuple[float, float],
    k_sigma: float,
    runlen: int,
    smooth_ms: float,
    feature: str,  # "C" for category, "R" for direction
    area1: str,    # First area (e.g., "FEF", "LIP", "SC")
    area2: str,    # Second area
    tag: str,
) -> Optional[Dict]:
    """Process one session for one pair and feature. Returns dict with results or None if failed."""
    
    cache_dir = out_root / align / sid / "caches"
    if not cache_dir.exists():
        return None
    
    areas = list_areas(cache_dir)
    a1 = pick_area(areas, area1)
    a2 = pick_area(areas, area2)
    
    if a1 is None or a2 is None:
        return None
    
    try:
        cache1 = load_npz(cache_path(out_root, align, sid, a1))
        cache2 = load_npz(cache_path(out_root, align, sid, a2))
        
        keep = trial_mask(cache1, orientation, pt_min_ms) & trial_mask(cache2, orientation, pt_min_ms)
        if keep.sum() < 60:
            return None
        
        # Load axes
        axes1 = load_npz(axis_path(out_root, align, sid, axes_tag, a1))
        axes2 = load_npz(axis_path(out_root, align, sid, axes_tag, a2))
        
        if feature == "C":
            s1 = axes1.get("sC", np.array([]))
            s2 = axes2.get("sC", np.array([]))
        elif feature == "R":
            s1 = axes1.get("sR", np.array([[]]))
            s2 = axes2.get("sR", np.array([[]]))
            if s1.size == 0 or s2.size == 0:
                return None
        else:
            return None
        
        s1 = s1.ravel() if s1.ndim > 1 and s1.shape[1] == 1 else (s1[:, 0] if s1.ndim > 1 else s1)
        s2 = s2.ravel() if s2.ndim > 1 and s2.shape[1] == 1 else (s2[:, 0] if s2.ndim > 1 else s2)
        
        if s1.size == 0 or s2.size == 0:
            return None
        
        time = cache1["time"].astype(float)
        Z1 = cache1["Z"][keep].astype(float)  # (N,B,U)
        Z2 = cache2["Z"][keep].astype(float)
        
        if feature == "C":
            C = cache1["lab_C"][keep].astype(float)
            C = np.sign(C)
            if np.unique(C).size < 2:
                return None
        elif feature == "R":
            R = cache1.get("lab_R", np.full(len(keep), np.nan))[keep].astype(float)
            if not np.any(np.isfinite(R)):
                return None
            # For direction, encode as sign or use a binary encoding
            # Round to nearest integer for discrete directions
            R_rounded = np.round(R)
            R_vals = np.unique(R_rounded[np.isfinite(R_rounded)])
            if R_vals.size < 2:
                return None
            # Use sign of centered R (positive vs negative directions)
            R_centered = R_rounded - np.nanmedian(R_rounded)
            R_sign = np.sign(R_centered)
            # If all signs are same, try alternative encoding
            if np.unique(R_sign[np.isfinite(R_sign)]).size < 2:
                # Try splitting by median
                R_median = np.nanmedian(R_rounded)
                R_sign = np.where(R_rounded > R_median, 1, -1)
                if np.unique(R_sign[np.isfinite(R_sign)]).size < 2:
                    return None
        
        # Projections
        y1 = project_1d(Z1, s1)  # (N,B)
        y2 = project_1d(Z2, s2)  # (N,B)
        
        # Signed evidence
        if feature == "C":
            e1 = (C[:, None] * y1)
            e2 = (C[:, None] * y2)
        else:  # R
            e1 = (R_sign[:, None] * y1)
            e2 = (R_sign[:, None] * y2)
        
        # Smooth
        dt = float(np.nanmedian(np.diff(time)))
        sigma_bins = (smooth_ms / 1000.0) / dt if dt > 0 else 0.0
        e1 = smooth_timewise(e1, sigma_bins)
        e2 = smooth_timewise(e2, sigma_bins)
        
        # Calculate thresholds
        thr1 = np.array([calculate_threshold(e1[i], time, baseline, k_sigma)
                         for i in range(e1.shape[0])], dtype=float)
        thr2 = np.array([calculate_threshold(e2[i], time, baseline, k_sigma)
                         for i in range(e2.shape[0])], dtype=float)
        
        # Onsets
        t1 = np.array([onset_time_per_trial(e1[i], time, baseline, search, k_sigma, runlen)
                       for i in range(e1.shape[0])], dtype=float)
        t2 = np.array([onset_time_per_trial(e2[i], time, baseline, search, k_sigma, runlen)
                       for i in range(e2.shape[0])], dtype=float)
        
        # Keep trials where both exist
        good = np.isfinite(t1) & np.isfinite(t2)
        n_good = int(good.sum())
        
        if n_good < 10:
            return None
        
        return {
            "sid": sid,
            "area1": a1,
            "area2": a2,
            "feature": feature,
            "t1": t1,
            "t2": t2,
            "good": good,
            "n_good": n_good,
            "thr1": thr1,
            "thr2": thr2,
        }
    except Exception as e:
        print(f"[warning] Failed to process {sid} {area1}-{area2} {feature}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Comprehensive trial onset analysis: category and direction, all pairs, all sessions.")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--sid_list", default="sid_list.txt", help="File with list of session IDs")
    ap.add_argument("--align", choices=["stim"], default="stim")
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--axes_tag", default="axes_sweep-stim-vertical",
                    help="Axes tag to read axes from")
    ap.add_argument("--baseline", default="-0.20:0.00",
                    help="Baseline window (sec) for per-trial threshold")
    ap.add_argument("--search", default="0.00:0.60",
                    help="Search window (sec) for onset")
    ap.add_argument("--k_sigma", type=float, default=3.0,
                    help="Threshold = baseline mean + k_sigma*baseline std")
    ap.add_argument("--runlen", type=int, default=4,
                    help="Consecutive bins above threshold")
    ap.add_argument("--smooth_ms", type=float, default=20.0,
                    help="Gaussian smoothing sigma in ms")
    ap.add_argument("--tag", default="trialonset_comprehensive",
                    help="Output subfolder name")
    args = ap.parse_args()
    
    out_root = Path(args.out_root)
    align = args.align
    baseline = parse_window(args.baseline)
    search = parse_window(args.search)
    
    # Read session IDs
    sid_list_path = Path(args.sid_list)
    if not sid_list_path.exists():
        raise SystemExit(f"Session list file not found: {sid_list_path}")
    
    sids = []
    with open(sid_list_path) as f:
        for line in f:
            sid = line.strip()
            if sid and not sid.startswith("#"):
                sids.append(sid)
    
    print(f"[info] Processing {len(sids)} sessions")
    
    # Define pairs and features
    # Note: SC areas are named MSC or SSC depending on monkey, but we search for "SC"
    pairs = [
        ("SC", "LIP"),
        ("SC", "FEF"),
        ("FEF", "LIP"),
    ]
    features = ["C", "R"]  # Category and Direction
    
    # Process all sessions
    all_results = {}  # (feature, area1, area2) -> list of results
    
    for feature in features:
        for area1, area2 in pairs:
            key = (feature, area1, area2)
            all_results[key] = []
            
            for sid in sids:
                result = process_one_session(
                    out_root, align, sid, args.orientation, args.pt_min_ms,
                    args.axes_tag, baseline, search, args.k_sigma, args.runlen,
                    args.smooth_ms, feature, area1, area2, args.tag
                )
                if result is not None:
                    all_results[key].append(result)
                    print(f"[{sid}] {area1}-{area2} {feature}: {result['n_good']} trials")
    
    # Create output directory
    out_dir = out_root / align / "trialtiming" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate by monkey and create summary plots
    for feature in features:
        feature_name = "category" if feature == "C" else "direction"
        
        for area1, area2 in pairs:
            key = (feature, area1, area2)
            results = all_results[key]
            
            if len(results) == 0:
                continue
            
            # Aggregate by monkey
            monkey_data = {"M": ([], [], []), "S": ([], [], [])}
            
            for result in results:
                sid = result["sid"]
                monkey = get_monkey(sid)
                if monkey not in monkey_data:
                    continue
                
                t1 = result["t1"][result["good"]] * 1000.0  # Convert to ms
                t2 = result["t2"][result["good"]] * 1000.0
                
                valid = np.isfinite(t1) & np.isfinite(t2)
                if valid.sum() > 0:
                    monkey_data[monkey][0].append(t1[valid])
                    monkey_data[monkey][1].append(t2[valid])
                    monkey_data[monkey][2].append(sid)
            
            # Create plots for each monkey
            for monkey, (t1_list, t2_list, sids_list) in monkey_data.items():
                if len(t1_list) == 0:
                    continue
                
                t1_all = np.concatenate(t1_list)
                t2_all = np.concatenate(t2_list)
                n_trials = len(t1_all)
                n_sessions = len(set(sids_list))
                
                # Calculate statistics
                dt_ms = t2_all - t1_all
                median_dt = np.nanmedian(dt_ms)
                mean_dt = np.nanmean(dt_ms)
                
                print(f"[{monkey}] {area1}-{area2} {feature_name}: {n_trials} trials from {n_sessions} sessions | median({area2}-{area1}) = {median_dt:.1f} ms | mean = {mean_dt:.1f} ms")
                
                # Create plot
                fig = plt.figure(figsize=(7.0, 6.5))
                ax = fig.add_subplot(1, 1, 1)
                
                # Scatter plot
                ax.plot(t1_all, t2_all, "k.", ms=2, alpha=0.4, label=f"N={n_trials} trials")
                
                # Mark mean and median (without legend labels to avoid overlap)
                mean_t1 = np.nanmean(t1_all)
                mean_t2 = np.nanmean(t2_all)
                median_t1 = np.nanmedian(t1_all)
                median_t2 = np.nanmedian(t2_all)
                
                ax.plot(mean_t1, mean_t2, "ro", ms=10, markerfacecolor="red", 
                        markeredgecolor="darkred", markeredgewidth=2, zorder=5)
                ax.plot(median_t1, median_t2, "bs", ms=10, markerfacecolor="blue", 
                        markeredgecolor="darkblue", markeredgewidth=2, zorder=5)
                
                # Diagonal line
                lo = np.nanmin(np.r_[t1_all, t2_all])
                hi = np.nanmax(np.r_[t1_all, t2_all])
                ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x", alpha=0.7)
                
                # Labels and title
                monkey_name = "Monkey M" if monkey == "M" else "Monkey S"
                area_prefix = "M" if monkey == "M" else "S"
                # SC always stays as "SC", FEF and LIP get monkey prefix
                a1_name = "SC" if area1 == "SC" else f"{area_prefix}{area1}"
                a2_name = "SC" if area2 == "SC" else f"{area_prefix}{area2}"
                
                ax.set_xlabel(f"{a1_name} onset time (ms)", fontsize=12)
                ax.set_ylabel(f"{a2_name} onset time (ms)", fontsize=12)
                
                # Title with mean and median values
                title = f"{monkey_name} ({a1_name} vs {a2_name}) - {feature_name}\n"
                title += f"{n_sessions} sessions, {n_trials} trials\n"
                title += f"mean: ({mean_t1:.1f}, {mean_t2:.1f}) ms | median: ({median_t1:.1f}, {median_t2:.1f}) ms"
                ax.set_title(title, fontsize=12)
                
                # Add statistics text
                stats_text = f"median({a2_name}-{a1_name}) = {median_dt:.1f} ms\n"
                stats_text += f"mean({a2_name}-{a1_name}) = {mean_dt:.1f} ms"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.legend(frameon=False, loc="upper left", fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                fig.tight_layout()
                
                # Save
                out_png = out_dir / f"monkey_{monkey}_{area1}_vs_{area2}_{feature_name}_summary.png"
                out_pdf = out_dir / f"monkey_{monkey}_{area1}_vs_{area2}_{feature_name}_summary.pdf"
                fig.savefig(out_png, dpi=300)
                fig.savefig(out_pdf)
                plt.close(fig)
                print(f"[ok] wrote {out_png} and {out_pdf}")
                
                # Save data
                out_npz = out_dir / f"monkey_{monkey}_{area1}_vs_{area2}_{feature_name}_summary.npz"
                np.savez_compressed(
                    out_npz,
                    t1_ms=t1_all,
                    t2_ms=t2_all,
                    dt_ms=dt_ms,
                    sids=np.array(sids_list),
                    meta=dict(
                        monkey=monkey,
                        area1=area1,
                        area2=area2,
                        feature=feature,
                        feature_name=feature_name,
                        n_trials=n_trials,
                        n_sessions=n_sessions,
                        median_dt_ms=float(median_dt),
                        mean_dt_ms=float(mean_dt),
                    )
                )
                print(f"[ok] wrote {out_npz}")


if __name__ == "__main__":
    main()

