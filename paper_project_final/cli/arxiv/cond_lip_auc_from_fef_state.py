#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def parse_window(s: str) -> Tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def load_npz(path: Path) -> Dict:
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta" in out and not isinstance(out["meta"], dict):
        try:
            out["meta"] = json.loads(out["meta"].item())
        except Exception:
            pass
    return out


def find_area(cache_dir: Path, key: str) -> str:
    """Pick an area name containing key (e.g., 'FEF', 'LIP')."""
    areas = sorted([p.name[5:-4] for p in cache_dir.glob("area_*.npz")])
    hits = [a for a in areas if key.upper() in a.upper()]
    if not hits:
        raise SystemExit(f"No area containing '{key}' found in {cache_dir}. Areas={areas}")
    return hits[0]


def axis_path(out_root: Path, align: str, sid: str, axes_tag: str, area: str) -> Path:
    return out_root / align / sid / "axes" / axes_tag / f"axes_{area}.npz"


def cache_path(out_root: Path, align: str, sid: str, area: str) -> Path:
    return out_root / align / sid / "caches" / f"area_{area}.npz"


def trial_mask(cache: Dict, orientation: str, pt_min_ms: float) -> np.ndarray:
    Z = cache["Z"]
    N = Z.shape[0]
    keep = np.ones(N, dtype=bool)
    if "lab_is_correct" in cache:
        keep &= cache["lab_is_correct"].astype(bool)
    if orientation != "pooled" and "lab_orientation" in cache:
        keep &= (cache["lab_orientation"].astype(str) == orientation)
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))
    # need labels
    if "lab_C" in cache:
        keep &= np.isfinite(cache["lab_C"].astype(float))
    if "lab_R" in cache:
        keep &= np.isfinite(cache["lab_R"].astype(float))
    return keep


def project_1d(cache: Dict, s: np.ndarray, keep: np.ndarray) -> np.ndarray:
    Z = cache["Z"][keep].astype(float)  # (N,B,U)
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size != Z.shape[2]:
        raise ValueError(f"Axis dim {s.size} != n_units {Z.shape[2]}")
    return np.tensordot(Z, s, axes=([2],[0]))  # (N,B)


def encode_CR(cache: Dict, keep: np.ndarray) -> np.ndarray:
    C = np.round(cache["lab_C"][keep].astype(float)).astype(int)
    R = np.round(cache["lab_R"][keep].astype(float)).astype(int)
    base = 1000
    return (C * base + R).astype(int)


def split_high_low_within_strata(dA: np.ndarray, strata: np.ndarray, min_per_stratum: int = 6):
    """
    For each stratum, split indices into equal-size low/high by median.
    Returns two index arrays (low_idx, high_idx) into dA.
    """
    low = []
    high = []
    for v in np.unique(strata):
        idx = np.where(strata == v)[0]
        if idx.size < min_per_stratum:
            continue
        order = np.argsort(dA[idx])
        half = idx.size // 2
        if half == 0:
            continue
        low.append(idx[order[:half]])
        high.append(idx[order[-half:]])
    if not low or not high:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(low), np.concatenate(high)


def auc_from_scores(scores: np.ndarray, C_pm1: np.ndarray) -> float:
    y = (C_pm1 > 0).astype(int)
    if np.unique(y).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return np.nan


def dprime_from_scores(scores: np.ndarray, C_pm1: np.ndarray) -> float:
    x1 = scores[C_pm1 > 0]
    x0 = scores[C_pm1 < 0]
    if x1.size < 5 or x0.size < 5:
        return np.nan
    m1, m0 = np.nanmean(x1), np.nanmean(x0)
    s = np.sqrt(0.5*(np.nanvar(x1, ddof=1) + np.nanvar(x0, ddof=1)))
    return float((m1 - m0) / s) if s > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description="Time-resolved LIP category encoding conditioned on early FEF state.")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--align", choices=["stim"], default="stim")
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical",
                    help="Trial subset to analyze (default: vertical)")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--axes_tag_A", default="axes_sweep-stim-pooled",
                    help="Axes tag for FEF (source) axis (default: axes_sweep-stim-pooled)")
    ap.add_argument("--axes_tag_B", default="axes_sweep-stim-pooled",
                    help="Axes tag for LIP (target) axis (default: axes_sweep-stim-pooled)")
    ap.add_argument("--winA", default="0.12:0.20",
                    help="Early FEF window in seconds (default: 0.12:0.20)")
    ap.add_argument("--min_per_stratum", type=int, default=6,
                    help="Min trials per (C,R) stratum to split (default: 6)")
    ap.add_argument("--perms", type=int, default=200,
                    help="Permutations for null on ΔAUC(t) (default: 200; set 0 to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="condAUC_v1",
                    help="Output subfolder name (default: condAUC_v1)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    sid = args.sid
    align = args.align
    ori = args.orientation
    winA = parse_window(args.winA)

    cache_dir = out_root / align / sid / "caches"
    A_area = find_area(cache_dir, "FEF")
    B_area = find_area(cache_dir, "LIP")

    cacheA = load_npz(cache_path(out_root, align, sid, A_area))
    cacheB = load_npz(cache_path(out_root, align, sid, B_area))

    keepA = trial_mask(cacheA, ori, args.pt_min_ms)
    keepB = trial_mask(cacheB, ori, args.pt_min_ms)
    if keepA.shape[0] != keepB.shape[0]:
        raise SystemExit("A and B caches have different #trials.")
    keep = keepA & keepB
    if keep.sum() < 50:
        raise SystemExit(f"Too few trials after filtering: N={keep.sum()}")

    # Load axes
    axesA = load_npz(axis_path(out_root, align, sid, args.axes_tag_A, A_area))
    axesB = load_npz(axis_path(out_root, align, sid, args.axes_tag_B, B_area))
    sA = axesA.get("sC", np.array([])).ravel()
    sB = axesB.get("sC", np.array([])).ravel()
    if sA.size == 0 or sB.size == 0:
        raise SystemExit("Missing sC axis in A or B axes tag.")

    # Projections
    YA = project_1d(cacheA, sA, keep)  # (N,B)
    YB = project_1d(cacheB, sB, keep)  # (N,B)
    time = cacheA["time"].astype(float)
    C = cacheA["lab_C"][keep].astype(float)

    # Early window on FEF
    wmaskA = (time >= winA[0]) & (time <= winA[1])
    if not np.any(wmaskA):
        raise SystemExit(f"No bins in winA={winA} on this time grid.")
    dA = np.nanmean(YA[:, wmaskA], axis=1)  # (N,)

    # Split high/low within (C,R)
    strata = encode_CR(cacheA, keep)
    low_idx, high_idx = split_high_low_within_strata(dA, strata, min_per_stratum=args.min_per_stratum)
    if low_idx.size < 30 or high_idx.size < 30:
        raise SystemExit(f"Too few trials after within-stratum split: low={low_idx.size}, high={high_idx.size}")

    # Compute LIP AUC(t), d'(t) for groups
    Bbins = time.size
    auc_all = np.full(Bbins, np.nan)
    auc_low = np.full(Bbins, np.nan)
    auc_high = np.full(Bbins, np.nan)
    d_all = np.full(Bbins, np.nan)
    d_low = np.full(Bbins, np.nan)
    d_high = np.full(Bbins, np.nan)

    for b in range(Bbins):
        auc_all[b] = auc_from_scores(YB[:, b], C)
        auc_low[b] = auc_from_scores(YB[low_idx, b], C[low_idx])
        auc_high[b] = auc_from_scores(YB[high_idx, b], C[high_idx])
        d_all[b] = dprime_from_scores(YB[:, b], C)
        d_low[b] = dprime_from_scores(YB[low_idx, b], C[low_idx])
        d_high[b] = dprime_from_scores(YB[high_idx, b], C[high_idx])

    delta_auc = auc_high - auc_low

    # Permutation null for delta_auc(t): shuffle within strata (preserve C,R balance)
    rng = np.random.default_rng(args.seed)
    null_mean = np.full(Bbins, np.nan)
    null_std = np.full(Bbins, np.nan)
    z_delta = np.full(Bbins, np.nan)

    if args.perms > 0:
        P = int(args.perms)
        deltas = np.full((P, Bbins), np.nan)

        # Precompute indices per stratum
        stratum_to_idx = {v: np.where(strata == v)[0] for v in np.unique(strata)}

        for p in range(P):
            low_p = []
            high_p = []
            for v, idx in stratum_to_idx.items():
                if idx.size < args.min_per_stratum:
                    continue
                perm = rng.permutation(idx)
                half = idx.size // 2
                if half == 0:
                    continue
                low_p.append(perm[:half])
                high_p.append(perm[-half:])
            if not low_p or not high_p:
                continue
            low_p = np.concatenate(low_p)
            high_p = np.concatenate(high_p)

            for b in range(Bbins):
                a_hi = auc_from_scores(YB[high_p, b], C[high_p])
                a_lo = auc_from_scores(YB[low_p, b], C[low_p])
                deltas[p, b] = a_hi - a_lo

        null_mean = np.nanmean(deltas, axis=0)
        null_std = np.nanstd(deltas, axis=0, ddof=1)
        good = np.isfinite(delta_auc) & np.isfinite(null_mean) & np.isfinite(null_std) & (null_std > 0)
        z_delta[good] = (delta_auc[good] - null_mean[good]) / null_std[good]

    # Save outputs
    out_dir = out_root / align / sid / "condenc" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{A_area}_to_{B_area}_condAUC_{ori}.npz"

    np.savez_compressed(
        out_npz,
        time=time,
        auc_all=auc_all, auc_low=auc_low, auc_high=auc_high,
        d_all=d_all, d_low=d_low, d_high=d_high,
        delta_auc=delta_auc,
        delta_auc_null_mean=null_mean,
        delta_auc_null_std=null_std,
        delta_auc_z=z_delta,
        meta=dict(
            sid=sid, align=align, orientation=ori,
            A_area=A_area, B_area=B_area,
            axes_tag_A=args.axes_tag_A, axes_tag_B=args.axes_tag_B,
            winA=winA, pt_min_ms=args.pt_min_ms,
            n_trials=int(keep.sum()),
            n_low=int(low_idx.size), n_high=int(high_idx.size),
            perms=int(args.perms),
            min_per_stratum=int(args.min_per_stratum),
        )
    )
    print(f"[ok] wrote {out_npz}")

    # Plot
    tms = time * 1000.0
    fig = plt.figure(figsize=(8.0, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvline(0, ls="--", c="k", lw=0.8)
    ax1.axvspan(winA[0]*1000, winA[1]*1000, color="0.9", alpha=0.5, label="FEF early window")
    ax1.plot(tms, auc_all, lw=2.0, label="LIP AUC(all)")
    ax1.plot(tms, auc_high, lw=2.0, label="LIP AUC(high FEF state)")
    ax1.plot(tms, auc_low, lw=2.0, label="LIP AUC(low FEF state)")
    ax1.set_ylabel("AUC(C | LIP)")
    ax1.set_title(f"{sid} {A_area}→{B_area} | ori={ori} | axesA={args.axes_tag_A} axesB={args.axes_tag_B}")
    ax1.legend(frameon=False, loc="lower right")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axvline(0, ls="--", c="k", lw=0.8)
    ax2.plot(tms, delta_auc, lw=2.0, label="ΔAUC = high - low")
    if args.perms > 0 and np.any(np.isfinite(null_mean)):
        ax2.plot(tms, null_mean, lw=1.5, ls=":", label="null mean")
        ax2.fill_between(tms, null_mean-null_std, null_mean+null_std, alpha=0.2, linewidth=0, label="null ±1σ")
    ax2.set_ylabel("ΔAUC")
    ax2.legend(frameon=False, loc="upper right")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axvline(0, ls="--", c="k", lw=0.8)
    if args.perms > 0:
        ax3.plot(tms, z_delta, lw=2.0, label="z(ΔAUC)")
        ax3.axhline(0, ls=":", c="k", lw=0.8)
        ax3.set_ylabel("Z")
        ax3.legend(frameon=False, loc="upper right")
    else:
        ax3.plot(tms, d_all, lw=2.0, label="LIP d'(all)")
        ax3.plot(tms, d_high, lw=2.0, label="LIP d'(high)")
        ax3.plot(tms, d_low, lw=2.0, label="LIP d'(low)")
        ax3.set_ylabel("d'")
        ax3.legend(frameon=False, loc="upper right")

    ax3.set_xlabel("Time (ms)")
    fig.tight_layout()
    out_png = out_npz.with_suffix(".png")
    out_pdf = out_npz.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[ok] wrote {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
