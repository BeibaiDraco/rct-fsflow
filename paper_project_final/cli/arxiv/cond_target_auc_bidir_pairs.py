#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

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


def cache_path(out_root: Path, align: str, sid: str, area: str) -> Path:
    return out_root / align / sid / "caches" / f"area_{area}.npz"


def axis_path(out_root: Path, align: str, sid: str, axes_tag: str, area: str) -> Path:
    return out_root / align / sid / "axes" / axes_tag / f"axes_{area}.npz"


def list_areas(cache_dir: Path) -> List[str]:
    return sorted([p.name[5:-4] for p in cache_dir.glob("area_*.npz")])


def canonical_pairs(areas: List[str]) -> List[Tuple[str, str]]:
    """
    Canonical pairs if FEF/LIP/SC exist; otherwise returns all unordered pairs.
    """
    aset = set(areas)
    pref = areas[0][0].upper()
    if pref == "M":
        want = ["MFEF", "MLIP", "MSC"]
    else:
        want = ["SFEF", "SLIP", "SSC"]

    present = [a for a in want if a in aset]
    pairs = []
    if len(present) >= 2:
        # canonical order
        if want[0] in aset and want[1] in aset:
            pairs.append((want[0], want[1]))
        if want[0] in aset and want[2] in aset:
            pairs.append((want[0], want[2]))
        if want[1] in aset and want[2] in aset:
            pairs.append((want[1], want[2]))
        if pairs:
            return pairs

    # fallback: all unordered pairs
    out = []
    for i in range(len(areas)):
        for j in range(i+1, len(areas)):
            out.append((areas[i], areas[j]))
    return out


def trial_mask(cache: Dict, orientation: str, pt_min_ms: Optional[float]) -> np.ndarray:
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
    # need labels for split
    keep &= np.isfinite(cache.get("lab_C", np.full(N, np.nan)).astype(float))
    keep &= np.isfinite(cache.get("lab_R", np.full(N, np.nan)).astype(float))
    return keep


def project_1d(cache: Dict, s: np.ndarray, keep: np.ndarray) -> np.ndarray:
    Z = cache["Z"][keep].astype(float)  # (N,B,U)
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size != Z.shape[2]:
        raise ValueError(f"Axis dim {s.size} != n_units {Z.shape[2]}")
    return np.tensordot(Z, s, axes=([2], [0]))  # (N,B)


def encode_CR(cache: Dict, keep: np.ndarray) -> np.ndarray:
    C = np.round(cache["lab_C"][keep].astype(float)).astype(int)
    R = np.round(cache["lab_R"][keep].astype(float)).astype(int)
    base = 1000
    return (C * base + R).astype(int)


def split_high_low_within_strata(d_src: np.ndarray, strata: np.ndarray, min_per_stratum: int) -> Tuple[np.ndarray, np.ndarray]:
    low = []
    high = []
    for v in np.unique(strata):
        idx = np.where(strata == v)[0]
        if idx.size < min_per_stratum:
            continue
        order = np.argsort(d_src[idx])
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


def run_one_direction(
    out_root: Path,
    sid: str,
    align: str,
    orientation: str,
    pt_min_ms: float,
    axes_tag: str,
    src_area: str,
    tgt_area: str,
    win_src: Tuple[float, float],
    min_per_stratum: int,
    perms: int,
    seed: int,
    out_tag: str,
):
    cache_src = load_npz(cache_path(out_root, align, sid, src_area))
    cache_tgt = load_npz(cache_path(out_root, align, sid, tgt_area))

    keep_src = trial_mask(cache_src, orientation, pt_min_ms)
    keep_tgt = trial_mask(cache_tgt, orientation, pt_min_ms)
    if keep_src.shape[0] != keep_tgt.shape[0]:
        raise SystemExit("Source and target caches have different #trials.")
    keep = keep_src & keep_tgt
    if keep.sum() < 60:
        print(f"[skip] {sid} {src_area}->{tgt_area}: too few trials after filter N={keep.sum()}")
        return

    # axes
    axes_src = load_npz(axis_path(out_root, align, sid, axes_tag, src_area))
    axes_tgt = load_npz(axis_path(out_root, align, sid, axes_tag, tgt_area))
    s_src = axes_src.get("sC", np.array([])).ravel()
    s_tgt = axes_tgt.get("sC", np.array([])).ravel()
    if s_src.size == 0 or s_tgt.size == 0:
        print(f"[skip] {sid} {src_area}->{tgt_area}: missing sC axis in {axes_tag}")
        return

    time = cache_src["time"].astype(float)
    YA = project_1d(cache_src, s_src, keep)  # (N,B)
    YB = project_1d(cache_tgt, s_tgt, keep)  # (N,B)
    C = cache_src["lab_C"][keep].astype(float)

    # early window on source
    wmask = (time >= win_src[0]) & (time <= win_src[1])
    if not np.any(wmask):
        print(f"[skip] {sid} {src_area}->{tgt_area}: no bins in win_src={win_src}")
        return
    d_src = np.nanmean(YA[:, wmask], axis=1)

    strata = encode_CR(cache_src, keep)
    low_idx, high_idx = split_high_low_within_strata(d_src, strata, min_per_stratum)
    if low_idx.size < 30 or high_idx.size < 30:
        print(f"[skip] {sid} {src_area}->{tgt_area}: too few after split low={low_idx.size} high={high_idx.size}")
        return

    Bbins = time.size
    auc_all = np.full(Bbins, np.nan)
    auc_low = np.full(Bbins, np.nan)
    auc_high = np.full(Bbins, np.nan)
    for b in range(Bbins):
        auc_all[b] = auc_from_scores(YB[:, b], C)
        auc_low[b] = auc_from_scores(YB[low_idx, b], C[low_idx])
        auc_high[b] = auc_from_scores(YB[high_idx, b], C[high_idx])
    delta_auc = auc_high - auc_low

    # permutation null for delta_auc(t)
    rng = np.random.default_rng(seed)
    null_mean = np.full(Bbins, np.nan)
    null_std = np.full(Bbins, np.nan)
    z_delta = np.full(Bbins, np.nan)

    if perms > 0:
        P = int(perms)
        deltas = np.full((P, Bbins), np.nan)
        stratum_to_idx = {v: np.where(strata == v)[0] for v in np.unique(strata)}

        for p in range(P):
            low_p = []
            high_p = []
            for v, idx in stratum_to_idx.items():
                if idx.size < min_per_stratum:
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
                deltas[p, b] = auc_from_scores(YB[high_p, b], C[high_p]) - auc_from_scores(YB[low_p, b], C[low_p])

        null_mean = np.nanmean(deltas, axis=0)
        null_std = np.nanstd(deltas, axis=0, ddof=1)
        good = np.isfinite(delta_auc) & np.isfinite(null_mean) & np.isfinite(null_std) & (null_std > 0)
        z_delta[good] = (delta_auc[good] - null_mean[good]) / null_std[good]

    # save
    out_dir = out_root / align / sid / "condenc" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{src_area}_to_{tgt_area}_condAUC_{orientation}.npz"

    np.savez_compressed(
        out_npz,
        time=time,
        auc_all=auc_all, auc_low=auc_low, auc_high=auc_high,
        delta_auc=delta_auc,
        delta_auc_null_mean=null_mean,
        delta_auc_null_std=null_std,
        delta_auc_z=z_delta,
        meta=dict(
            sid=sid, align=align, orientation=orientation,
            src_area=src_area, tgt_area=tgt_area,
            axes_tag=axes_tag,
            win_src=win_src, pt_min_ms=pt_min_ms,
            n_trials=int(keep.sum()),
            n_low=int(low_idx.size), n_high=int(high_idx.size),
            perms=int(perms),
            min_per_stratum=int(min_per_stratum),
        )
    )

    # plot (3 panels)
    tms = time * 1000.0
    fig = plt.figure(figsize=(8.0, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvline(0, ls="--", c="k", lw=0.8)
    ax1.axvspan(win_src[0]*1000, win_src[1]*1000, color="0.9", alpha=0.5, label="SRC early window")
    ax1.plot(tms, auc_all, lw=2.0, label="AUC(all)")
    ax1.plot(tms, auc_high, lw=2.0, label="AUC(high SRC state)")
    ax1.plot(tms, auc_low, lw=2.0, label="AUC(low SRC state)")
    ax1.set_ylabel(f"AUC(C | {tgt_area})")
    ax1.set_title(f"{sid} {src_area}→{tgt_area} | ori={orientation} | axes={axes_tag}")
    ax1.legend(frameon=False, loc="lower right")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axvline(0, ls="--", c="k", lw=0.8)
    ax2.plot(tms, delta_auc, lw=2.0, label="ΔAUC = high - low")
    if perms > 0 and np.any(np.isfinite(null_mean)):
        ax2.plot(tms, null_mean, lw=1.5, ls=":", label="null mean")
        ax2.fill_between(tms, null_mean-null_std, null_mean+null_std, alpha=0.2, linewidth=0, label="null ±1σ")
    ax2.set_ylabel("ΔAUC")
    ax2.legend(frameon=False, loc="upper right")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axvline(0, ls="--", c="k", lw=0.8)
    if perms > 0:
        ax3.plot(tms, z_delta, lw=2.0, label="z(ΔAUC)")
        ax3.axhline(0, ls=":", c="k", lw=0.8)
        ax3.set_ylabel("Z")
        ax3.legend(frameon=False, loc="upper right")
    else:
        ax3.plot(tms, delta_auc, lw=2.0, label="ΔAUC (no perms)")
        ax3.set_ylabel("ΔAUC")
        ax3.legend(frameon=False, loc="upper right")

    ax3.set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_npz.with_suffix(".png"), dpi=300)
    fig.savefig(out_npz.with_suffix(".pdf"))
    plt.close(fig)

    print(f"[ok] wrote {out_npz}")


def main():
    ap = argparse.ArgumentParser(description="Bidirectional, multi-pair conditional target AUC over time.")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--align", choices=["stim"], default="stim")
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--axes_tag", default="axes_sweep-stim-pooled",
                    help="Axes tag to use for ALL areas (default: axes_sweep-stim-pooled)")
    ap.add_argument("--pairs", default="auto",
                    help="Pairs to analyze: 'auto' or comma list like 'MFEF-MLIP,MFEF-MSC'")
    ap.add_argument("--both_directions", action="store_true",
                    help="If set, run both A->B and B->A for each pair")
    ap.add_argument("--win_src", default="0.12:0.20",
                    help="Early window for source area in seconds (default: 0.12:0.20)")
    ap.add_argument("--win_src_rev", default=None,
                    help="Optional early window for the reverse direction (default: same as win_src)")
    ap.add_argument("--min_per_stratum", type=int, default=6)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="condAUC_bidir_v1")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    sid = args.sid
    align = args.align
    orientation = args.orientation
    win_src = parse_window(args.win_src)
    win_src_rev = parse_window(args.win_src_rev) if args.win_src_rev is not None else win_src

    cache_dir = out_root / align / sid / "caches"
    areas = list_areas(cache_dir)
    if not areas:
        raise SystemExit(f"No caches under {cache_dir}")

    # decide pairs
    if args.pairs.strip().lower() == "auto":
        pairs = canonical_pairs(areas)
    else:
        pairs = []
        for tok in args.pairs.split(","):
            a, b = tok.strip().split("-")
            pairs.append((a, b))

    for (A, B) in pairs:
        if A not in areas or B not in areas:
            print(f"[skip] pair {A}-{B} not present in {sid}")
            continue

        # A->B
        run_one_direction(
            out_root=out_root, sid=sid, align=align,
            orientation=orientation, pt_min_ms=args.pt_min_ms,
            axes_tag=args.axes_tag,
            src_area=A, tgt_area=B,
            win_src=win_src,
            min_per_stratum=args.min_per_stratum,
            perms=args.perms, seed=args.seed,
            out_tag=args.tag,
        )

        if args.both_directions:
            run_one_direction(
                out_root=out_root, sid=sid, align=align,
                orientation=orientation, pt_min_ms=args.pt_min_ms,
                axes_tag=args.axes_tag,
                src_area=B, tgt_area=A,
                win_src=win_src_rev,
                min_per_stratum=args.min_per_stratum,
                perms=args.perms, seed=args.seed + 1,
                out_tag=args.tag,
            )


if __name__ == "__main__":
    main()
