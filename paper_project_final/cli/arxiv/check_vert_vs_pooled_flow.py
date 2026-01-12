#!/usr/bin/env python3
"""
Diagnostic script to compare vertical-trained vs pooled-trained FEF C-axes
and their corresponding FEF→LIP C-flows for one session.

It will:

1) Load FEF C-axes from two axes tags:
   - axes_tag_vert
   - axes_tag_pooled
   and compute cos(sC_vert, sC_pooled).

2) Compute category decoding AUC for each axis:
   - on pooled trials,
   - on vertical-only trials,
   - on horizontal-only trials.

3) Load FEF→LIP C-flow for up to FOUR configs:
   - vert-train / vert-flow      (flow_tag_vert_vert)
   - pooled-train / pooled-flow  (flow_tag_pooled_pooled)
   - vert-train / pooled-flow    (flow_tag_vertTrain_pooled)
   - vert-train / horiz-flow     (flow_tag_vert_horiz)  <-- NEW

And compute window-averaged:
   - mean bits_AtoB,
   - mean excess(bits_AtoB),
   - mean z_AtoB,
   - fraction of bins in window with p<alpha.

Usage example:

  python cli/check_vert_vs_pooled_flow.py \
    --out_root out \
    --sid 20201211 \
    --align stim \
    --axes_tag_vert axes_sweep-stim-vertical \
    --axes_tag_pooled axes_sweep-stim-pooled \
    --flow_tag_vert_vert crsweep-stim-vertical-none-trial \
    --flow_tag_pooled_pooled crsweep-stim-pooled-none-trial \
    --flow_tag_vertTrain_pooled crvertTrainPooled-stim-pooled-none-trial \
    --flow_tag_vert_horiz crvertTrain-stim-horizontal-none-trial \
    --win 0.15:0.35 \
    --alpha 0.05
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def parse_window(s: str) -> Tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def load_cache(out_root: Path, align: str, sid: str, area: str):
    p = out_root / align / sid / "caches" / f"area_{area}.npz"
    d = np.load(p, allow_pickle=True)
    cache = {k: d[k] for k in d.files}
    meta = cache.get("meta", {})
    if not isinstance(meta, dict):
        try:
            meta = json.loads(meta.item() if hasattr(meta, "item") else str(meta))
        except Exception:
            meta = {}
    cache["meta"] = meta
    return cache


def load_axes(out_root: Path, align: str, sid: str, area: str, axes_tag: str):
    p = out_root / align / sid / "axes" / axes_tag / f"axes_{area}.npz"
    d = np.load(p, allow_pickle=True)
    axes = {k: d[k] for k in d.files}
    meta = axes.get("meta", {})
    if not isinstance(meta, dict):
        try:
            meta = json.loads(meta.item() if hasattr(meta, "item") else str(meta))
        except Exception:
            meta = {}
    axes["meta"] = meta
    return axes


def orientation_from_cache(cache) -> np.ndarray:
    OR = cache.get("lab_orientation", None)
    if OR is None:
        N = cache["Z"].shape[0]
        return np.array(["pooled"] * N, dtype=object)
    return OR.astype(str)


def compute_auc_for_axis(cache, sC: np.ndarray, win: Tuple[float, float], ori_filter: str | None):
    """
    Compute category decoding AUC for 1D axis sC on a given subset of trials,
    averaging over a time window [win_start, win_end] in seconds.

    ori_filter: "vertical", "horizontal", or None (pooled)
    """
    Z = cache["Z"].astype(float)           # (N,B,U)
    C = cache.get("lab_C", np.full(Z.shape[0], np.nan)).astype(float)
    OR = orientation_from_cache(cache)
    time_s = cache["time"].astype(float)
    is_correct = cache.get("lab_is_correct", np.ones(Z.shape[0], dtype=bool)).astype(bool)

    # trial mask
    keep = is_correct & np.isfinite(C)
    if ori_filter is not None and ori_filter.lower() != "pooled":
        keep &= (OR == ori_filter)

    if not np.any(keep):
        return np.nan, 0

    Zk = Z[keep]    # (Nk,B,U)
    Ck = C[keep]
    time_mask = (time_s >= win[0]) & (time_s <= win[1])
    if not np.any(time_mask):
        return np.nan, int(Zk.shape[0])

    # average over window: (Nk, U)
    X = Zk[:, time_mask, :].mean(axis=1)   # (Nk,U)
    y = (Ck > 0).astype(int)
    scores = X @ sC
    if np.unique(y).size < 2:
        return np.nan, int(X.shape[0])
    try:
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = np.nan
    return auc, int(X.shape[0])


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = u.ravel().astype(float)
    v = v.ravel().astype(float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))


def safe_z(bits: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.full_like(bits, np.nan, dtype=float)
    mask = np.isfinite(bits) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    z[mask] = (bits[mask] - mu[mask]) / sd[mask]
    return z


def summarize_flow(flow_tag: str | None, label: str,
                   out_root: Path, align: str, sid: str,
                   feature: str, A: str, B: str,
                   win: Tuple[float, float], alpha: float):
    if flow_tag is None:
        print(f"\n=== Flow {label}: NO TAG PROVIDED ===")
        return
    base = out_root / align / sid / "flow" / flow_tag / feature
    p_fwd = base / f"flow_{feature}_{A}to{B}.npz"
    if not p_fwd.is_file():
        print(f"\n=== Flow {label}: FILE MISSING ===")
        print(f"  expected {p_fwd}")
        return
    Zf = np.load(p_fwd, allow_pickle=True)
    time = Zf["time"].astype(float)
    bits_AB = Zf["bits_AtoB"].astype(float)
    mu_AB = Zf["null_mean_AtoB"].astype(float)
    sd_AB = Zf["null_std_AtoB"].astype(float)
    p_AB = Zf["p_AtoB"].astype(float)
    meta = Zf["meta"].item() if hasattr(Zf["meta"], "item") else Zf["meta"]
    z_AB = safe_z(bits_AB, mu_AB, sd_AB)
    ws, we = win
    wmask = (time >= ws) & (time <= we)
    if np.any(wmask):
        w_bits = float(np.nanmean(bits_AB[wmask]))
        w_excess = float(np.nanmean((bits_AB - mu_AB)[wmask]))
        w_z = float(np.nanmean(z_AB[wmask]))
        frac_sig = float(np.nanmean((p_AB[wmask] < alpha) & np.isfinite(p_AB[wmask])))
    else:
        w_bits = w_excess = w_z = frac_sig = np.nan

    print(f"\n=== Flow {label} ({flow_tag}) ===")
    print("meta:", meta)
    print(f"[window {ws:.3f},{we:.3f}s] "
          f"mean bits_AtoB = {w_bits:.3f}, "
          f"mean excess(bits) = {w_excess:.3f}, "
          f"mean z = {w_z:.3f}, "
          f"frac p<{alpha} = {frac_sig:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--sid", required=True,
                    help="Session ID (e.g., 20201211)")
    ap.add_argument("--align", choices=["stim", "sacc"], default="stim",
                    help="Alignment to check (default: stim)")
    ap.add_argument("--feature", choices=["C"], default="C",
                    help="Feature to inspect (currently only C)")
    ap.add_argument("--axes_tag_vert", required=True,
                    help="Axes tag for vertical-trained axes (e.g., axes_sweep-stim-vertical)")
    ap.add_argument("--axes_tag_pooled", required=True,
                    help="Axes tag for pooled-trained axes (e.g., axes_sweep-stim-pooled)")
    ap.add_argument("--flow_tag_vert_vert", default=None,
                    help="Flow tag for vert-train / vert-flow (e.g., crsweep-stim-vertical-none-trial)")
    ap.add_argument("--flow_tag_pooled_pooled", default=None,
                    help="Flow tag for pooled-train / pooled-flow (e.g., crsweep-stim-pooled-none-trial)")
    ap.add_argument("--flow_tag_vertTrain_pooled", default=None,
                    help="Flow tag for vert-train / pooled-flow (e.g., crvertTrainPooled-stim-pooled-none-trial)")
    ap.add_argument("--flow_tag_vert_horiz", default=None,
                    help="Flow tag for vert-train / horizontal-flow "
                         "(e.g., crvertTrain-stim-horizontal-none-trial)")
    ap.add_argument("--pair", default=None,
                    help="Area pair A-B, e.g. MFEF-MLIP or SFEF-SLIP. "
                         "If omitted, inferred from caches.")
    ap.add_argument("--win", default="0.15:0.35",
                    help="Time window [start:end] in seconds for AUC and flow summary "
                         "(default: 0.15:0.35)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for p-values (default: 0.05)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    align = args.align
    sid = args.sid
    feature = args.feature
    win = parse_window(args.win)

    # determine pair & monkey
    cache_dir = out_root / align / sid / "caches"
    area_files = sorted(cache_dir.glob("area_*.npz"))
    if not area_files:
        raise SystemExit(f"No caches in {cache_dir}")
    areas = sorted([f.name[5:-4] for f in area_files])
    print(f"[info] areas in session {sid}: {areas}")
    first_area = areas[0]
    monkey_label = first_area[0].upper()

    if args.pair is not None:
        A_name, B_name = args.pair.split("-")
        pair = (A_name, B_name)
    else:
        if monkey_label == "M":
            pair = ("MFEF", "MLIP")
        else:
            pair = ("SFEF", "SLIP")
    A, B = pair
    print(f"[info] using pair: {A}->{B} (C-flow)")

    # load caches and axes for A=FEF
    cache_A = load_cache(out_root, align, sid, A)
    axes_vert = load_axes(out_root, align, sid, A, args.axes_tag_vert)
    axes_pool = load_axes(out_root, align, sid, A, args.axes_tag_pooled)

    sC_vert = axes_vert.get("sC", np.array([])).ravel()
    sC_pool = axes_pool.get("sC", np.array([])).ravel()
    if sC_vert.size == 0 or sC_pool.size == 0:
        raise SystemExit("One of the sC axes is empty; check axes tags and feature=C.")

    print("\n=== Axes meta (FEF) ===")
    print("[vert] meta:", axes_vert["meta"])
    print("[pool] meta:", axes_pool["meta"])

    cos_sc = cosine(sC_vert, sC_pool)
    print(f"[axes] cos(sC_vert, sC_pooled) = {cos_sc:.4f}")

    # AUC diagnostics
    print("\n=== AUC diagnostics (FEF C) ===")
    for name, sC in [("vert-axis", sC_vert), ("pooled-axis", sC_pool)]:
        for ori in [None, "vertical", "horizontal"]:
            label = "pooled" if ori is None else ori
            auc, ntr = compute_auc_for_axis(cache_A, sC, win, ori)
            print(f"[{name}] AUC(C | {label:9s}) = {auc:.3f}  (N={ntr})")

    # Flow diagnostics
    summarize_flow(args.flow_tag_vert_vert, "vert-train / vert-flow",
                   out_root, align, sid, feature, A, B, win, args.alpha)
    summarize_flow(args.flow_tag_pooled_pooled, "pooled-train / pooled-flow",
                   out_root, align, sid, feature, A, B, win, args.alpha)
    summarize_flow(args.flow_tag_vertTrain_pooled, "vert-train / pooled-flow",
                   out_root, align, sid, feature, A, B, win, args.alpha)
    summarize_flow(args.flow_tag_vert_horiz, "vert-train / horiz-flow",
                   out_root, align, sid, feature, A, B, win, args.alpha)

    print("\n[done] diagnostics complete.")


if __name__ == "__main__":
    main()
