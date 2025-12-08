#!/usr/bin/env python3
"""
Multi-session diagnostic for vertical-trained vs pooled-trained FEF C-axes
and FEF→LIP C-flow, across sessions.

Per session (align=stim by default) it computes:

1) Axes diagnostics (FEF):
   - cos(sC_vert, sC_pooled)
   - AUC(C | pooled), AUC(C | vertical), AUC(C | horizontal)
     for both:
       - vert-trained axis (axes_tag_vert)
       - pooled-trained axis (axes_tag_pooled)

2) Flow diagnostics (FEF→LIP, feature=C):
   - For each config, if present:
       - vert-train / vert-flow      (flow_tag_vert_vert)
       - pooled-train / pooled-flow  (flow_tag_pooled_pooled)
       - vert-train / pooled-flow    (flow_tag_vertTrain_pooled)
   - Window-averaged (over [win_start, win_end] in seconds):
       - mean bits_AtoB
       - mean excess bits_AtoB (obs - null_mean)
       - mean z_AtoB
       - fraction of bins with p<alpha

Writes a TSV summary per align:

  out/<align>/diagnostics/vert_vs_pooled_C_flow_<align>.tsv

Columns include:
  sid, monkey_label, pair, cos_sC, AUCs, and flow window stats per config.

Usage example (stim-align):

  python cli/check_vert_vs_pooled_flow_multi.py \
    --out_root out \
    --align stim \
    --axes_tag_vert_base axes_sweep \
    --axes_tag_pooled_base axes_sweep \
    --flow_tag_sweep_base crsweep \
    --flow_tag_vertTrainPooled_base crvertTrainPooled \
    --win 0.10:0.30 \
    --alpha 0.05
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------- helpers ----------------

def parse_window(s: str) -> Tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def find_sessions(out_root: Path, align: str) -> List[str]:
    base = out_root / align
    if not base.exists():
        return []
    sids = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "caches").is_dir():
            sids.append(p.name)
    return sids


def load_cache(out_root: Path, align: str, sid: str, area: str):
    p = out_root / align / sid / "caches" / f"area_{area}.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
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
    if not p.is_file():
        raise FileNotFoundError(p)
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


def load_flow_window(
    out_root: Path,
    align: str,
    sid: str,
    feature: str,
    flow_tag: str,
    A: str,
    B: str,
    win: Tuple[float, float],
    alpha: float,
):
    """
    Load FEF->LIP flow_AtoB for given tag, and compute window-averaged stats.
    Returns (present_flag, meta, N, mean_bits, mean_excess, mean_z, frac_sig).
    """
    base = out_root / align / sid / "flow" / flow_tag / feature
    p_fwd = base / f"flow_{feature}_{A}to{B}.npz"
    if not p_fwd.is_file():
        return False, None, 0, np.nan, np.nan, np.nan, np.nan
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
    N = int(meta.get("N", 0))
    return True, meta, N, w_bits, w_excess, w_z, frac_sig


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--align", choices=["stim", "sacc"], default="stim",
                    help="Alignment to check (default: stim)")
    ap.add_argument("--feature", choices=["C"], default="C",
                    help="Feature to inspect (currently only C)")
    # axes tags: we build full tags from bases
    ap.add_argument("--axes_tag_vert_base", default="axes_sweep",
                    help="Base for vertical-trained axes tag: "
                         "full = <axes_tag_vert_base>-<align>-vertical "
                         "(default: axes_sweep)")
    ap.add_argument("--axes_tag_pooled_base", default="axes_sweep",
                    help="Base for pooled-trained axes tag: "
                         "full = <axes_tag_pooled_base>-<align>-pooled "
                         "(default: axes_sweep)")
    # flow tags: we build full tags for trial_shuffle / none standardization config
    ap.add_argument("--flow_tag_sweep_base", default="crsweep",
                    help="Base for sweep flow tags (vert/vert and pooled/pooled); "
                         "full vert/vert tag = <base>-<align>-vertical-none-trial, "
                         "full pooled/pooled tag = <base>-<align>-pooled-none-trial "
                         "(default: crsweep)")
    ap.add_argument("--flow_tag_vertTrainPooled_base", default="crvertTrainPooled",
                    help="Base for vertical-trained / pooled-flow tags; "
                         "full tag = <base>-<align>-pooled-none-trial "
                         "(default: crvertTrainPooled)")
    ap.add_argument("--pair", default=None,
                    help="Area pair A-B for flow, e.g. MFEF-MLIP or SFEF-SLIP. "
                         "If omitted, inferred from area names.")
    ap.add_argument("--win", default="0.10:0.30",
                    help="Time window [start:end] in seconds for AUC and flow summary "
                         "(default: 0.10:0.30)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for p-values (default: 0.05)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    align = args.align
    feature = args.feature
    win = parse_window(args.win)

    sids = find_sessions(out_root, align)
    if not sids:
        raise SystemExit(f"No sessions found under {out_root}/{align}")

    print(f"[info] align={align}, found {len(sids)} sessions: {sids}")

    # Output TSV
    diag_dir = out_root / align / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = diag_dir / f"vert_vs_pooled_C_flow_{align}.tsv"

    with out_tsv.open("w") as f_out:
        # header
        header_cols = [
            "sid", "monkey_label", "pair",
            "cos_sC_vert_pooled",
            "auc_vert_axis_pooled", "auc_vert_axis_vertical", "auc_vert_axis_horizontal",
            "auc_pool_axis_pooled", "auc_pool_axis_vertical", "auc_pool_axis_horizontal",
            # flow metrics: vert/vert
            "vv_present", "vv_N", "vv_mean_bits", "vv_mean_excess", "vv_mean_z", "vv_frac_sig",
            # pooled/pooled
            "pp_present", "pp_N", "pp_mean_bits", "pp_mean_excess", "pp_mean_z", "pp_frac_sig",
            # vertTrain/pooled
            "vp_present", "vp_N", "vp_mean_bits", "vp_mean_excess", "vp_mean_z", "vp_frac_sig",
        ]
        f_out.write("\t".join(header_cols) + "\n")

        for sid in sids:
            try:
                cache_dir = out_root / align / sid / "caches"
                area_files = sorted(cache_dir.glob("area_*.npz"))
                if not area_files:
                    print(f"[skip] {sid}: no caches")
                    continue
                areas = sorted([af.name[5:-4] for af in area_files])
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
                A_area, B_area = pair

                print(f"\n[session {sid}] monkey={monkey_label}, pair={A_area}->{B_area}")

                # Load FEF cache & axes
                cache_A = load_cache(out_root, align, sid, A_area)
                axes_tag_vert = f"{args.axes_tag_vert_base}-{align}-vertical"
                axes_tag_pooled = f"{args.axes_tag_pooled_base}-{align}-pooled"
                axes_vert = load_axes(out_root, align, sid, A_area, axes_tag_vert)
                axes_pool = load_axes(out_root, align, sid, A_area, axes_tag_pooled)

                sC_vert = axes_vert.get("sC", np.array([])).ravel()
                sC_pool = axes_pool.get("sC", np.array([])).ravel()
                if sC_vert.size == 0 or sC_pool.size == 0:
                    print(f"[warn] {sid}: empty sC in one of the axes tags; skipping session")
                    continue

                cos_sc = cosine(sC_vert, sC_pool)

                # AUC diagnostics
                auc_vert_pooled, n_pooled = compute_auc_for_axis(cache_A, sC_vert, win, None)
                auc_vert_vert, n_vert = compute_auc_for_axis(cache_A, sC_vert, win, "vertical")
                auc_vert_horiz, n_h = compute_auc_for_axis(cache_A, sC_vert, win, "horizontal")

                auc_pool_pooled, _ = compute_auc_for_axis(cache_A, sC_pool, win, None)
                auc_pool_vert, _ = compute_auc_for_axis(cache_A, sC_pool, win, "vertical")
                auc_pool_horiz, _ = compute_auc_for_axis(cache_A, sC_pool, win, "horizontal")

                print(f"  cos(sC_vert, sC_pooled) = {cos_sc:.4f}")
                print(f"  vert-axis:   AUC(C|pooled)={auc_vert_pooled:.3f}, "
                      f"AUC(C|vert)={auc_vert_vert:.3f}, AUC(C|horiz)={auc_vert_horiz:.3f}")
                print(f"  pooled-axis: AUC(C|pooled)={auc_pool_pooled:.3f}, "
                      f"AUC(C|vert)={auc_pool_vert:.3f}, AUC(C|horiz)={auc_pool_horiz:.3f}")

                # Flow tags
                flow_tag_vert_vert = f"{args.flow_tag_sweep_base}-{align}-vertical-none-trial"
                flow_tag_pooled_pooled = f"{args.flow_tag_sweep_base}-{align}-pooled-none-trial"
                flow_tag_vertTrain_pooled = f"{args.flow_tag_vertTrainPooled_base}-{align}-pooled-none-trial"

                # Flow diagnostics
                vv_present, vv_meta, vv_N, vv_bits, vv_excess, vv_z, vv_frac = load_flow_window(
                    out_root, align, sid, feature,
                    flow_tag_vert_vert, A_area, B_area, win, args.alpha)

                pp_present, pp_meta, pp_N, pp_bits, pp_excess, pp_z, pp_frac = load_flow_window(
                    out_root, align, sid, feature,
                    flow_tag_pooled_pooled, A_area, B_area, win, args.alpha)

                vp_present, vp_meta, vp_N, vp_bits, vp_excess, vp_z, vp_frac = load_flow_window(
                    out_root, align, sid, feature,
                    flow_tag_vertTrain_pooled, A_area, B_area, win, args.alpha)

                print("  [flow vv] present=", vv_present, " N=", vv_N,
                      " mean bits=", f"{vv_bits:.3f}", " mean z=", f"{vv_z:.3f}")
                print("  [flow pp] present=", pp_present, " N=", pp_N,
                      " mean bits=", f"{pp_bits:.3f}", " mean z=", f"{pp_z:.3f}")
                print("  [flow vp] present=", vp_present, " N=", vp_N,
                      " mean bits=", f"{vp_bits:.3f}", " mean z=", f"{vp_z:.3f}")

                row = [
                    sid,
                    monkey_label,
                    f"{A_area}-{B_area}",
                    f"{cos_sc:.6f}",
                    f"{auc_vert_pooled:.6f}",
                    f"{auc_vert_vert:.6f}",
                    f"{auc_vert_horiz:.6f}",
                    f"{auc_pool_pooled:.6f}",
                    f"{auc_pool_vert:.6f}",
                    f"{auc_pool_horiz:.6f}",
                    # vv
                    ("1" if vv_present else "0"),
                    str(vv_N),
                    f"{vv_bits:.6f}",
                    f"{vv_excess:.6f}",
                    f"{vv_z:.6f}",
                    f"{vv_frac:.6f}",
                    # pp
                    ("1" if pp_present else "0"),
                    str(pp_N),
                    f"{pp_bits:.6f}",
                    f"{pp_excess:.6f}",
                    f"{pp_z:.6f}",
                    f"{pp_frac:.6f}",
                    # vp
                    ("1" if vp_present else "0"),
                    str(vp_N),
                    f"{vp_bits:.6f}",
                    f"{vp_excess:.6f}",
                    f"{vp_z:.6f}",
                    f"{vp_frac:.6f}",
                ]
                f_out.write("\t".join(row) + "\n")

            except FileNotFoundError as e:
                print(f"[skip] {sid}: missing file {e}")
                continue

    print(f"\n[done] wrote diagnostics to {out_tsv}")


if __name__ == "__main__":
    main()
