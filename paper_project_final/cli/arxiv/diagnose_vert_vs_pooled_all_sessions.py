#!/usr/bin/env python3
"""
Diagnostics across ALL stim sessions for vertical vs pooled axes and flows.

For each stim session SID with MFEF/MLIP present, this script:

  1) Loads FEF C-axes from:
        axes_sweep-stim-vertical
        axes_sweep-stim-pooled
     and computes cos(sC_vert, sC_pooled).

  2) Computes C-decoding AUC for each axis:
        - pooled trials
        - vertical-only trials
        - horizontal-only trials
     in a given window [win_start, win_end] (in seconds).

  3) Computes ORIENTATION decoding AUC for each axis:
        label = 1 if vertical, 0 if horizontal
     on pooled trials.

  4) Loads FEF→MLIP C-flow from three tags:
        - crsweep-stim-vertical-none-trial      (vert-train / vert-flow)
        - crsweep-stim-pooled-none-trial       (pooled-train / pooled-flow)
        - crvertTrainPooled-stim-pooled-none-trial (vert-train / pooled-flow)
     and computes, in the same window:
        - mean bits_AtoB
        - mean excess bits_AtoB (obs - null_mean)
        - mean z_AtoB

Outputs:
  out/stim/diagnostics/vert_vs_pooled_diag.npz

containing arrays over sessions:
  - session_ids
  - cos_sC
  - auc_C_vertAxis_pooled, auc_C_vertAxis_vert, auc_C_vertAxis_horiz
  - auc_C_pooledAxis_pooled, ...
  - auc_O_vertAxis_pooled, auc_O_pooledAxis_pooled
  - flow_bits_vertVert, flow_excess_vertVert, flow_z_vertVert
  - flow_bits_poolPool,  ...
  - flow_bits_vertPool,  ...

Use this to see if the pattern from SID 20201211 is systematic across sessions.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.metrics import roc_auc_score


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
        return None
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
        return None
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


def compute_auc_C_for_axis(cache, sC: np.ndarray, win: Tuple[float, float], ori_filter: str | None):
    Z = cache["Z"].astype(float)           # (N,B,U)
    C = cache.get("lab_C", np.full(Z.shape[0], np.nan)).astype(float)
    OR = orientation_from_cache(cache)
    time_s = cache["time"].astype(float)
    is_correct = cache.get("lab_is_correct", np.ones(Z.shape[0], dtype=bool)).astype(bool)

    keep = is_correct & np.isfinite(C)
    if ori_filter is not None and ori_filter.lower() != "pooled":
        keep &= (OR == ori_filter)
    if not np.any(keep):
        return np.nan, 0

    Zk = Z[keep]
    Ck = C[keep]
    tmask = (time_s >= win[0]) & (time_s <= win[1])
    if not np.any(tmask):
        return np.nan, int(Zk.shape[0])

    X = Zk[:, tmask, :].mean(axis=1)   # (Nk,U)
    y = (Ck > 0).astype(int)
    scores = X @ sC
    if np.unique(y).size < 2:
        return np.nan, int(X.shape[0])
    try:
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = np.nan
    return auc, int(X.shape[0])


def compute_auc_O_for_axis(cache, sC: np.ndarray, win: Tuple[float, float]):
    Z = cache["Z"].astype(float)
    OR = orientation_from_cache(cache)
    time_s = cache["time"].astype(float)
    is_correct = cache.get("lab_is_correct", np.ones(Z.shape[0], dtype=bool)).astype(bool)

    # only consider trials that are explicitly vertical or horizontal
    mask = is_correct & np.isfinite(Z[:, 0, 0])
    mask &= np.isin(OR, ["vertical", "horizontal"])
    if not np.any(mask):
        return np.nan, 0

    Zk = Z[mask]
    ORk = OR[mask]
    tmask = (time_s >= win[0]) & (time_s <= win[1])
    if not np.any(tmask):
        return np.nan, int(Zk.shape[0])

    X = Zk[:, tmask, :].mean(axis=1)  # (Nk,U)
    # orientation label: 1 for vertical, 0 for horizontal
    y = (ORk == "vertical").astype(int)
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


def summarize_flow_window(out_root: Path, align: str, sid: str, feature: str,
                          flow_tag: str, A: str, B: str,
                          win: Tuple[float, float]) -> Tuple[float, float, float]:
    base = out_root / align / sid / "flow" / flow_tag / feature
    p_fwd = base / f"flow_{feature}_{A}to{B}.npz"
    if not p_fwd.is_file():
        return np.nan, np.nan, np.nan
    Zf = np.load(p_fwd, allow_pickle=True)
    time = Zf["time"].astype(float)
    bits_AB = Zf["bits_AtoB"].astype(float)
    mu_AB = Zf["null_mean_AtoB"].astype(float)
    sd_AB = Zf["null_std_AtoB"].astype(float)
    z_AB = safe_z(bits_AB, mu_AB, sd_AB)
    ws, we = win
    wmask = (time >= ws) & (time <= we)
    if not np.any(wmask):
        return np.nan, np.nan, np.nan
    w_bits = float(np.nanmean(bits_AB[wmask]))
    w_excess = float(np.nanmean((bits_AB - mu_AB)[wmask]))
    w_z = float(np.nanmean(z_AB[wmask]))
    return w_bits, w_excess, w_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--axes_tag_vert", default="axes_sweep-stim-vertical",
                    help="Axes tag for vertical-trained axes (FEF) (default: axes_sweep-stim-vertical)")
    ap.add_argument("--axes_tag_pooled", default="axes_sweep-stim-pooled",
                    help="Axes tag for pooled-trained axes (FEF) (default: axes_sweep-stim-pooled)")
    ap.add_argument("--flow_tag_vert_vert", default="crsweep-stim-vertical-none-trial",
                    help="Flow tag for vert-train / vert-flow (default: crsweep-stim-vertical-none-trial)")
    ap.add_argument("--flow_tag_pooled_pooled", default="crsweep-stim-pooled-none-trial",
                    help="Flow tag for pooled-train / pooled-flow (default: crsweep-stim-pooled-none-trial)")
    ap.add_argument("--flow_tag_vertTrain_pooled", default="crvertTrainPooled-stim-pooled-none-trial",
                    help="Flow tag for vert-train / pooled-flow (default: crvertTrainPooled-stim-pooled-none-trial)")
    ap.add_argument("--win", default="0.10:0.30",
                    help="Window [start:end] in seconds (default: 0.10:0.30)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    align = "stim"
    win = parse_window(args.win)

    sids = find_sessions(out_root, align)
    print(f"[info] Found {len(sids)} stim sessions under {out_root}")

    session_ids = []
    cos_sc_list = []

    auc_C_vertAxis_pooled = []
    auc_C_vertAxis_vert = []
    auc_C_vertAxis_horiz = []

    auc_C_pooledAxis_pooled = []
    auc_C_pooledAxis_vert = []
    auc_C_pooledAxis_horiz = []

    auc_O_vertAxis_pooled = []
    auc_O_pooledAxis_pooled = []

    flow_bits_vertVert = []
    flow_excess_vertVert = []
    flow_z_vertVert = []

    flow_bits_poolPool = []
    flow_excess_poolPool = []
    flow_z_poolPool = []

    flow_bits_vertPool = []
    flow_excess_vertPool = []
    flow_z_vertPool = []

    for sid in sids:
        # require MFEF & MLIP
        cache_dir = out_root / align / sid / "caches"
        area_files = sorted(cache_dir.glob("area_*.npz"))
        areas = sorted([f.name[5:-4] for f in area_files])
        if not areas:
            continue
        if not any(a.endswith("FEF") for a in areas) or not any(a.endswith("LIP") for a in areas):
            continue
        if areas[0].startswith("M"):
            A, B = "MFEF", "MLIP"
        else:
            A, B = "SFEF", "SLIP"

        cache_A = load_cache(out_root, align, sid, A)
        if cache_A is None:
            continue

        axes_vert = load_axes(out_root, align, sid, A, args.axes_tag_vert)
        axes_pool = load_axes(out_root, align, sid, A, args.axes_tag_pooled)
        if axes_vert is None or axes_pool is None:
            print(f"[warn] Missing axes for SID {sid}; skipping")
            continue

        sC_vert = axes_vert.get("sC", np.array([])).ravel()
        sC_pool = axes_pool.get("sC", np.array([])).ravel()
        if sC_vert.size == 0 or sC_pool.size == 0:
            print(f"[warn] empty sC for SID {sid}; skipping")
            continue

        cos_sc = cosine(sC_vert, sC_pool)

        # AUC(C) for both axes
        auc_v_pool, _ = compute_auc_C_for_axis(cache_A, sC_vert, win, ori_filter=None)
        auc_v_vert, _ = compute_auc_C_for_axis(cache_A, sC_vert, win, ori_filter="vertical")
        auc_v_horiz, _ = compute_auc_C_for_axis(cache_A, sC_vert, win, ori_filter="horizontal")

        auc_p_pool, _ = compute_auc_C_for_axis(cache_A, sC_pool, win, ori_filter=None)
        auc_p_vert, _ = compute_auc_C_for_axis(cache_A, sC_pool, win, ori_filter="vertical")
        auc_p_horiz, _ = compute_auc_C_for_axis(cache_A, sC_pool, win, ori_filter="horizontal")

        # AUC(O) for both axes
        auc_O_vert, _ = compute_auc_O_for_axis(cache_A, sC_vert, win)
        auc_O_pool, _ = compute_auc_O_for_axis(cache_A, sC_pool, win)

        # flows
        fb_vv, fx_vv, fz_vv = summarize_flow_window(
            out_root, align, sid, "C", args.flow_tag_vert_vert, A, B, win)
        fb_pp, fx_pp, fz_pp = summarize_flow_window(
            out_root, align, sid, "C", args.flow_tag_pooled_pooled, A, B, win)
        fb_vp, fx_vp, fz_vp = summarize_flow_window(
            out_root, align, sid, "C", args.flow_tag_vertTrain_pooled, A, B, win)

        session_ids.append(sid)
        cos_sc_list.append(cos_sc)

        auc_C_vertAxis_pooled.append(auc_v_pool)
        auc_C_vertAxis_vert.append(auc_v_vert)
        auc_C_vertAxis_horiz.append(auc_v_horiz)

        auc_C_pooledAxis_pooled.append(auc_p_pool)
        auc_C_pooledAxis_vert.append(auc_p_vert)
        auc_C_pooledAxis_horiz.append(auc_p_horiz)

        auc_O_vertAxis_pooled.append(auc_O_vert)
        auc_O_pooledAxis_pooled.append(auc_O_pool)

        flow_bits_vertVert.append(fb_vv)
        flow_excess_vertVert.append(fx_vv)
        flow_z_vertVert.append(fz_vv)

        flow_bits_poolPool.append(fb_pp)
        flow_excess_poolPool.append(fx_pp)
        flow_z_poolPool.append(fz_pp)

        flow_bits_vertPool.append(fb_vp)
        flow_excess_vertPool.append(fx_vp)
        flow_z_vertPool.append(fz_vp)

        print(f"[diag] SID={sid} cos={cos_sc:.3f} "
              f"flow_z_vertPool={fz_vp:.2f}, flow_z_poolPool={fz_pp:.2f}, flow_z_vertVert={fz_vv:.2f}")

    if not session_ids:
        print("[warn] No valid sessions found; nothing to save.")
        return

    out_dir = out_root / "stim" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vert_vs_pooled_diag.npz"

    np.savez_compressed(
        out_path,
        session_ids=np.array(session_ids, dtype="U"),
        cos_sC=np.array(cos_sc_list, dtype=float),
        auc_C_vertAxis_pooled=np.array(auc_C_vertAxis_pooled, dtype=float),
        auc_C_vertAxis_vert=np.array(auc_C_vertAxis_vert, dtype=float),
        auc_C_vertAxis_horiz=np.array(auc_C_vertAxis_horiz, dtype=float),
        auc_C_pooledAxis_pooled=np.array(auc_C_pooledAxis_pooled, dtype=float),
        auc_C_pooledAxis_vert=np.array(auc_C_pooledAxis_vert, dtype=float),
        auc_C_pooledAxis_horiz=np.array(auc_C_pooledAxis_horiz, dtype=float),
        auc_O_vertAxis_pooled=np.array(auc_O_vertAxis_pooled, dtype=float),
        auc_O_pooledAxis_pooled=np.array(auc_O_pooledAxis_pooled, dtype=float),
        flow_bits_vertVert=np.array(flow_bits_vertVert, dtype=float),
        flow_excess_vertVert=np.array(flow_excess_vertVert, dtype=float),
        flow_z_vertVert=np.array(flow_z_vertVert, dtype=float),
        flow_bits_poolPool=np.array(flow_bits_poolPool, dtype=float),
        flow_excess_poolPool=np.array(flow_excess_poolPool, dtype=float),
        flow_z_poolPool=np.array(flow_z_poolPool, dtype=float),
        flow_bits_vertPool=np.array(flow_bits_vertPool, dtype=float),
        flow_excess_vertPool=np.array(flow_excess_vertPool, dtype=float),
        flow_z_vertPool=np.array(flow_z_vertPool, dtype=float),
        win_start_s=win[0],
        win_end_s=win[1],
        axes_tag_vert=np.array(args.axes_tag_vert),
        axes_tag_pooled=np.array(args.axes_tag_pooled),
        flow_tag_vert_vert=np.array(args.flow_tag_vert_vert),
        flow_tag_pooled_pooled=np.array(args.flow_tag_pooled_pooled),
        flow_tag_vertTrain_pooled=np.array(args.flow_tag_vertTrain_pooled),
    )

    print(f"[done] diagnostics saved to {out_path}")


if __name__ == "__main__":
    main()
