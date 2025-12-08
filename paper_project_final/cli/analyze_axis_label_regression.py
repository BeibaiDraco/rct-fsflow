#!/usr/bin/env python3
"""
Analyze what FEF axes encode in terms of category (C), context (O), and C×O.

For each stim session and for FEF only:

  - Load FEF axes from:
      - pooled:   axes_sweep-stim-pooled/axes_<FEF>.npz
      - vertical: axes_sweep-stim-vertical/axes_<FEF>.npz
      - horizontal: axes_sweep-stim-horizontal/axes_<FEF>.npz (if present)
  - Load FEF cache: out/stim/<sid>/caches/area_<FEF>.npz

  - For each axis type:
      - C_pool   (sC from pooled)
      - C_vert   (sC from vertical)
      - C_horiz  (sC from horizontal) if available
      - O_pool   (sO from pooled) if available

    Compute trial-wise projection y(n) as time-window-averaged projection:

      y_n = mean_t∈[winC_start, winC_end] s^T Z(n,t)

    Regress y_n on:
      - C_n = lab_C ∈ {-1,+1}
      - O_n = +1 for "vertical", -1 for "horizontal"
      - C×O

    via linear regression:

      y = β0 + βC C + βO O + βCO C*O + ε

    and compute full-model R^2.

Writes TSV:
  out/stim/diagnostics/axis_label_regression_stim.tsv

Columns:
  sid, area, axis_type, win_start, win_end,
  beta_C, beta_O, beta_CO, R2, n_trials
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple

import numpy as np


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


def load_axes_if_exists(root: Path, align: str, sid: str, area: str, tag: str):
    p = root / align / sid / "axes" / tag / f"axes_{area}.npz"
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


def regress_axis_on_labels(
    Z: np.ndarray,
    time_s: np.ndarray,
    s: np.ndarray,
    lab_C: np.ndarray,
    lab_O_str: np.ndarray,
    win: Tuple[float, float],
) -> Tuple[float, float, float, float, int]:
    """
    Z: (N,B,U), time_s: (B,), s: (U,), lab_C: (N,), lab_O_str: (N,)
    win: [start,end] in seconds
    """
    N, B, U = Z.shape
    s = s.reshape(-1)
    if s.size != U:
        raise ValueError(f"Axis dim {s.size} != n_units {U}")

    # build trial mask: need C and O
    C = lab_C.astype(float)
    O = np.full(N, np.nan, dtype=float)
    lab_O_str = lab_O_str.astype(str)
    O[lab_O_str == "vertical"] = +1.0
    O[lab_O_str == "horizontal"] = -1.0

    keep = np.isfinite(C) & np.isfinite(O)
    if not np.any(keep):
        return np.nan, np.nan, np.nan, np.nan, 0

    Zk = Z[keep]  # (Nk,B,U)
    Ck = C[keep]
    Ok = O[keep]

    # time window
    ws, we = win
    tmask = (time_s >= ws) & (time_s <= we)
    if not np.any(tmask):
        return np.nan, np.nan, np.nan, np.nan, 0

    # average projection in window
    # Zk: (Nk,B,U), s: (U,)
    # first average over time: (Nk,U)
    Xmean = Zk[:, tmask, :].mean(axis=1)  # (Nk,U)
    y = Xmean @ s  # (Nk,)

    # design matrix: [1, C, O, C*O]
    CO = Ck * Ok
    X = np.column_stack([
        np.ones_like(Ck),
        Ck,
        Ok,
        CO,
    ])  # (Nk,4)

    # solve least-squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum((y - y_hat)**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    beta0, betaC, betaO, betaCO = [float(b) for b in beta]
    n_trials = int(len(y))
    return betaC, betaO, betaCO, float(R2), n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--axes_tag_base", default="axes_sweep",
                    help="Base for axes tags (default: axes_sweep)")
    ap.add_argument("--align", choices=["stim"], default="stim",
                    help="Alignment to analyze (currently stim only)")
    ap.add_argument("--winC_stim", default="0.10:0.30",
                    help="Window [start:end] for C/O regression, in seconds (default: 0.10:0.30)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    align = args.align
    winC = tuple(float(x) for x in args.winC_stim.split(":"))

    sids = find_sessions(out_root, align)
    if not sids:
        print(f"[warn] no sessions under {out_root}/{align}")
        return

    diag_dir = out_root / align / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = diag_dir / "axis_label_regression_stim.tsv"

    print(f"[info] align={align}, sessions={len(sids)}")
    with out_tsv.open("w") as f:
        f.write("\t".join([
            "sid", "area", "axis_type",
            "win_start", "win_end",
            "beta_C", "beta_O", "beta_CO", "R2", "n_trials",
        ]) + "\n")

        for sid in sids:
            # find FEF area
            cache_dir = out_root / align / sid / "caches"
            area_files = sorted(cache_dir.glob("area_*.npz"))
            if not area_files:
                print(f"[skip] {sid}: no caches")
                continue
            areas = sorted([af.name[5:-4] for af in area_files])
            fef_candidates = [a for a in areas if "FEF" in a.upper()]
            if not fef_candidates:
                print(f"[skip] {sid}: no FEF area among {areas}")
                continue
            area = fef_candidates[0]

            print(f"\n[session {sid}] FEF area={area}")
            cache = load_cache(out_root, align, sid, area)
            Z = cache["Z"].astype(float)
            time_s = cache["time"].astype(float)
            lab_C = cache.get("lab_C", np.full(Z.shape[0], np.nan)).astype(float)
            lab_O = cache.get("lab_orientation", np.array(["pooled"]*Z.shape[0], dtype=object))

            # load axes
            axes_pooled = load_axes_if_exists(out_root, align, sid, area,
                                              f"{args.axes_tag_base}-stim-pooled")
            axes_vert = load_axes_if_exists(out_root, align, sid, area,
                                            f"{args.axes_tag_base}-stim-vertical")
            axes_horiz = load_axes_if_exists(out_root, align, sid, area,
                                             f"{args.axes_tag_base}-stim-horizontal")

            # meta windows: use winC from pooled if present
            if axes_pooled is not None:
                metaA = axes_pooled["meta"]
                # winC might be list or tuple
                try:
                    winC_meta = metaA.get("winC", None)
                    if winC_meta is not None:
                        ws, we = winC_meta
                        winC = (float(ws), float(we))
                except Exception:
                    pass

            def maybe_reg(axis_vec, axis_type):
                if axis_vec is None or axis_vec.size == 0:
                    print(f"  [{axis_type}] missing; skipping")
                    return
                betaC, betaO, betaCO, R2, n = regress_axis_on_labels(
                    Z, time_s, axis_vec, lab_C, lab_O, winC
                )
                print(f"  [{axis_type}] βC={betaC:.3f}, βO={betaO:.3f}, "
                      f"βCO={betaCO:.3f}, R2={R2:.3f}, N={n}")
                f.write("\t".join([
                    sid,
                    area,
                    axis_type,
                    f"{winC[0]:.3f}", f"{winC[1]:.3f}",
                    f"{betaC:.6f}",
                    f"{betaO:.6f}",
                    f"{betaCO:.6f}",
                    f"{R2:.6f}",
                    str(n),
                ]) + "\n")

            # pooled C axis
            if axes_pooled is not None:
                sC_pool = axes_pooled.get("sC", np.array([])).ravel()
                sO_pool = axes_pooled.get("sO", np.array([])).ravel()
                maybe_reg(sC_pool, "C_pool")
                maybe_reg(sO_pool, "O_pool")
            # vertical C axis
            if axes_vert is not None:
                sC_vert = axes_vert.get("sC", np.array([])).ravel()
                maybe_reg(sC_vert, "C_vert")
            # horizontal C axis
            if axes_horiz is not None:
                sC_horiz = axes_horiz.get("sC", np.array([])).ravel()
                maybe_reg(sC_horiz, "C_horiz")

    print(f"\n[done] wrote {out_tsv}")
