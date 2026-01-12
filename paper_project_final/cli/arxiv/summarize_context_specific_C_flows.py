#!/usr/bin/env python3
"""
Summarize context-specific category flows across sessions (stim-align, C feature).

For each session (align=stim), for the FEF-LIP pair (MFEF-MLIP or SFEF-SLIP),
we compute window-averaged mean bits and mean z for:

  - VV: crsweep-stim-vertical-none-trial
  - HH: crsweep-stim-horizontal-none-trial
  - PP: crsweep-stim-pooled-none-trial
  - VP: crvertTrainPooled-stim-pooled-none-trial
  - HP: crhorizTrainPooled-stim-pooled-none-trial

For each config we extract:

  - FEF→LIP: mean_bits_AB, mean_z_AB
  - LIP→FEF: mean_bits_BA, mean_z_BA

over a specified time window (default 0.15–0.35 s), and write:

  out/stim/diagnostics/context_specific_C_flow_summary.tsv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


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


def load_flow_pair(
    out_root: Path,
    align: str,
    sid: str,
    feature: str,
    tag: str,
    A: str,
    B: str,
    win: Tuple[float, float],
):
    """
    Load flow files for A->B and B->A for given tag and feature.
    Returns (present_flag, stats) where stats is:
      (mean_bits_AB, mean_z_AB, mean_bits_BA, mean_z_BA)
      or NaNs if missing.
    """
    base = out_root / align / sid / "flow" / tag / feature
    p_fwd = base / f"flow_{feature}_{A}to{B}.npz"
    p_rev = base / f"flow_{feature}_{B}to{A}.npz"
    if not (p_fwd.is_file() and p_rev.is_file()):
        return False, (np.nan, np.nan, np.nan, np.nan)

    Zf = np.load(p_fwd, allow_pickle=True)
    Zr = np.load(p_rev, allow_pickle=True)
    time = Zf["time"].astype(float)
    bits_AB = Zf["bits_AtoB"].astype(float)
    mu_AB = Zf["null_mean_AtoB"].astype(float)
    sd_AB = Zf["null_std_AtoB"].astype(float)
    z_AB = np.full_like(bits_AB, np.nan, dtype=float)
    mask_AB = np.isfinite(bits_AB) & np.isfinite(mu_AB) & np.isfinite(sd_AB) & (sd_AB > 0)
    z_AB[mask_AB] = (bits_AB[mask_AB] - mu_AB[mask_AB]) / sd_AB[mask_AB]

    bits_BA = Zr["bits_AtoB"].astype(float)  # reverse file uses AtoB for B->A
    mu_BA = Zr["null_mean_AtoB"].astype(float)
    sd_BA = Zr["null_std_AtoB"].astype(float)
    z_BA = np.full_like(bits_BA, np.nan, dtype=float)
    mask_BA = np.isfinite(bits_BA) & np.isfinite(mu_BA) & np.isfinite(sd_BA) & (sd_BA > 0)
    z_BA[mask_BA] = (bits_BA[mask_BA] - mu_BA[mask_BA]) / sd_BA[mask_BA]

    ws, we = win
    wmask = (time >= ws) & (time <= we)
    if not np.any(wmask):
        return True, (np.nan, np.nan, np.nan, np.nan)

    mean_bits_AB = float(np.nanmean(bits_AB[wmask]))
    mean_z_AB = float(np.nanmean(z_AB[wmask]))
    mean_bits_BA = float(np.nanmean(bits_BA[wmask]))
    mean_z_BA = float(np.nanmean(z_BA[wmask]))

    return True, (mean_bits_AB, mean_z_AB, mean_bits_BA, mean_z_BA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--flow_tag_sweep_base", default="crsweep",
                    help="Base for standard sweep tags (vv,hh,pp) (default: crsweep)")
    ap.add_argument("--flow_tag_vertTrainPooled_base", default="crvertTrainPooled",
                    help="Base for VP tags (default: crvertTrainPooled)")
    ap.add_argument("--flow_tag_horizTrainPooled_base", default="crhorizTrainPooled",
                    help="Base for HP tags (default: crhorizTrainPooled)")
    ap.add_argument("--win", default="0.15:0.35",
                    help="Window [start:end] in seconds for averaging (default: 0.15:0.35)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    align = "stim"
    win = parse_window(args.win)

    sids = find_sessions(out_root, align)
    if not sids:
        print(f"[warn] no stim sessions under {out_root}")
        return

    diag_dir = out_root / align / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = diag_dir / "context_specific_C_flow_summary.tsv"

    print(f"[info] align={align}, sessions={len(sids)}")
    with out_tsv.open("w") as f:
        f.write("\t".join([
            "sid", "monkey_label", "pair",
            # VV
            "vv_mean_bits_F2L", "vv_mean_z_F2L",
            "vv_mean_bits_L2F", "vv_mean_z_L2F",
            # HH
            "hh_mean_bits_F2L", "hh_mean_z_F2L",
            "hh_mean_bits_L2F", "hh_mean_z_L2F",
            # PP
            "pp_mean_bits_F2L", "pp_mean_z_F2L",
            "pp_mean_bits_L2F", "pp_mean_z_L2F",
            # VP
            "vp_mean_bits_F2L", "vp_mean_z_F2L",
            "vp_mean_bits_L2F", "vp_mean_z_L2F",
            # HP
            "hp_mean_bits_F2L", "hp_mean_z_F2L",
            "hp_mean_bits_L2F", "hp_mean_z_L2F",
        ]) + "\n")

        for sid in sids:
            # find FEF/LIP areas
            cache_dir = out_root / align / sid / "caches"
            area_files = sorted(cache_dir.glob("area_*.npz"))
            if not area_files:
                print(f"[skip] {sid}: no caches")
                continue
            areas = sorted([af.name[5:-4] for af in area_files])
            fef_candidates = [a for a in areas if "FEF" in a.upper()]
            lip_candidates = [a for a in areas if "LIP" in a.upper()]
            if not (fef_candidates and lip_candidates):
                print(f"[skip] {sid}: no FEF/LIP pair among {areas}")
                continue
            A = fef_candidates[0]
            B = lip_candidates[0]
            monkey_label = A[0].upper()

            print(f"\n[session {sid}] monkey={monkey_label}, pair={A}->{B}")

            # tags
            vv_tag = f"{args.flow_tag_sweep_base}-stim-vertical-none-trial"
            hh_tag = f"{args.flow_tag_sweep_base}-stim-horizontal-none-trial"
            pp_tag = f"{args.flow_tag_sweep_base}-stim-pooled-none-trial"
            vp_tag = f"{args.flow_tag_vertTrainPooled_base}-stim-pooled-none-trial"
            hp_tag = f"{args.flow_tag_horizTrainPooled_base}-stim-pooled-none-trial"

            def get_stats(tag):
                present, stats = load_flow_pair(out_root, align, sid, "C", tag, A, B, win)
                return stats

            vv_bits_F2L, vv_z_F2L, vv_bits_L2F, vv_z_L2F = get_stats(vv_tag)
            hh_bits_F2L, hh_z_F2L, hh_bits_L2F, hh_z_L2F = get_stats(hh_tag)
            pp_bits_F2L, pp_z_F2L, pp_bits_L2F, pp_z_L2F = get_stats(pp_tag)
            vp_bits_F2L, vp_z_F2L, vp_bits_L2F, vp_z_L2F = get_stats(vp_tag)
            hp_bits_F2L, hp_z_F2L, hp_bits_L2F, hp_z_L2F = get_stats(hp_tag)

            row = [
                sid,
                monkey_label,
                f"{A}-{B}",
                f"{vv_bits_F2L:.6f}", f"{vv_z_F2L:.6f}",
                f"{vv_bits_L2F:.6f}", f"{vv_z_L2F:.6f}",
                f"{hh_bits_F2L:.6f}", f"{hh_z_F2L:.6f}",
                f"{hh_bits_L2F:.6f}", f"{hh_z_L2F:.6f}",
                f"{pp_bits_F2L:.6f}", f"{pp_z_F2L:.6f}",
                f"{pp_bits_L2F:.6f}", f"{pp_z_L2F:.6f}",
                f"{vp_bits_F2L:.6f}", f"{vp_z_F2L:.6f}",
                f"{vp_bits_L2F:.6f}", f"{vp_z_L2F:.6f}",
                f"{hp_bits_F2L:.6f}", f"{hp_z_F2L:.6f}",
                f"{hp_bits_L2F:.6f}", f"{hp_z_L2F:.6f}",
            ]
            print("  vv F→L z=", vv_z_F2L, "vp F→L z=", vp_z_F2L, "hp F→L z=", hp_z_F2L)
            f.write("\t".join(row) + "\n")

    print(f"\n[done] wrote {out_tsv}")


if __name__ == "__main__":
    main()
