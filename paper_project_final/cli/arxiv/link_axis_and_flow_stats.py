#!/usr/bin/env python3
"""
Link FEF axis label regression stats with context-specific C-flow stats across sessions.

Inputs:
  - out/stim/diagnostics/axis_label_regression_stim.tsv
    (from analyze_axis_label_regression.py)
  - out/stim/diagnostics/context_specific_C_flow_summary.tsv
    (from summarize_context_specific_C_flows.py)

For each session SID and for the FEF area (MFEF or SFEF), we extract:

  Axis regression (C_pool, C_vert, C_horiz):
    - beta_C, beta_O, beta_CO, R2

  Flow stats (FEF->LIP) in a given window:
    - vv_mean_z_F2L, hh_mean_z_F2L, pp_mean_z_F2L,
      vp_mean_z_F2L, hp_mean_z_F2L

We then write a combined TSV:

  out/stim/diagnostics/axis_flow_link_stim.tsv

Columns:
  sid, monkey_label, pair,
  C_pool_beta_C,  C_pool_beta_O,  C_pool_beta_CO,  C_pool_R2,
  C_vert_beta_C,  C_vert_beta_O,  C_vert_beta_CO,  C_vert_R2,
  C_horiz_beta_C, C_horiz_beta_O, C_horiz_beta_CO, C_horiz_R2,
  vv_z_F2L, hh_z_F2L, pp_z_F2L, vp_z_F2L, hp_z_F2L

The script also prints Pearson correlations between vp_z_F2L and:
  - C_vert_beta_CO
  - C_vert_beta_CO - C_pool_beta_CO
  - C_vert_R2 - C_pool_R2
"""

from __future__ import annotations
import argparse
from pathlib import Path
import csv
import math
from typing import Dict, Any, List

import numpy as np


def load_axis_reg_table(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load axis_label_regression_stim.tsv into:
      axis_stats[sid][axis_type] = dict(beta_C, beta_O, beta_CO, R2)
    for FEF area only.
    """
    axis_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    with path.open("r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["sid"]
            area = row["area"]
            axis_type = row["axis_type"]  # e.g. C_pool, O_pool, C_vert, C_horiz
            # keep only FEF rows
            if "FEF" not in area.upper():
                continue
            beta_C = float(row["beta_C"])
            beta_O = float(row["beta_O"])
            beta_CO = float(row["beta_CO"])
            R2 = float(row["R2"])
            axis_stats.setdefault(sid, {})[axis_type] = dict(
                beta_C=beta_C,
                beta_O=beta_O,
                beta_CO=beta_CO,
                R2=R2,
            )
    return axis_stats


def load_flow_table(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load context_specific_C_flow_summary.tsv into:
      flow_stats[sid] = dict(monkey_label, pair, vv_z_F2L, ..., hp_z_F2L)
    """
    flow_stats: Dict[str, Dict[str, Any]] = {}
    with path.open("r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["sid"]
            flow_stats[sid] = dict(
                monkey_label=row["monkey_label"],
                pair=row["pair"],
                vv_z_F2L=float(row["vv_mean_z_F2L"]),
                hh_z_F2L=float(row["hh_mean_z_F2L"]),
                pp_z_F2L=float(row["pp_mean_z_F2L"]),
                vp_z_F2L=float(row["vp_mean_z_F2L"]),
                hp_z_F2L=float(row["hp_mean_z_F2L"]),
            )
    return flow_stats


def pearson_corr(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson correlation between x and y (ignoring NaNs).
    """
    arr_x = np.array(x, dtype=float)
    arr_y = np.array(y, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if mask.sum() < 3:
        return float("nan")
    xm = arr_x[mask].mean()
    ym = arr_y[mask].mean()
    num = np.sum((arr_x[mask] - xm) * (arr_y[mask] - ym))
    den = math.sqrt(np.sum((arr_x[mask] - xm)**2) * np.sum((arr_y[mask] - ym)**2))
    if den <= 0:
        return float("nan")
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--axis_reg_tsv",
                    default="out/stim/diagnostics/axis_label_regression_stim.tsv",
                    help="Path to axis_label_regression_stim.tsv")
    ap.add_argument("--flow_tsv",
                    default="out/stim/diagnostics/context_specific_C_flow_summary.tsv",
                    help="Path to context_specific_C_flow_summary.tsv")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    axis_reg_path = Path(args.axis_reg_tsv)
    flow_path = Path(args.flow_tsv)

    if not axis_reg_path.is_file():
        raise SystemExit(f"Axis regression TSV not found: {axis_reg_path}")
    if not flow_path.is_file():
        raise SystemExit(f"Flow summary TSV not found: {flow_path}")

    axis_stats = load_axis_reg_table(axis_reg_path)
    flow_stats = load_flow_table(flow_path)

    # Prepare output
    diag_dir = out_root / "stim" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = diag_dir / "axis_flow_link_stim.tsv"

    rows = []
    for sid, fstats in flow_stats.items():
        if sid not in axis_stats:
            # no FEF axis stats for this sid
            continue
        astats = axis_stats[sid]
        monkey_label = fstats["monkey_label"]
        pair = fstats["pair"]

        def get_axis(axis_type: str):
            d = astats.get(axis_type, None)
            if d is None:
                return (float("nan"), float("nan"), float("nan"), float("nan"))
            return (d["beta_C"], d["beta_O"], d["beta_CO"], d["R2"])

        C_pool_stats = get_axis("C_pool")
        C_vert_stats = get_axis("C_vert")
        C_horiz_stats = get_axis("C_horiz")

        row = {
            "sid": sid,
            "monkey_label": monkey_label,
            "pair": pair,
            "C_pool_beta_C": C_pool_stats[0],
            "C_pool_beta_O": C_pool_stats[1],
            "C_pool_beta_CO": C_pool_stats[2],
            "C_pool_R2": C_pool_stats[3],
            "C_vert_beta_C": C_vert_stats[0],
            "C_vert_beta_O": C_vert_stats[1],
            "C_vert_beta_CO": C_vert_stats[2],
            "C_vert_R2": C_vert_stats[3],
            "C_horiz_beta_C": C_horiz_stats[0],
            "C_horiz_beta_O": C_horiz_stats[1],
            "C_horiz_beta_CO": C_horiz_stats[2],
            "C_horiz_R2": C_horiz_stats[3],
            "vv_z_F2L": fstats["vv_z_F2L"],
            "hh_z_F2L": fstats["hh_z_F2L"],
            "pp_z_F2L": fstats["pp_z_F2L"],
            "vp_z_F2L": fstats["vp_z_F2L"],
            "hp_z_F2L": fstats["hp_z_F2L"],
        }
        rows.append(row)

    # Write combined TSV
    fieldnames = [
        "sid", "monkey_label", "pair",
        "C_pool_beta_C", "C_pool_beta_O", "C_pool_beta_CO", "C_pool_R2",
        "C_vert_beta_C", "C_vert_beta_O", "C_vert_beta_CO", "C_vert_R2",
        "C_horiz_beta_C", "C_horiz_beta_O", "C_horiz_beta_CO", "C_horiz_R2",
        "vv_z_F2L", "hh_z_F2L", "pp_z_F2L", "vp_z_F2L", "hp_z_F2L",
    ]
    with out_tsv.open("w") as f_out:
        writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[done] wrote {out_tsv}")

    # Compute correlations
    vp_z = [r["vp_z_F2L"] for r in rows]
    C_vert_beta_CO = [r["C_vert_beta_CO"] for r in rows]
    C_pool_beta_CO = [r["C_pool_beta_CO"] for r in rows]
    C_vert_R2 = [r["C_vert_R2"] for r in rows]
    C_pool_R2 = [r["C_pool_R2"] for r in rows]

    diff_betaCO = [v - p for v, p in zip(C_vert_beta_CO, C_pool_beta_CO)]
    diff_R2 = [v - p for v, p in zip(C_vert_R2, C_pool_R2)]

    print("\n[correlations with vp_z_F2L]")
    print("  corr(vp_z_F2L, C_vert_beta_CO)    =",
          pearson_corr(vp_z, C_vert_beta_CO))
    print("  corr(vp_z_F2L, C_vert_beta_CO - C_pool_beta_CO) =",
          pearson_corr(vp_z, diff_betaCO))
    print("  corr(vp_z_F2L, C_vert_R2 - C_pool_R2)          =",
          pearson_corr(vp_z, diff_R2))


if __name__ == "__main__":
    main()