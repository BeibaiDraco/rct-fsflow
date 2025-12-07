#!/usr/bin/env python3
"""
Sweep flow parameters for a single session (stim-align, feature C).

For each orientation in {vertical, horizontal, pooled} (configurable):
  1) Train axes (stim, that orientation)
  2) QC axes
  3) For each (standardize_mode, null_method) combination:
       - run flow_session.py (align=stim, feature=C)
       - run flow_overlays.py for that flow_tag

Flow tags use the convention:
  flow_tag = <flow_tag_base>-<orientation>-<std_short>-<null_short>
where:
  orientation = 'vertical', 'horizontal', or 'pooled'
  std_short  = 'none' or 'zreg'
  null_short = 'trial', 'circ', or 'phase'
and must match the logic in flow_session.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

CLI_TRAIN = "cli/train_axes.py"
CLI_QC    = "cli/qc_axes.py"
CLI_FLOW  = "cli/flow_session.py"
CLI_OVL   = "cli/flow_overlays.py"


def run(cmd, dry=False, continue_on_error=False):
    print("[RUN]", " ".join(cmd))
    if dry:
        return 0
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] command failed (exit {e.returncode}): {' '.join(cmd)}", file=sys.stderr)
        if continue_on_error:
            return e.returncode
        raise


def main():
    ap = argparse.ArgumentParser(description="Sweep null/standardization/orientation for stim-align C-flow on one session.")
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--sid", required=True,
                    help="Session ID to process (under out/stim/<sid>/caches)")
    ap.add_argument("--axes_tag_base", default="axes_sweep",
                    help="Base tag for axes; final axes_tag = <axes_tag_base>-<orientation> (default: axes_sweep)")
    ap.add_argument("--flow_tag_base", default="crsweep",
                    help="Base tag for flow; final flow_tag = <flow_tag_base>-<orientation>-<std>-<null> (default: crsweep)")
    ap.add_argument("--lags_ms", type=float, default=50.0,
                    help="Lags (ms) for stim-aligned C flow (default: 50)")
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="Ridge penalty (default: 1e-2)")
    ap.add_argument("--perms", type=int, default=500,
                    help="Number of permutations (default: 500)")
    ap.add_argument("--pt_min_ms", type=float, default=200.0,
                    help="PT ≥ threshold (ms) for stim-align gating (default: 200)")
    ap.add_argument("--no_pt_filter", action="store_true",
                    help="Disable PT gating")
    ap.add_argument("--perm-within", default="CR",
                    choices=["CR", "other", "C", "R", "none"],
                    help="Stratification for trial-shuffle null (default: CR)")
    ap.add_argument("--standardize_modes", nargs="*",
                    choices=["none", "zscore_regressors"],
                    default=None,
                    help="Standardization modes to sweep. Default: ['none','zscore_regressors']")
    ap.add_argument("--null_methods", nargs="*",
                    choices=["trial_shuffle", "circular_shift", "phase_randomize"],
                    default=None,
                    help="Null methods to sweep. Default: ['trial_shuffle','circular_shift']")
    ap.add_argument("--orientations", nargs="*",
                    choices=["vertical", "horizontal", "pooled"],
                    default=None,
                    help="Orientations to sweep. Default: ['vertical','horizontal','pooled']")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default: current)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands only; do not execute")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue remaining combos if a command fails")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    # Defaults if user didn't specify
    std_modes = args.standardize_modes or ["none", "zscore_regressors"]
    null_methods = args.null_methods or ["trial_shuffle", "circular_shift"]
    orientations = args.orientations or ["vertical", "horizontal", "pooled"]

    print(f"[info] Sweeping session {args.sid} (align=stim, feature=C)")
    print(f"[info] orientations      = {orientations}")
    print(f"[info] standardize_modes = {std_modes}")
    print(f"[info] null_methods      = {null_methods}")

    # Short-tag mapping (must match flow_session.py)
    null_short_map = {"trial_shuffle": "trial",
                      "circular_shift": "circ",
                      "phase_randomize": "phase"}
    std_short_map = {"none": "none",
                     "zscore_regressors": "zreg"}

    for ori in orientations:
        print(f"\n[block] Orientation = {ori}")

        # Axes tag per orientation so they don't collide
        axes_tag = f"{args.axes_tag_base}-{ori}"

        # 1) Train axes (stim, this orientation)
        train_cmd = [args.python, CLI_TRAIN,
                     "--out_root", str(out_root),
                     "--align", "stim",
                     "--sid", args.sid,
                     "--orientation", ori,
                     "--tag", axes_tag]
        if args.no_pt_filter:
            train_cmd.append("--no_pt_filter")
        else:
            train_cmd += ["--pt_min_ms_stim", str(args.pt_min_ms)]
        rc = run(train_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error:
            return

        # 2) QC axes
        qc_cmd = [args.python, CLI_QC,
                  "--out_root", str(out_root),
                  "--align", "stim",
                  "--sid", args.sid,
                  "--orientation", ori,
                  "--tag", axes_tag]
        if args.no_pt_filter:
            qc_cmd.append("--no_pt_filter")
        else:
            qc_cmd += ["--pt_min_ms_stim", str(args.pt_min_ms)]
        rc = run(qc_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error:
            return

        # 3) Sweep (standardize_mode, null_method)
        for std_mode in std_modes:
            for null_method in null_methods:
                std_short = std_short_map[std_mode]
                null_short = null_short_map[null_method]
                # Include orientation in the tag so flows don't overwrite each other
                flow_tag_base_ori = f"{args.flow_tag_base}-{ori}"
                flow_tag = f"{flow_tag_base_ori}-{std_short}-{null_short}"

                print(f"\n[combo] ori={ori}, std={std_mode}, null={null_method} → flow_tag={flow_tag}")

                # Flow: stim-align, feature C
                flow_cmd = [args.python, CLI_FLOW,
                            "--out_root", str(out_root),
                            "--align", "stim",
                            "--sid", args.sid,
                            "--feature", "C",
                            "--orientation", ori,
                            "--lags_ms", str(args.lags_ms),
                            "--ridge", str(args.ridge),
                            "--perms", str(args.perms),
                            "--perm-within", args.perm_within,
                            "--tag", flow_tag_base_ori,          # base
                            "--flow_tag_base", flow_tag_base_ori,
                            "--axes_tag", axes_tag,
                            "--null_method", null_method,
                            "--standardize_mode", std_mode]
                if not args.no_pt_filter:
                    flow_cmd += ["--pt_min_ms", str(args.pt_min_ms)]
                rc = run(flow_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
                if rc and not args.continue_on_error:
                    return

                # Overlays
                ovl_cmd = [args.python, CLI_OVL,
                           "--out_root", str(out_root),
                           "--align", "stim",
                           "--sid", args.sid,
                           "--feature", "C",
                           "--orientation", ori,
                           "--tag", flow_tag]
                rc = run(ovl_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
                if rc and not args.continue_on_error:
                    return

    print(f"\n[done] Completed sweep for session {args.sid} (stim, C).")


if __name__ == "__main__":
    main()
