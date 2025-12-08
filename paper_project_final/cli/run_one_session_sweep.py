#!/usr/bin/env python3
"""
Sweep flow parameters (null method, standardization, orientation)
for ONE session (both alignments, stim & sacc).

For a given SID, we do:

for align in {stim, sacc}:
  for ori in {vertical, horizontal, pooled}:
    1) train_axes.py
    2) qc_axes.py
    3) for features:
         - stim: C and R
         - sacc: S
       for each (standardize_mode, null_method):
         - flow_session.py
         - flow_overlays.py

Flow tags encode align + orientation + std + null:
  flow_tag_base_align_ori = <flow_tag_base>-<align>-<ori>
  flow_tag                = <flow_tag_base_align_ori>-<std_short>-<null_short>
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
    ap = argparse.ArgumentParser(
        description="Run axes/QC/flow/overlays for one session, "
                    "sweeping null_method × standardize_mode × orientation."
    )
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--sid", required=True,
                    help="Session ID (subdirectory under out/stim and/or out/sacc)")
    ap.add_argument("--lags_ms_stim", type=float, default=50.0,
                    help="Lags (ms) for stim-aligned flows (default: 50)")
    ap.add_argument("--lags_ms_sacc", type=float, default=30.0,
                    help="Lags (ms) for sacc-aligned flows (default: 30)")
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="Ridge penalty (default: 1e-2)")
    ap.add_argument("--perms", type=int, default=500,
                    help="Number of permutations (default: 500)")
    ap.add_argument("--pt_min_ms_stim", type=float, default=200.0,
                    help="PT ≥ threshold (ms) for stim-align (default: 200)")
    ap.add_argument("--pt_min_ms_sacc", type=float, default=200.0,
                    help="PT ≥ threshold (ms) for sacc-align (default: 200)")
    ap.add_argument("--no_pt_filter", action="store_true",
                    help="Disable PT gating for axes/QC/flow")
    ap.add_argument("--perm-within", default="CR",
                    choices=["CR", "other", "C", "R", "none"],
                    help="Stratification for trial-shuffle null (default: CR)")
    ap.add_argument("--axes_tag_base", default="axes_sweep",
                    help="Base tag for axes; final tag = <axes_tag_base>-<align>-<ori> (default: axes_sweep)")
    ap.add_argument("--flow_tag_base", default="crsweep",
                    help="Base for flow; final base = <flow_tag_base>-<align>-<ori> (default: crsweep)")
    ap.add_argument("--orientations", nargs="*",
                    choices=["vertical", "horizontal", "pooled"],
                    default=None,
                    help="Orientations to sweep. Default: ['vertical','horizontal','pooled']")
    ap.add_argument("--standardize_modes", nargs="*",
                    choices=["none", "zscore_regressors"],
                    default=None,
                    help="Standardization modes to sweep. Default: ['none','zscore_regressors']")
    ap.add_argument("--null_methods", nargs="*",
                    choices=["trial_shuffle", "circular_shift", "phase_randomize"],
                    default=None,
                    help="Null methods to sweep. Default: ['trial_shuffle','circular_shift']")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default: current)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands only; do not execute")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue remaining jobs if a command fails")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    # Defaults
    orientations = args.orientations or ["vertical", "horizontal", "pooled"]
    std_modes = args.standardize_modes or ["none", "zscore_regressors"]
    null_methods = args.null_methods or ["trial_shuffle", "circular_shift"]

    # Short-tag mapping (must match flow_session.py)
    null_short_map = {
        "trial_shuffle": "trial",
        "circular_shift": "circ",
        "phase_randomize": "phase",
    }
    std_short_map = {
        "none": "none",
        "zscore_regressors": "zreg",
    }

    print(f"[info] SID={args.sid}")
    print(f"[info] orientations      = {orientations}")
    print(f"[info] standardize_modes = {std_modes}")
    print(f"[info] null_methods      = {null_methods}")

    # Top-level alignments
    aligns = []
    if (out_root / "stim" / args.sid / "caches").is_dir():
        aligns.append("stim")
    if (out_root / "sacc" / args.sid / "caches").is_dir():
        aligns.append("sacc")
    if not aligns:
        print(f"[WARN] No caches found for SID {args.sid} under {out_root}/stim or /sacc")
        return

    for align in aligns:
        print(f"\n[align={align}] SID={args.sid}")
        for ori in orientations:
            print(f"\n[block] ori={ori}")

            # Axes tag per (align, ori)
            axes_tag = f"{args.axes_tag_base}-{align}-{ori}"

            # --- 1) train_axes ---
            train_cmd = [args.python, CLI_TRAIN,
                         "--out_root", str(out_root),
                         "--align", align,
                         "--sid", args.sid,
                         "--orientation", ori,
                         "--tag", axes_tag]
            if args.no_pt_filter:
                train_cmd.append("--no_pt_filter")
            else:
                if align == "stim":
                    train_cmd += ["--pt_min_ms_stim", str(args.pt_min_ms_stim)]
                else:
                    train_cmd += ["--pt_min_ms_sacc", str(args.pt_min_ms_sacc)]
            rc = run(train_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
            if rc and not args.continue_on_error:
                return

            # --- 2) qc_axes ---
            qc_cmd = [args.python, CLI_QC,
                      "--out_root", str(out_root),
                      "--align", align,
                      "--sid", args.sid,
                      "--orientation", ori,
                      "--tag", axes_tag]
            if args.no_pt_filter:
                qc_cmd.append("--no_pt_filter")
            else:
                if align == "stim":
                    qc_cmd += ["--pt_min_ms_stim", str(args.pt_min_ms_stim)]
                else:
                    qc_cmd += ["--pt_min_ms_sacc", str(args.pt_min_ms_sacc)]
            rc = run(qc_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
            if rc and not args.continue_on_error:
                return

            # Features per align
            if align == "stim":
                feats = ["C", "R", "O"]  # add context flow
            else:
                feats = ["S"]            # you can add "O" here too later if you want
            lags_ms = args.lags_ms_stim if align == "stim" else args.lags_ms_sacc
            pt_thr = None if args.no_pt_filter else (
                args.pt_min_ms_stim if align == "stim" else args.pt_min_ms_sacc
            )

            for feat in feats:
                for std_mode in std_modes:
                    for null_method in null_methods:
                        std_short = std_short_map[std_mode]
                        null_short = null_short_map[null_method]
                        flow_tag_base_align_ori = f"{args.flow_tag_base}-{align}-{ori}"
                        flow_tag = f"{flow_tag_base_align_ori}-{std_short}-{null_short}"

                        print(f"\n[combo] sid={args.sid}, align={align}, ori={ori}, "
                              f"feat={feat}, std={std_mode}, null={null_method} "
                              f"→ flow_tag={flow_tag}")

                        # --- 3) flow_session ---
                        flow_cmd = [args.python, CLI_FLOW,
                                    "--out_root", str(out_root),
                                    "--align", align,
                                    "--sid", args.sid,
                                    "--feature", feat,
                                    "--orientation", ori,
                                    "--lags_ms", str(lags_ms),
                                    "--ridge", str(args.ridge),
                                    "--perms", str(args.perms),
                                    "--perm-within", args.perm_within,
                                    "--tag", flow_tag_base_align_ori,
                                    "--flow_tag_base", flow_tag_base_align_ori,
                                    "--axes_tag", axes_tag,
                                    "--null_method", null_method,
                                    "--standardize_mode", std_mode]
                        if pt_thr is not None:
                            flow_cmd += ["--pt_min_ms", str(pt_thr)]
                        rc = run(flow_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
                        if rc and not args.continue_on_error:
                            return

                        # --- 4) flow_overlays ---
                        ovl_cmd = [args.python, CLI_OVL,
                                   "--out_root", str(out_root),
                                   "--align", align,
                                   "--sid", args.sid,
                                   "--feature", feat,
                                   "--orientation", ori,
                                   "--tag", flow_tag]
                        rc = run(ovl_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
                        if rc and not args.continue_on_error:
                            return

    print(f"\n[done] SID={args.sid} sweep complete.")


if __name__ == "__main__":
    main()
