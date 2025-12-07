#!/usr/bin/env python3
"""
Run axes training, QC, and flow (single null+std combination) for one session.

Steps (per align):
  1) train_axes.py
  2) qc_axes.py
  3) flow_session.py (one combination of null_method + standardize_mode)
  4) flow_overlays.py

You can choose align=stim, align=sacc, or both. Features default to:
  - stim: C and R
  - sacc: S
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
    ap = argparse.ArgumentParser(description="Train axes, QC, and flow (one null+std combo) for a single session.")
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--sid", required=True,
                    help="Session ID (subdirectory under <out_root>/<align>/)")
    ap.add_argument("--align", choices=["stim", "sacc", "both"], default="stim",
                    help="Which alignment(s) to process (default: stim)")
    ap.add_argument("--features", nargs="+", default=None,
                    help="Which features to run flow for. "
                         "Default: ['C','R'] for stim; ['S'] for sacc.")
    ap.add_argument("--orientation", choices=["vertical", "horizontal", "pooled"],
                    default="vertical",
                    help="Orientation filter used in axes/trial masking (default: vertical)")
    ap.add_argument("--axes_tag", default="axes_final",
                    help="Tag for axes directory, e.g. axes_final (default: axes_final)")
    ap.add_argument("--flow_tag_base", default="crfinal",
                    help="Base for flow tag; final tag = <flow_tag_base>-<orientation>-<std>-<null> (default: crfinal)")
    ap.add_argument("--lags_ms_stim", type=float, default=50.0,
                    help="Lags (ms) for stim-aligned flows")
    ap.add_argument("--lags_ms_sacc", type=float, default=30.0,
                    help="Lags (ms) for sacc-aligned flows")
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="Ridge penalty (default: 1e-2)")
    ap.add_argument("--perms", type=int, default=500,
                    help="Number of permutations (default: 500)")
    ap.add_argument("--pt_min_ms_stim", type=float, default=200.0,
                    help="PT ≥ threshold (ms) for stim-align gating")
    ap.add_argument("--pt_min_ms_sacc", type=float, default=200.0,
                    help="PT ≥ threshold (ms) for sacc-align gating")
    ap.add_argument("--no_pt_filter", action="store_true",
                    help="Disable PT gating in axes/QC/flow")
    ap.add_argument("--perm-within", default="CR",
                    choices=["CR", "other", "C", "R", "none"],
                    help="Stratification for trial-shuffle null (default: CR)")
    ap.add_argument("--null_method",
                    default="trial_shuffle",
                    choices=["trial_shuffle", "circular_shift", "phase_randomize"],
                    help="Null method: trial_shuffle, circular_shift, or phase_randomize (default: trial_shuffle)")
    ap.add_argument("--standardize_mode",
                    default="none",
                    choices=["none", "zscore_regressors"],
                    help="Standardization of regressors: none or zscore_regressors (default: none)")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default: current)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands only; do not execute")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue remaining steps if a command fails")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    # Determine alignments to process
    if args.align == "both":
        aligns = ["stim", "sacc"]
    else:
        aligns = [args.align]

    # Map null/standardize to short tags (must match flow_session.py)
    null_short = {"trial_shuffle": "trial",
                  "circular_shift": "circ",
                  "phase_randomize": "phase"}[args.null_method]
    std_short = {"none": "none",
                 "zscore_regressors": "zreg"}[args.standardize_mode]
    # Include orientation in the tag so flows don't overwrite each other
    flow_tag_base_ori = f"{args.flow_tag_base}-{args.orientation}"
    flow_tag = f"{flow_tag_base_ori}-{std_short}-{null_short}"

    for align in aligns:
        # Choose features if not explicitly given
        if args.features is not None:
            feats = args.features
        else:
            feats = ["C", "R"] if align == "stim" else ["S"]

        # PT threshold per alignment
        if args.no_pt_filter:
            pt_thr = None
        else:
            pt_thr = args.pt_min_ms_stim if align == "stim" else args.pt_min_ms_sacc

        # 1) Train axes
        train_cmd = [args.python, CLI_TRAIN,
                     "--out_root", str(out_root),
                     "--align", align,
                     "--sid", args.sid,
                     "--orientation", args.orientation,
                     "--tag", args.axes_tag]
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

        # 2) QC axes
        qc_cmd = [args.python, CLI_QC,
                  "--out_root", str(out_root),
                  "--align", align,
                  "--sid", args.sid,
                  "--orientation", args.orientation,
                  "--tag", args.axes_tag]
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

        # 3) Flow per feature
        for feat in feats:
            lags_ms = args.lags_ms_stim if align == "stim" else args.lags_ms_sacc
            pt_flow = None if args.no_pt_filter else (args.pt_min_ms_stim if align == "stim" else args.pt_min_ms_sacc)

            flow_cmd = [args.python, CLI_FLOW,
                        "--out_root", str(out_root),
                        "--align", align,
                        "--sid", args.sid,
                        "--feature", feat,
                        "--orientation", args.orientation,
                        "--lags_ms", str(lags_ms),
                        "--ridge", str(args.ridge),
                        "--perms", str(args.perms),
                        "--perm-within", args.perm_within,
                        "--tag", flow_tag_base_ori,
                        "--flow_tag_base", flow_tag_base_ori,
                        "--axes_tag", args.axes_tag,
                        "--null_method", args.null_method,
                        "--standardize_mode", args.standardize_mode]
            if pt_flow is not None:
                flow_cmd += ["--pt_min_ms", str(pt_flow)]
            rc = run(flow_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
            if rc and not args.continue_on_error:
                return

            # 4) Overlays for that feature + flow_tag
            ovl_cmd = [args.python, CLI_OVL,
                       "--out_root", str(out_root),
                       "--align", align,
                       "--sid", args.sid,
                       "--feature", feat,
                       "--orientation", args.orientation,
                       "--tag", flow_tag]
            rc = run(ovl_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
            if rc and not args.continue_on_error:
                return

    print(f"[done] Session {args.sid} completed for null={args.null_method}, std={args.standardize_mode}.")


if __name__ == "__main__":
    main()
