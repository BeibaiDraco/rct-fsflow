#!/usr/bin/env python3
"""
Run axes (horizontal-only), QC (horizontal), and flow (pooled trials)
for ONE session (both stim & sacc, if caches exist).

For each align in {stim, sacc}:

  1) train_axes.py         (orientation=horizontal)
  2) qc_axes.py            (orientation=horizontal)
  3) flow_session.py       (orientation=pooled, but axes_tag from horizontal training)
  4) flow_overlays.py      (orientation=pooled)

Flow tags encode:
  flow_tag_base_align_pooled = <flow_tag_base>-<align>-pooled
  flow_tag                   = <flow_tag_base_align_pooled>-<std_short>-<null_short>

So these are distinct from your standard crsweep-* tags and from the vertical-trained
crvertTrainPooled-* tags.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

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
        description="Horizontal-only axes, pooled-trial flow, for one session (stim & sacc)."
    )
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--sid", required=True,
                    help="Session ID (under out/stim and/or out/sacc)")
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
    ap.add_argument("--axes_tag_base", default="axes_horizTrain",
                    help="Base tag for horizontal-trained axes; "
                         "final tag = <axes_tag_base>-<align>-horizontal (default: axes_horizTrain)")
    ap.add_argument("--flow_tag_base", default="crhorizTrainPooled",
                    help="Base for pooled-flow tags; "
                         "final base = <flow_tag_base>-<align>-pooled (default: crhorizTrainPooled)")
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
                    help="Continue remaining combos if a command fails")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    std_modes = args.standardize_modes or ["none", "zscore_regressors"]
    null_methods = args.null_methods or ["trial_shuffle", "circular_shift"]

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
    print(f"[info] standardize_modes = {std_modes}")
    print(f"[info] null_methods      = {null_methods}")

    # detect which alignments exist
    aligns: List[str] = []
    if (out_root / "stim" / args.sid / "caches").is_dir():
        aligns.append("stim")
    if (out_root / "sacc" / args.sid / "caches").is_dir():
        aligns.append("sacc")
    if not aligns:
        print(f"[WARN] No caches found for SID={args.sid} under {out_root}/stim or /sacc")
        return

    for align in aligns:
        print(f"\n[align={align}] SID={args.sid}")

        # axes: horizontal only
        train_ori = "horizontal"
        qc_ori = "horizontal"
        flow_ori = "pooled"  # pooled trials (orientation=None inside flow)

        axes_tag = f"{args.axes_tag_base}-{align}-{train_ori}"

        # --- 1) train_axes (horizontal only) ---
        train_cmd = [args.python, CLI_TRAIN,
                     "--out_root", str(out_root),
                     "--align", align,
                     "--sid", args.sid,
                     "--orientation", train_ori,
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

        # --- 2) qc_axes (horizontal only) ---
        qc_cmd = [args.python, CLI_QC,
                  "--out_root", str(out_root),
                  "--align", align,
                  "--sid", args.sid,
                  "--orientation", qc_ori,
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

        # features per align
        feats = ["C", "R"] if align == "stim" else ["S"]
        lags_ms = args.lags_ms_stim if align == "stim" else args.lags_ms_sacc
        pt_thr = None if args.no_pt_filter else (
            args.pt_min_ms_stim if align == "stim" else args.pt_min_ms_sacc
        )

        for feat in feats:
            for std_mode in std_modes:
                for null_method in null_methods:
                    std_short = std_short_map[std_mode]
                    null_short = null_short_map[null_method]
                    flow_tag_base_align_pooled = f"{args.flow_tag_base}-{align}-pooled"
                    flow_tag = f"{flow_tag_base_align_pooled}-{std_short}-{null_short}"

                    print(f"\n[combo] SID={args.sid}, align={align}, "
                          f"train_ori={train_ori}, flow_ori={flow_ori}, "
                          f"feat={feat}, std={std_mode}, null={null_method} "
                          f"→ flow_tag={flow_tag}")

                    # --- 3) flow_session (pooled trials, horizontal axes) ---
                    flow_cmd = [args.python, CLI_FLOW,
                                "--out_root", str(out_root),
                                "--align", align,
                                "--sid", args.sid,
                                "--feature", feat,
                                "--orientation", flow_ori,        # pooled
                                "--lags_ms", str(lags_ms),
                                "--ridge", str(args.ridge),
                                "--perms", str(args.perms),
                                "--perm-within", args.perm_within,
                                "--tag", flow_tag_base_align_pooled,
                                "--flow_tag_base", flow_tag_base_align_pooled,
                                "--axes_tag", axes_tag,
                                "--null_method", null_method,
                                "--standardize_mode", std_mode]
                    if pt_thr is not None:
                        flow_cmd += ["--pt_min_ms", str(pt_thr)]
                    rc = run(flow_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
                    if rc and not args.continue_on_error:
                        return

                    # --- 4) overlays (pooled) ---
                    ovl_cmd = [args.python, CLI_OVL,
                               "--out_root", str(out_root),
                               "--align", align,
                               "--sid", args.sid,
                               "--feature", feat,
                               "--orientation", flow_ori,
                               "--tag", flow_tag]
                    rc = run(ovl_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
                    if rc and not args.continue_on_error:
                        return

    print(f"\n[done] SID={args.sid} horizontal-axes + pooled-flow sweep complete.")


if __name__ == "__main__":
    main()
