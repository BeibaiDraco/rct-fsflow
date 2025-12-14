#!/usr/bin/env python3
"""
Run old-style evoked subtraction flow analysis for ONE session.

For each alignment (stim, sacc, targ):
  - stim: feature C (default)
  - sacc: feature S (default)
  - targ: feature T (target configuration aligned to targets onset)

Uses pre-existing axes (axes_sweep-<align>-<axes_orientation>).
Only runs flow_session.py (assumes axes already exist).

Supports:
  --orientation: flow orientation (vertical, horizontal, pooled)
  --axes_orientation: which axes to load (defaults to --orientation)

Example: vertical axes + pooled flow:
  --axes_orientation vertical --orientation pooled

Example: run T flow aligned to targets onset:
  --targ_feature T --train_axes
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

CLI_FLOW = "cli/flow_session.py"
CLI_TRAIN = "cli/train_axes.py"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run(cmd, dry=False, continue_on_error=False):
    print("[RUN]", " ".join(cmd))
    if dry:
        return 0
    try:
        env = os.environ.copy()
        # Make CLI scripts runnable even if PYTHONPATH wasn't set by the caller.
        # (Cluster sbatch scripts already do this, but local runs often don't.)
        pp = str(PROJECT_ROOT)
        env["PYTHONPATH"] = pp + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
        res = subprocess.run(cmd, check=True, env=env)
        return res.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] command failed (exit {e.returncode}): {' '.join(cmd)}", file=sys.stderr)
        if continue_on_error:
            return e.returncode
        raise


def main():
    ap = argparse.ArgumentParser(
        description="Run evoked subtraction flow for one session (stim: C, sacc: S, targ: T)."
    )
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc/targ live (default: out)")
    ap.add_argument("--sid", required=True,
                    help="Session ID (under out/stim, out/sacc, and/or out/targ)")
    ap.add_argument("--lags_ms_stim", type=float, default=50.0,
                    help="Lags (ms) for stim-aligned flows (default: 50)")
    ap.add_argument("--lags_ms_sacc", type=float, default=30.0,
                    help="Lags (ms) for sacc-aligned flows (default: 30)")
    ap.add_argument("--lags_ms_targ", type=float, default=30.0,
                    help="Lags (ms) for targ-aligned flows (default: 30)")
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="Ridge penalty (default: 1e-2)")
    ap.add_argument("--perms", type=int, default=500,
                    help="Number of permutations (default: 500)")
    ap.add_argument("--pt_min_ms", type=float, default=200.0,
                    help="PT ≥ threshold (ms) (default: 200)")
    ap.add_argument("--perm-within", default="CR",
                    choices=["CR", "other", "C", "R", "none"],
                    help="Stratification for trial-shuffle null (default: CR)")
    ap.add_argument("--axes_tag_base", default="axes_sweep",
                    help="Base for axes tags; final = <base>-<align>-<orientation> (default: axes_sweep)")
    ap.add_argument("--flow_tag_base", default="evoked_subtract",
                    help="Base for flow tags (default: evoked_subtract)")
    ap.add_argument("--orientation", default="vertical",
                    choices=["vertical", "horizontal", "pooled"],
                    help="Orientation for flow (default: vertical)")
    ap.add_argument("--axes_orientation", default=None,
                    choices=["vertical", "horizontal", "pooled"],
                    help="Orientation for axes tag; defaults to --orientation if not specified")
    ap.add_argument("--evoked_sigma_ms", type=float, default=10.0,
                    help="Gaussian smoothing sigma for evoked PSTH (ms) (default: 10)")
    ap.add_argument("--stim_feature", default="C",
                    choices=["C", "R", "T", "O", "none"],
                    help="Feature to run for stim alignment; use 'none' to skip stim (default: C)")
    ap.add_argument("--sacc_feature", default="S",
                    choices=["S", "C", "T", "O", "none"],
                    help="Feature to run for sacc alignment; use 'none' to skip sacc (default: S)")
    ap.add_argument("--targ_feature", default="none",
                    choices=["T", "O", "none"],
                    help="Feature to run for targ alignment (targets onset); use 'none' to skip (default: none)")
    ap.add_argument("--train_axes", action="store_true",
                    help="Train axes before running flow (writes axes into the requested axes_tag).")
    ap.add_argument("--train_features_stim", nargs="+", default=None,
                    choices=["C", "R", "T", "O"],
                    help="Features to train for stim axes when --train_axes is set (default: [stim_feature])")
    ap.add_argument("--train_features_sacc", nargs="+", default=None,
                    choices=["C", "S", "T", "O"],
                    help="Features to train for sacc axes when --train_axes is set (default: [sacc_feature] if not 'none')")
    ap.add_argument("--train_features_targ", nargs="+", default=None,
                    choices=["T", "O"],
                    help="Features to train for targ axes when --train_axes is set (default: [targ_feature] if not 'none')")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default: current)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands only; do not execute")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue if a command fails")
    ap.add_argument("--save_null_samples", action="store_true",
                    help="Save per-permutation null samples (needed for old-style group p(t) DIFF).")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    # Resolve axes_orientation (defaults to orientation if not specified)
    axes_orientation = args.axes_orientation or args.orientation

    print(f"[info] SID={args.sid}")
    print(f"[info] axes_tag_base={args.axes_tag_base}")
    print(f"[info] flow_tag_base={args.flow_tag_base}")
    print(f"[info] orientation={args.orientation}")
    print(f"[info] axes_orientation={axes_orientation}")
    print(f"[info] evoked_sigma_ms={args.evoked_sigma_ms}")
    print(f"[info] stim_feature={args.stim_feature}")
    print(f"[info] sacc_feature={args.sacc_feature}")
    print(f"[info] targ_feature={args.targ_feature}")
    print(f"[info] train_axes={args.train_axes}")

    # Detect which alignments exist
    aligns: List[str] = []
    if (out_root / "stim" / args.sid / "caches").is_dir():
        if args.stim_feature != "none":
            aligns.append("stim")
    if (out_root / "sacc" / args.sid / "caches").is_dir():
        if args.sacc_feature != "none":
            aligns.append("sacc")
    if (out_root / "targ" / args.sid / "caches").is_dir():
        if args.targ_feature != "none":
            aligns.append("targ")
    if not aligns:
        print(f"[WARN] No caches found for SID={args.sid} under {out_root}/stim, /sacc, or /targ (or all features set to 'none')")
        return

    for align in aligns:
        print(f"\n[align={align}] SID={args.sid}")

        # Construct axes tag per alignment (uses axes_orientation)
        axes_tag = f"{args.axes_tag_base}-{align}-{axes_orientation}"

        # Features per align
        if align == "stim":
            feat = args.stim_feature
            lags_ms = args.lags_ms_stim
        elif align == "sacc":
            feat = args.sacc_feature
            lags_ms = args.lags_ms_sacc
        else:  # targ
            feat = args.targ_feature
            lags_ms = args.lags_ms_targ

        # Optionally train axes first (writes into the same axes_tag we will load)
        if args.train_axes:
            if align == "stim":
                feats_train = args.train_features_stim or [feat]
            elif align == "sacc":
                feats_train = args.train_features_sacc or [feat]
            else:  # targ
                feats_train = args.train_features_targ or [feat]
            # filter out accidental 'none'
            feats_train = [f for f in feats_train if f != "none"]
            train_cmd = [args.python, CLI_TRAIN,
                         "--out_root", str(out_root),
                         "--align", align,
                         "--sid", args.sid,
                         "--orientation", axes_orientation,
                         "--tag", axes_tag,
                         "--features", *feats_train]
            rc = run(train_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
            if rc and not args.continue_on_error:
                return

        # Flow tag
        flow_tag_base_align = f"{args.flow_tag_base}-{align}"
        flow_tag = f"{flow_tag_base_align}-none-trial"  # standardize_mode=none, null_method=trial_shuffle

        print(f"\n[combo] SID={args.sid}, align={align}, feat={feat}, axes_tag={axes_tag} → flow_tag={flow_tag}")

        # Run flow_session
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
                    "--tag", flow_tag_base_align,
                    "--flow_tag_base", flow_tag_base_align,
                    "--axes_tag", axes_tag,
                    "--null_method", "trial_shuffle",
                    "--standardize_mode", "none",
                    "--pt_min_ms", str(args.pt_min_ms),
                    "--no_induced",
                    "--evoked_subtract",
                    "--evoked_sigma_ms", str(args.evoked_sigma_ms)]

        if args.save_null_samples:
            flow_cmd.append("--save_null_samples")

        rc = run(flow_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error:
            return

    print(f"\n[done] SID={args.sid} evoked subtraction flow complete.")


if __name__ == "__main__":
    main()

