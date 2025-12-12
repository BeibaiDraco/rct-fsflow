#!/usr/bin/env python3
"""
Run old-style evoked subtraction flow analysis for ONE session.

For each alignment (stim, sacc):
  - stim: feature C
  - sacc: feature S

Uses pre-existing axes (axes_sweep-<align>-<axes_orientation>).
Only runs flow_session.py (assumes axes already exist).

Supports:
  --orientation: flow orientation (vertical, horizontal, pooled)
  --axes_orientation: which axes to load (defaults to --orientation)

Example: vertical axes + pooled flow:
  --axes_orientation vertical --orientation pooled
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

CLI_FLOW = "cli/flow_session.py"


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
        description="Run evoked subtraction flow for one session (stim: C, sacc: S)."
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
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default: current)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands only; do not execute")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue if a command fails")
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

    # Detect which alignments exist
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

        # Construct axes tag per alignment (uses axes_orientation)
        axes_tag = f"{args.axes_tag_base}-{align}-{axes_orientation}"

        # Features per align
        if align == "stim":
            feat = "C"
            lags_ms = args.lags_ms_stim
        else:  # sacc
            feat = "S"
            lags_ms = args.lags_ms_sacc

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

        rc = run(flow_cmd, dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error:
            return

    print(f"\n[done] SID={args.sid} evoked subtraction flow complete.")


if __name__ == "__main__":
    main()

