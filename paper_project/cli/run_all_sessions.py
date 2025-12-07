#!/usr/bin/env python3
"""
Run axes training, QC, and flow for all sessions with caches.

Pipeline per session (for each align):
  1) train_axes.py
  2) qc_axes.py
  3) flow_session.py
     - align=stim: features C and R
     - align=sacc: feature S

Notes:
- Uses underscored CLI flags (e.g., --out_root), matching your tools.
- Relies on flow.py defaults for perm_within="CR", induced=True, include_B_lags=True.
"""

import argparse
import subprocess
import sys
from pathlib import Path

CLI_TRAIN = "cli/train_axes.py"
CLI_QC    = "cli/qc_axes.py"
CLI_FLOW  = "cli/flow_session.py"

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

def find_sessions(out_root: Path, align: str):
    base = out_root / align
    if not base.exists():
        return []
    sids = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "caches").is_dir():
            sids.append(p.name)
    return sids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out", help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--lags_ms_stim", type=float, default=50.0, help="Lags (ms) for stim-aligned flows")
    ap.add_argument("--lags_ms_sacc", type=float, default=30.0, help="Lags (ms) for sacc-aligned flows")
    ap.add_argument("--ridge", type=float, default=1e-2, help="Ridge penalty")
    ap.add_argument("--perms", type=int, default=500, help="Number of permutations")
    ap.add_argument("--orientation", default="vertical",
                    choices=["vertical","horizontal","pooled"],
                    help="Orientation for flow_session (default: vertical)")
    ap.add_argument("--tag", default="crnull-vertical", help="Flow tag subfolder (default: crnull-vertical)")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only; do not execute")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue remaining jobs if a command fails")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    sids_stim = find_sessions(out_root, "stim")
    sids_sacc = find_sessions(out_root, "sacc")
    print(f"[info] Found {len(sids_stim)} stim sessions and {len(sids_sacc)} sacc sessions under {out_root}")

    # ---------- STIM alignment ----------
    for sid in sids_stim:
        # 1) train_axes
        rc = run([args.python, CLI_TRAIN,
                  "--out_root", str(out_root),
                  "--align", "stim",
                  "--sid", sid,
                  "--orientation", args.orientation,
                  "--tag", args.tag],
                 dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error: return

        # 2) qc_axes
        rc = run([args.python, CLI_QC,
                  "--out_root", str(out_root),
                  "--align", "stim",
                  "--sid", sid,
                  "--orientation", args.orientation,
                  "--tag", args.tag],
                 dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error: return

        # 3) flow for C and R
        for feat in ("C", "R"):
            rc = run([args.python, CLI_FLOW,
                      "--out_root", str(out_root),
                      "--align", "stim",
                      "--sid", sid,
                      "--feature", feat,
                      "--lags_ms", str(args.lags_ms_stim),
                      "--ridge", str(args.ridge),
                      "--perms", str(args.perms),
                      "--orientation", args.orientation,
                      "--tag", args.tag,
                      "--axes_tag", args.tag],
                     dry=args.dry_run, continue_on_error=args.continue_on_error)
            if rc and not args.continue_on_error: return

    # ---------- SACC alignment ----------
    for sid in sids_sacc:
        # 1) train_axes
        rc = run([args.python, CLI_TRAIN,
                  "--out_root", str(out_root),
                  "--align", "sacc",
                  "--sid", sid,
                  "--orientation", args.orientation,
                  "--tag", args.tag],
                 dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error: return

        # 2) qc_axes
        rc = run([args.python, CLI_QC,
                  "--out_root", str(out_root),
                  "--align", "sacc",
                  "--sid", sid,
                  "--orientation", args.orientation,
                  "--tag", args.tag],
                 dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error: return

        # 3) flow for S (saccade)
        rc = run([args.python, CLI_FLOW,
                  "--out_root", str(out_root),
                  "--align", "sacc",
                  "--sid", sid,
                  "--feature", "S",
                  "--lags_ms", str(args.lags_ms_sacc),
                  "--ridge", str(args.ridge),
                  "--perms", str(args.perms),
                  "--orientation", args.orientation,
                  "--tag", args.tag,
                  "--axes_tag", args.tag],
                 dry=args.dry_run, continue_on_error=args.continue_on_error)
        if rc and not args.continue_on_error: return

    print("[done] All runs completed.")

if __name__ == "__main__":
    main()
