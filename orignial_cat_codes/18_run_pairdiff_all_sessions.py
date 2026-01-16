#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the pair-difference + metrics pipeline for MANY sessions.

It dispatches to:
  18_run_pairdiff_session.py --sid <SID> --tags <TAG1> <TAG2> ...

Session selection:
  • --sid-list-file FILE   : newline-delimited SIDs (e.g., 20200402)
  • --sids S1 S2 ...       : explicit SIDs on the CLI
  • (default) auto-discover from results/session/*

Array-friendly:
  • --idx N                : process only the Nth SID from the list
  • --one-based            : interpret --idx as 1-based (for SLURM arrays)

Examples
--------
# Run all discovered sessions for two induced tags
python 18_run_pairdiff_all_sessions.py \
  --tags induced_k2_win016_p500 induced_k5_win016_p500

# Use an explicit sid list file; process 1 session (1-based index=7)
python 18_run_pairdiff_all_sessions.py \
  --sid-list-file results/sid_list.txt \
  --tags induced_k2_win016_p500 induced_k4_win016_p500 induced_k5_win016_p500 \
  --idx 7 --one-based

# Include non-induced tags too
python 18_run_pairdiff_all_sessions.py \
  --tags induced_k2_win016_p500 induced_k5_win016_p500 win160_k2_perm500_integrated
"""
from __future__ import annotations
from pathlib import Path
import argparse, sys, subprocess

def read_sid_list_file(p: Path) -> list[str]:
    sids = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sids.append(line)
    return sids

def discover_sids(session_root: Path) -> list[str]:
    sids = []
    if not session_root.exists():
        return sids
    for d in session_root.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            sids.append(d.name)
    sids.sort()
    return sids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="One or more tag folders per session "
                         "(e.g., induced_k2_win016_p500 win160_k2_perm500_integrated)")
    ap.add_argument("--session-root", default="results/session",
                    help="Base directory containing per-session outputs (default: results/session)")
    ap.add_argument("--sid-list-file", default="",
                    help="Optional newline-delimited file of SIDs (overrides auto-discovery if given)")
    ap.add_argument("--sids", nargs="*", default=[],
                    help="Optional explicit list of SIDs (overrides file & discovery if provided)")
    ap.add_argument("--idx", type=int, default=None,
                    help="If set, process only this index from the SID list (use with arrays)")
    ap.add_argument("--one-based", action="store_true",
                    help="Interpret --idx as 1-based index (for SLURM arrays)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--redo", action="store_true",
                    help="Recompute even if pair outputs already exist")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    session_driver = repo_root / "18_run_pairdiff_session.py"
    if not session_driver.exists():
        print("[FATAL] 18_run_pairdiff_session.py not found next to this script.", file=sys.stderr)
        sys.exit(2)

    session_root = Path(args.session_root)

    # Build the SID list by precedence: --sids > --sid-list-file > discovery
    if args.sids:
        sid_list = list(dict.fromkeys(args.sids))  # dedupe, preserve order
    elif args.sid_list_file:
        sid_list = read_sid_list_file(Path(args.sid_list_file))
    else:
        sid_list = discover_sids(session_root)

    if not sid_list:
        print("[FATAL] No sessions found. Provide --sids or --sid-list-file, "
              "or ensure results/session/* exists.", file=sys.stderr)
        sys.exit(2)

    # If --idx is specified, pick exactly one SID (array-friendly)
    if args.idx is not None:
        idx0 = args.idx - 1 if args.one_based else args.idx
        if idx0 < 0 or idx0 >= len(sid_list):
            print(f"[FATAL] --idx out of range: {args.idx} for N={len(sid_list)}", file=sys.stderr)
            sys.exit(2)
        sid_list = [sid_list[idx0]]

    print(f"[info] will process {len(sid_list)} session(s).")
    print(f"[info] tags: {' '.join(args.tags)}")

    # Dispatch to per-session driver
    ok = True
    for sid in sid_list:
        cmd = [sys.executable, str(session_driver), "--sid", sid, "--alpha", str(args.alpha)]
        if args.redo:
            cmd.append("--redo")
        cmd.append("--tags")
        cmd.extend(args.tags)
        print(f"[run] {' '.join(cmd)}")
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode != 0:
            print(f"[ERR] session {sid} failed with code {r.returncode}", file=sys.stderr)
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
