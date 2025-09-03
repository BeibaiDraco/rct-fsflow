#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_run_all_sessions.py
Run the full vertical-only pipeline for ONE session (by array index):
  03_cache_binned.py  -> caches (-0.25..0.80s, 10ms), auto-detect M/S areas
  04_build_axes.py    -> subspaces (pooled windows, per-category center/whiten for R)
  08p_fsflow_timesliding_parallel.py --all_pairs -> GC time series for every ordered pair

Usage (from Slurm array):
  python 10_run_all_sessions.py --index ${SLURM_ARRAY_TASK_ID} \
      --root RCT --perms 200 --n_jobs 8 --win 0.12 --step 0.02 --k 4

Notes:
- If results/worklists/all_pairs.json is missing, we build it via 02a_list_all_pairs.py.
- We do NOT monopolize cores: permutations use --n_jobs, set your --cpus-per-task accordingly.
"""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

def sh(*args, check=True):
    print("[cmd]", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=check)

def ensure_worklist(root: Path, out_json: Path):
    if out_json.exists():
        return
    # build all-pairs worklist
    sh(sys.executable, BASE/"02a_list_all_pairs.py", "--root", root, "--out", out_json)

def load_worklist(out_json: Path):
    data = json.loads(out_json.read_text())
    work = data.get("work", [])
    return work

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True, help="Array index for session selection")
    ap.add_argument("--root", type=Path, required=True, help="Path to RCT/ (has manifest.json)")
    # flow params
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--win", type=float, default=0.12)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=4)
    # axes params (you can tweak if desired)
    ap.add_argument("--trainC_start", type=float, default=0.10)
    ap.add_argument("--trainC_end",   type=float, default=0.30)
    ap.add_argument("--trainR_start", type=float, default=0.05)
    ap.add_argument("--trainR_end",   type=float, default=0.20)
    ap.add_argument("--C_dim", type=int, default=1)
    ap.add_argument("--R_dim", type=int, default=2)
    ap.add_argument("--ridge", type=float, default=1e-2)
    args = ap.parse_args()

    root = args.root
    worklist_path = BASE/"results/worklists/all_pairs.json"
    worklist_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_worklist(root, worklist_path)

    work = load_worklist(worklist_path)
    if not work:
        print("[info] No eligible sessions in worklist.", file=sys.stderr)
        sys.exit(0)

    # Bound the index to avoid crashes if array is larger than worklist
    if args.index < 0 or args.index >= len(work):
        print(f"[info] Array index {args.index} out of range (0..{len(work)-1}). Exiting politely.")
        sys.exit(0)

    sess = work[args.index]
    sid = int(sess["session"])
    print(f"[info] Running session {sid}  (array index {args.index} of {len(work)})", flush=True)

    # === 03: cache vertical-only trials for all present areas in this session ===
    sh(sys.executable, BASE/"03_cache_binned.py",
       "--root", root, "--session", sid,
       "--bin", 0.010, "--t0", -0.25, "--t1", 0.80,
       "--targets_vert_only")

    # === 04: build subspaces with per-category centering/whitening ===
    sh(sys.executable, BASE/"04_build_axes.py",
       "--sid", sid,
       "--trainC_start", args.trainC_start, "--trainC_end", args.trainC_end,
       "--trainR_start", args.trainR_start, "--trainR_end", args.trainR_end,
       "--C_dim", args.C_dim, "--R_dim", args.R_dim)

    # === 08p: run time-sliding GC for ALL ordered pairs in this session ===
    sh(sys.executable, BASE/"08p_fsflow_timesliding_parallel.py",
       "--sid", sid, "--all_pairs",
       "--win", args.win, "--step", args.step,
       "--k", args.k, "--ridge", args.ridge,
       "--perms", args.perms, "--n_jobs", args.n_jobs)

    print(f"[done] Session {sid} completed.", flush=True)

if __name__ == "__main__":
    main()