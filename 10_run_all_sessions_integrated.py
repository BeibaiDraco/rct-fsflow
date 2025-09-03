#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_run_all_sessions_integrated.py
Run the pipeline for ONE session by array index:
  03 -> 04 -> 08i (all pairs, with integrated GC test)
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

def sh(*args, check=True):
    print("[cmd]", " ".join(map(str,args)), flush=True)
    return subprocess.run(list(map(str,args)), check=check)

def ensure_worklist(root: Path, out_json: Path):
    if out_json.exists(): return
    sh(sys.executable, BASE/"02a_list_all_pairs.py", "--root", root, "--out", out_json)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--root", type=Path, required=True)
    # flow params
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--win", type=float, default=0.16)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--band_start", type=float, default=0.12)
    ap.add_argument("--band_end",   type=float, default=0.28)
    # axes params
    ap.add_argument("--trainC_start", type=float, default=0.10)
    ap.add_argument("--trainC_end",   type=float, default=0.30)
    ap.add_argument("--trainR_start", type=float, default=0.05)
    ap.add_argument("--trainR_end",   type=float, default=0.20)
    ap.add_argument("--C_dim", type=int, default=1)
    ap.add_argument("--R_dim", type=int, default=2)
    args = ap.parse_args()

    root = args.root
    worklist = BASE/"results/worklists/all_pairs.json"
    worklist.parent.mkdir(parents=True, exist_ok=True)
    ensure_worklist(root, worklist)
    data = json.loads(worklist.read_text()).get("work", [])
    if not data:
        print("[info] Empty worklist."); sys.exit(0)
    if args.index<0 or args.index>=len(data):
        print(f"[info] index {args.index} out of range 0..{len(data)-1}"); sys.exit(0)

    sid = int(data[args.index]["session"])
    print(f"[info] Running sid={sid} (array idx {args.index})")

    # 03: cache (vertical-only)
    sh(sys.executable, BASE/"03_cache_binned.py",
       "--root", root, "--session", sid, "--bin", 0.010, "--t0", -0.25, "--t1", 0.80, "--targets_vert_only")
    # 04: subspaces (per-category center/whiten for R)
    sh(sys.executable, BASE/"04_build_axes.py",
       "--sid", sid,
       "--trainC_start", args.trainC_start, "--trainC_end", args.trainC_end,
       "--trainR_start", args.trainR_start, "--trainR_end", args.trainR_end,
       "--C_dim", args.C_dim, "--R_dim", args.R_dim)
    # 08i: time-sliding + integrated band test (all pairs)
    sh(sys.executable, BASE/"08i_fsflow_timesliding_integrated.py",
       "--sid", sid, "--all_pairs",
       "--win", args.win, "--step", args.step, "--k", args.k, "--ridge", args.ridge,
       "--perms", args.perms, "--n_jobs", args.n_jobs,
       "--band_start", args.band_start, "--band_end", args.band_end)

if __name__ == "__main__":
    main()
