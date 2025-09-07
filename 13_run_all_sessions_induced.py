#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_run_all_sessions_induced.py

Array-friendly driver for the induced/lagged/cond-B-only/integrated pipeline:

  03_cache_binned.py  (vertical-only; reuse when possible)
  04_build_axes.py    (subspaces)
  12_induced_fsflow_timesliding.py  (all ordered pairs)
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

def sh(*args, check=True):
    print("[cmd]", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=check)

def ensure_worklist(root: Path, out_json: Path):
    if out_json.exists(): return
    sh(sys.executable, BASE/"02a_list_all_pairs.py", "--root", root, "--out", out_json)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True, help="Array index selecting the session")
    ap.add_argument("--root", type=Path, required=True, help="Path to RCT/ (has manifest.json)")
    # flow params
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--win", type=float, default=0.16)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--bandC_start", type=float, default=0.12)
    ap.add_argument("--bandC_end",   type=float, default=0.28)
    ap.add_argument("--bandR_start", type=float, default=0.08)
    ap.add_argument("--bandR_end",   type=float, default=0.20)
    ap.add_argument("--int_win",     type=float, default=0.16)
    ap.add_argument("--evoked_sigma_ms", type=float, default=10.0)
    # axes params
    ap.add_argument("--trainC_start", type=float, default=0.10)
    ap.add_argument("--trainC_end",   type=float, default=0.30)
    ap.add_argument("--trainR_start", type=float, default=0.05)
    ap.add_argument("--trainR_end",   type=float, default=0.20)
    ap.add_argument("--C_dim", type=int, default=1)
    ap.add_argument("--R_dim", type=int, default=2)
    # run management
    ap.add_argument("--run_tag", type=str, default="", help="Write under results/session/<sid>/<tag>/")
    ap.add_argument("--skip_if_exists", action="store_true", default=False)
    ap.add_argument("--annotate_p", action="store_true", default=False)
    # toggles
    ap.add_argument("--no_x0", action="store_true", default=True)
    ap.add_argument("--cond_b_only", action="store_true", default=True)
    ap.add_argument("--pt_cov", action="store_true", default=False)

    args = ap.parse_args()

    root = args.root
    worklist = BASE/"results/worklists/all_pairs.json"
    worklist.parent.mkdir(parents=True, exist_ok=True)
    ensure_worklist(root, worklist)

    data = json.loads(worklist.read_text()).get("work", [])
    if not data:
        print("[info] Empty worklist."); sys.exit(0)
    if args.index < 0 or args.index >= len(data):
        print(f"[info] index {args.index} out of range 0..{len(data)-1}"); sys.exit(0)

    sid = int(data[args.index]["session"])
    print(f"[info] Running sid={sid} (array idx {args.index})")

    # 03 — cache
    sh(sys.executable, BASE/"03_cache_binned.py",
       "--root", root, "--session", sid, "--bin", 0.010, "--t0", -0.25, "--t1", 0.80,
       "--targets_vert_only", "--reuse_cache")

    # 04 — subspaces
    args04 = [sys.executable, str(BASE/"04_build_axes.py"),
              "--sid", str(sid),
              "--trainC_start", str(args.trainC_start), "--trainC_end", str(args.trainC_end),
              "--trainR_start", str(args.trainR_start), "--trainR_end", str(args.trainR_end),
              "--C_dim", str(args.C_dim), "--R_dim", str(args.R_dim)]
    if args.run_tag: args04 += ["--out_tag", args.run_tag]
    if args.skip_if_exists: args04 += ["--skip_if_exists"]
    sh(*args04)

    # 12 — induced/lagged/cond-B-only (all pairs)
    args12 = [sys.executable, str(BASE/"12_induced_fsflow_timesliding.py"),
              "--sid", str(sid), "--all_pairs",
              "--win", str(args.win), "--step", str(args.step),
              "--k", str(args.k), "--ridge", str(args.ridge),
              "--perms", str(args.perms), "--n_jobs", str(args.n_jobs),
              "--bandC_start", str(args.bandC_start), "--bandC_end", str(args.bandC_end),
              "--bandR_start", str(args.bandR_start), "--bandR_end", str(args.bandR_end),
              "--int_win", str(args.int_win),
              "--evoked_sigma_ms", str(args.evoked_sigma_ms)]
    if args.run_tag: args12 += ["--out_tag", args.run_tag]
    if args.skip_if_exists: args12 += ["--skip_if_exists"]
    if args.annotate_p: args12 += ["--annotate_p"]
    if args.no_x0: args12 += ["--no_x0"]
    if args.cond_b_only: args12 += ["--cond_b_only"]
    if args.pt_cov: args12 += ["--pt_cov"]
    sh(*args12)

if __name__ == "__main__":
    main()
