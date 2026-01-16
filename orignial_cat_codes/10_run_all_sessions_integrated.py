#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_run_all_sessions_integrated.py
03 -> 04 -> 08i (all pairs, with integrated GC test + sliding-integrated series)

Example:
  python 10_run_all_sessions_integrated.py \
      --index ${SLURM_ARRAY_TASK_ID} \
      --root RCT \
      --perms 500 --n_jobs 8 --win 0.16 --step 0.02 --k 5 --ridge 1e-2 \
      --band_start 0.12 --band_end 0.28 --int_win 0.16 \
      --run_tag win160_k5_p500 --skip_if_exists \
      --annotate_p_int --p_text_stride 5
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
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--win", type=float, default=0.16)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--band_start", type=float, default=0.12)
    ap.add_argument("--band_end",   type=float, default=0.28)
    ap.add_argument("--int_win",    type=float, default=None)  # default handled in 08i
    ap.add_argument("--trainC_start", type=float, default=0.10)
    ap.add_argument("--trainC_end",   type=float, default=0.30)
    ap.add_argument("--trainR_start", type=float, default=0.05)
    ap.add_argument("--trainR_end",   type=float, default=0.20)
    ap.add_argument("--C_dim", type=int, default=1)
    ap.add_argument("--R_dim", type=int, default=2)
    ap.add_argument("--run_tag", type=str, default="")
    ap.add_argument("--skip_if_exists", action="store_true", default=False)

    # New pass-throughs for plotting p-values and band span
    ap.add_argument("--annotate_p_raw", action="store_true", default=False)
    ap.add_argument("--annotate_p_int", action="store_true", default=False)
    ap.add_argument("--p_text_stride", type=int, default=4)
    ap.add_argument("--show_band_span", action="store_true", default=False)

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

    # 03 (vertical-only cache; reuse if meta matches)
    sh(sys.executable, BASE/"03_cache_binned.py",
       "--root", root, "--session", sid, "--bin", 0.010, "--t0", -0.25, "--t1", 0.80,
       "--targets_vert_only", "--reuse_cache")

    # 04 (subspaces; tag + skip support)
    args04 = [sys.executable, str(BASE/"04_build_axes.py"),
              "--sid", str(sid),
              "--trainC_start", str(args.trainC_start), "--trainC_end", str(args.trainC_end),
              "--trainR_start", str(args.trainR_start), "--trainR_end", str(args.trainR_end),
              "--C_dim", str(args.C_dim), "--R_dim", str(args.R_dim)]
    if args.run_tag: args04 += ["--out_tag", args.run_tag]
    if args.skip_if_exists: args04 += ["--skip_if_exists"]
    sh(*args04)

    # 08i (time-sliding + sliding-integrated + per-time p)
    args08 = [sys.executable, str(BASE/"08i_fsflow_timesliding_integrated.py"),
              "--sid", str(sid), "--all_pairs",
              "--win", str(args.win), "--step", str(args.step), "--k", str(args.k), "--ridge", str(args.ridge),
              "--perms", str(args.perms), "--n_jobs", str(args.n_jobs),
              "--band_start", str(args.band_start), "--band_end", str(args.band_end)]
    if args.int_win is not None:
        args08 += ["--int_win", str(args.int_win)]
    if args.run_tag: args08 += ["--out_tag", args.run_tag]
    if args.skip_if_exists: args08 += ["--skip_if_exists"]

    # pass-through new flags
    if args.annotate_p_raw: args08 += ["--annotate_p_raw"]
    if args.annotate_p_int: args08 += ["--annotate_p_int"]
    args08 += ["--p_text_stride", str(args.p_text_stride)]
    if args.show_band_span: args08 += ["--show_band_span"]

    sh(*args08)

if __name__ == "__main__":
    main()
