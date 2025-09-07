#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L02_run_all_trial_latency.py

Driver for per-trial decoding-latency analysis across sessions:
  03_cache_binned.py  (vertical-only; reuse)
  04_build_axes.py    (subspaces; use your axes_tag if needed)
  L01_trial_latency.py (per-trial latencies + pairwise lead/lag plots)

Example (array task):
  python L02_run_all_trial_latency.py --index ${SLURM_ARRAY_TASK_ID} --root RCT \
    --axes_tag win160_k5_perm500 --out_tag latency \
    --auto_thr --auto_strategy session --auto_targetC 0.60 --auto_targetR 0.60 \
    --hold_m 3 --tmin 0.0 --smooth_bins 3
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

def sh(*args, check=True):
    print("[cmd]", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str,args)), check=check)

def ensure_worklist(root: Path, out_json: Path):
    if out_json.exists(): return
    sh(sys.executable, BASE/"02a_list_all_pairs.py", "--root", root, "--out", out_json)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--root", type=Path, required=True)
    # 03 params (fixed in your pipeline)
    ap.add_argument("--bin", type=float, default=0.010)
    ap.add_argument("--t0", type=float, default=-0.25)
    ap.add_argument("--t1", type=float, default=0.80)
    # 04 params
    ap.add_argument("--trainC_start", type=float, default=0.10)
    ap.add_argument("--trainC_end",   type=float, default=0.30)
    ap.add_argument("--trainR_start", type=float, default=0.05)
    ap.add_argument("--trainR_end",   type=float, default=0.20)
    ap.add_argument("--C_dim", type=int, default=1)
    ap.add_argument("--R_dim", type=int, default=2)
    ap.add_argument("--axes_tag", type=str, default="", help="Write axes under results/session/<sid>/<axes_tag>/")
    ap.add_argument("--out_tag",  type=str, default="latency", help="Write latency outputs under results/session/<sid>/<out_tag>/")
    ap.add_argument("--skip_if_exists", action="store_true", default=False)
    # Latency params (also allowing AUTO)
    ap.add_argument("--thrC", type=float, default=0.75)
    ap.add_argument("--thrR", type=float, default=0.60)
    ap.add_argument("--hold_m", type=int, default=3)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--smooth_bins", type=int, default=3)
    ap.add_argument("--auto_thr", action="store_true", help="Auto-pick thresholds to hit target fraction")
    ap.add_argument("--auto_strategy", type=str, choices=["session","area"], default="session")
    ap.add_argument("--auto_targetC", type=float, default=0.60)
    ap.add_argument("--auto_targetR", type=float, default=0.60)
    ap.add_argument("--auto_grid_C", type=str, default="0.55:0.95:0.01")
    ap.add_argument("--auto_grid_R", type=str, default="0.40:0.95:0.01")
    args = ap.parse_args()

    # Worklist
    worklist = BASE/"results/worklists/all_pairs.json"
    worklist.parent.mkdir(parents=True, exist_ok=True)
    ensure_worklist(args.root, worklist)
    data = json.loads(worklist.read_text()).get("work", [])
    if not data:
        print("[info] Empty worklist."); sys.exit(0)
    if args.index < 0 or args.index >= len(data):
        print(f"[info] index {args.index} out of range 0..{len(data)-1}"); sys.exit(0)
    sid = int(data[args.index]["session"])
    print(f"[info] Running sid={sid} (array idx {args.index})")

    # 03: caches (vertical-only)
    sh(sys.executable, BASE/"03_cache_binned.py",
       "--root", args.root, "--session", sid,
       "--bin", args.bin, "--t0", args.t0, "--t1", args.t1,
       "--targets_vert_only", "--reuse_cache")

    # 04: axes (if tag requested, write under it)
    args04 = [sys.executable, str(BASE/"04_build_axes.py"),
              "--sid", str(sid),
              "--trainC_start", str(args.trainC_start), "--trainC_end", str(args.trainC_end),
              "--trainR_start", str(args.trainR_start), "--trainR_end", str(args.trainR_end),
              "--C_dim", str(args.C_dim), "--R_dim", str(args.R_dim)]
    if args.axes_tag: args04 += ["--out_tag", args.axes_tag]
    if args.skip_if_exists: args04 += ["--skip_if_exists"]
    sh(*args04)

    # L01: per-trial latencies
    cmd = [sys.executable, str(BASE/"L01_trial_latency.py"),
           "--sid", str(sid),
           "--axes_tag", args.axes_tag,
           "--out_tag", args.out_tag,
           "--hold_m", str(args.hold_m), "--tmin", str(args.tmin),
           "--smooth_bins", str(args.smooth_bins)]
    if args.auto_thr:
        cmd += ["--auto_thr", "--auto_strategy", args.auto_strategy,
                "--auto_targetC", str(args.auto_targetC),
                "--auto_targetR", str(args.auto_targetR),
                "--auto_grid_C", args.auto_grid_C,
                "--auto_grid_R", args.auto_grid_R]
    else:
        cmd += ["--thrC", str(args.thrC), "--thrR", str(args.thrR)]
    sh(*cmd)

if __name__ == "__main__":
    main()
