#!/usr/bin/env python
import argparse, os, subprocess, json
from saccflow.io import list_sessions

def run(cmd):
    print("[RUN]", " ".join(cmd)); subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Run saccade pipeline for ONE session index (for SLURM arrays).")
    ap.add_argument("--root", required=True, help="RCT_02 root")
    ap.add_argument("--index", type=int, required=True, help="0-based index into discovered sessions")
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical")
    ap.add_argument("--t0", type=float, default=-0.40)
    ap.add_argument("--t1", type=float, default=0.20)
    ap.add_argument("--bin_ms", type=float, default=5.0)
    ap.add_argument("--trainC", type=str, default="-0.30:-0.18")
    ap.add_argument("--trainS", type=str, default="-0.10:-0.03")
    ap.add_argument("--lags_ms", type=float, default=30.0)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--tag", default="sacc_v1")
    ap.add_argument("--out_root", default="results_sacc")
    args = ap.parse_args()

    sessions = list_sessions(args.root)
    sid = sessions[args.index]
    print(f"[info] session={sid} ({args.index}/{len(sessions)-1}), orient={args.orientation}")

    # 31) trials enrich (idempotent)
    run(["python", "scripts/31_trials_enrich.py", "--root", args.root, "--sid", sid, "--out_root", args.out_root])

    # 32) cache (sacc-aligned)
    run([
      "python","scripts/32_cache_binned_sacc.py","--root",args.root,"--sid",sid,
      "--t0",str(args.t0),"--t1",str(args.t1),"--bin_ms",str(args.bin_ms),
      "--out_root",args.out_root
    ])

    # 33) axes training
    run([
      "python","scripts/33_build_axes_sacc.py","--root",args.root,"--sid",sid,
      "--orientation",args.orientation,"--trainC",args.trainC,"--trainS",args.trainS,
      "--out_root",args.out_root
    ])

    # 34) QC (AUC curves & latencies)
    run([
      "python","scripts/34_axes_qc_sacc.py","--sid",sid,"--orientation",args.orientation,
      "--out_root",args.out_root
    ])

    # 35) flows (all ordered pairs)
    run([
      "python","scripts/35_fsflow_sacc_timesliding.py","--sid",sid,"--orientation",args.orientation,
      "--all_pairs","--lags_ms",str(args.lags_ms),"--perms",str(args.perms),
      "--tag",args.tag,"--out_root",args.out_root
    ])

    # 36) overlays (with null means)
    run([
      "python","scripts/36_session_overlays_sacc.py","--sid",sid,"--tag",args.tag,
      "--orientation",args.orientation,"--out_root",args.out_root,"--shade_null"
    ])

    # 37) pair-diff (paired null), all ordered pairs
    run([
      "python","scripts/37_pairdiff_sacc_session.py","--sid",sid,"--orientation",args.orientation,
      "--all_pairs","--lags_ms",str(args.lags_ms),"--perms",str(args.perms),
      "--tag",args.tag,"--out_root",args.out_root
    ])

if __name__ == "__main__":
    main()
