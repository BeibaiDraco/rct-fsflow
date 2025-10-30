#!/usr/bin/env python
import argparse, os, subprocess
from saccflow.io import list_sessions

VALID_AREAS = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def run(cmd):
    print("[RUN]", " ".join(cmd)); subprocess.run(cmd, check=True)

def area_count(root: str, sid: str) -> int:
    aroot = os.path.join(root, sid, "areas")
    if not os.path.isdir(aroot): return 0
    cnt = 0
    for name in os.listdir(aroot):
        p = os.path.join(aroot, name)
        if os.path.isdir(p) and name in VALID_AREAS:
            cnt += 1
    return cnt

def main():
    ap = argparse.ArgumentParser(description="Run saccade pipeline for ONE session.")
    ap.add_argument("--root", required=True, help="RCT_02 root")
    ap.add_argument("--sid", default=None, help="Session ID (8-digit). If absent, use --index.")
    ap.add_argument("--index", type=int, default=None, help="0-based index into discovered sessions")
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

    # Resolve session
    if args.sid:
        sid = args.sid
    else:
        if args.index is None:
            raise SystemExit("Provide --sid or --index")
        sessions = list_sessions(args.root)
        if args.index < 0 or args.index >= len(sessions):
            raise SystemExit(f"--index out of range 0..{len(sessions)-1}")
        sid = sessions[args.index]

    # Skip thin sessions (<2 areas)
    n_areas = area_count(args.root, sid)
    if n_areas < 2:
        print(f"[skip] {sid}: only {n_areas} area(s) present under {args.root}/{sid}/areas")
        return

    print(f"[info] session={sid}  orient={args.orientation}  areas>={n_areas}")

    # 31) trials enrich
    run([
        "python","scripts/31_trials_enrich.py",
        "--root", args.root,
        "--sid",  sid,
        f"--out_root={args.out_root}"
    ])

    # 32) cache (sacc-aligned)
    run([
        "python","scripts/32_cache_binned_sacc.py",
        "--root", args.root,
        "--sid",  sid,
        f"--t0={args.t0}",
        f"--t1={args.t1}",
        f"--bin_ms={args.bin_ms}",
        f"--out_root={args.out_root}"
    ])

    # 33) axes training  (NOTE the '=' for trainC/trainS to avoid argparse confusion)
    run([
        "python","scripts/33_build_axes_sacc.py",
        "--root", args.root,
        "--sid",  sid,
        "--orientation", args.orientation,
        f"--trainC={args.trainC}",
        f"--trainS={args.trainS}",
        f"--out_root={args.out_root}"
    ])

    # 34) QC
    run([
        "python","scripts/34_axes_qc_sacc.py",
        "--sid",  sid,
        "--orientation", args.orientation,
        f"--out_root={args.out_root}"
    ])

    # 35) flows
    run([
        "python","scripts/35_fsflow_sacc_timesliding.py",
        "--sid", sid,
        "--orientation", args.orientation,
        "--all_pairs",
        f"--lags_ms={args.lags_ms}",
        f"--perms={args.perms}",
        f"--tag={args.tag}",
        f"--out_root={args.out_root}"
    ])

    # 36) overlays
    run([
        "python","scripts/36_session_overlays_sacc.py",
        "--sid", sid,
        "--tag", args.tag,
        "--orientation", args.orientation,
        f"--out_root={args.out_root}",
        "--shade_null"
    ])

    # 37) pair-diff
    run([
        "python","scripts/37_pairdiff_sacc_session.py",
        "--sid", sid,
        "--orientation", args.orientation,
        "--all_pairs",
        f"--lags_ms={args.lags_ms}",
        f"--perms={args.perms}",
        f"--tag={args.tag}",
        f"--out_root={args.out_root}"
    ])

if __name__ == "__main__":
    main()
