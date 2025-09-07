#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 13_run_ttc_all_sessions.py
# Orchestrates trial-wise TTC computation (script 11) and per-session plots.

import argparse, os, json, glob, subprocess, sys

def load_worklist(worklist_path):
    if not os.path.exists(worklist_path):
        return None
    with open(worklist_path, "r") as f:
        data = json.load(f)
    sessions = []
    if isinstance(data, dict) and "sessions" in data:
        items = data["sessions"]
    elif isinstance(data, list):
        items = data
    else:
        items = []
    for it in items:
        sid = it.get("sid") or it.get("session") or it.get("session_id")
        areas = it.get("areas") or it.get("present_areas") or []
        if sid is not None:
            sessions.append({"sid": str(sid), "areas": areas})
    return sessions

def discover_sessions_from_axes(root_session):
    out = []
    if not os.path.isdir(root_session):
        return out
    for sid in sorted(os.listdir(root_session)):
        sess_dir = os.path.join(root_session, sid)
        if not os.path.isdir(sess_dir):
            continue
        axes = glob.glob(os.path.join(sess_dir, "axes_*.npz"))
        areas = [os.path.basename(p).split("axes_")[1].split(".npz")[0] for p in axes]
        if areas:
            out.append({"sid": sid, "areas": areas})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_session", default="results/session", help="Where axes_<AREA>.npz and TTC outputs live")
    ap.add_argument("--worklist", default="results/worklists/all_pairs.json")
    ap.add_argument("--index", type=int, help="Which session index to run (for arrays)")
    ap.add_argument("--run_all", action="store_true", help="Run all sessions sequentially")
    ap.add_argument("--target_areas", nargs="+", default=["MFEF","MLIP","MSC"], help="Areas to attempt per session")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--smoothing_bins", type=int, default=3)
    ap.add_argument("--min_consecutive", type=int, default=2)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=0.5)
    ap.add_argument("--exclude_test_direction", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip_if_exists", action="store_true")
    ap.add_argument("--python11", default="11_trialwise_ttc.py", help="Path to TTC script")
    ap.add_argument("--python_plot", default="13_plot_ttc_session.py", help="Path to plotting script")
    args = ap.parse_args()

    sessions = load_worklist(args.worklist)
    if not sessions:
        sessions = discover_sessions_from_axes(args.root_session)
    if not sessions:
        print("[err] No sessions found. Ensure worklist or axes files exist.", file=sys.stderr)
        sys.exit(2)

    if args.run_all:
        todo = sessions
    elif args.index is not None:
        if args.index < 0 or args.index >= len(sessions):
            print(f"[err] --index {args.index} out of range [0,{len(sessions)-1}]", file=sys.stderr)
            sys.exit(3)
        todo = [sessions[args.index]]
    else:
        print("[err] provide --index (array) or --run_all", file=sys.stderr); sys.exit(4)

    for item in todo:
        sid = item["sid"]
        sess_dir = os.path.join(args.root_session, sid)
        present_axes = [os.path.basename(p).split("axes_")[1].split(".npz")[0]
                        for p in glob.glob(os.path.join(sess_dir, "axes_*.npz"))]
        areas = [a for a in args.target_areas if a in present_axes]
        if len(areas) < 2:
            print(f"[warn] {sid}: need >=2 target areas; have {areas}. Skipping.")
            continue

        out_npz = os.path.join(sess_dir, f"ttc_{args.tag}.npz")
        if args.skip_if_exists and os.path.exists(out_npz):
            print(f"[info] {sid}: found {out_npz}; skipping compute.")
        else:
            cmd = [
                sys.executable, args.python11,
                "--sid", sid,
                "--areas", *areas,
                "--root", args.root_session,
                "--alpha", str(args.alpha),
                "--smoothing_bins", str(args.smoothing_bins),
                "--min_consecutive", str(args.min_consecutive),
                "--tmin", str(args.tmin),
                "--tmax", str(args.tmax),
                "--tag", args.tag,
            ]
            if args.exclude_test_direction:
                cmd.append("--exclude_test_direction")
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=True)

        cmd_plot = [sys.executable, args.python_plot, "--sid", sid, "--root_session", args.root_session, "--tag", args.tag]
        print("[plot]", " ".join(cmd_plot))
        subprocess.run(cmd_plot, check=True)

if __name__ == "__main__":
    main()
