#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from typing import List
from paperflow.io import list_sessions, list_areas
from paperflow.binning import build_cache_for_session

def main():
    ap = argparse.ArgumentParser(description="Build binned caches for MANY sessions (stim or sacc).")
    ap.add_argument("--root", default=os.environ.get("PAPER_DATA",""),
                    help="RCT_02 root (has manifest.json and <sid>/...)")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"),
                    help="Output root (default: $PAPER_HOME/out)")
    ap.add_argument("--align", choices=["stim","sacc"], default="stim",
                    help="Which alignment to build caches for (default: stim)")
    ap.add_argument("--min_areas", type=int, default=2,
                    help="Only build for sessions having at least this many valid areas (default: 2)")
    # align-specific defaults (overridden below if --t0/--t1/--bin_ms passed)
    ap.add_argument("--stim_t0", type=float, default=-0.25)
    ap.add_argument("--stim_t1", type=float, default=+0.80)
    ap.add_argument("--stim_bin_ms", type=float, default=10.0)
    ap.add_argument("--stim_targets_vert_only", action="store_true", default=False,
                    help="For stim-align, keep only vertical trials (default: keep both orientations).")
    ap.add_argument("--sacc_t0", type=float, default=-0.40)
    ap.add_argument("--sacc_t1", type=float, default=+0.30)
    ap.add_argument("--sacc_bin_ms", type=float, default=5.0)
    # optional explicit overrides
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--bin_ms", type=float, default=None)
    # selection
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="Optional explicit list of session IDs (8-digit). If omitted, scans all under --root.")
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("Provide --root or set $PAPER_DATA")

    # resolve session list
    sids: List[str] = args.sessions if args.sessions else list_sessions(args.root)
    if not sids:
        raise SystemExit(f"No sessions found under {args.root}")

    # resolve align-specific defaults, then apply explicit overrides (if any)
    if args.align == "stim":
        t0 = args.stim_t0 if args.t0 is None else args.t0
        t1 = args.stim_t1 if args.t1 is None else args.t1
        bin_ms = args.stim_bin_ms if args.bin_ms is None else args.bin_ms
        vert_only = args.stim_targets_vert_only
    else:
        t0 = args.sacc_t0 if args.t0 is None else args.t0
        t1 = args.sacc_t1 if args.t1 is None else args.t1
        bin_ms = args.sacc_bin_ms if args.bin_ms is None else args.bin_ms
        vert_only = False  # never drop orientations at cache time for sacc-align

    print(f"[info] building caches: align={args.align}  sessions={len(sids)}  "
          f"t0={t0} t1={t1} bin_ms={bin_ms}  min_areas={args.min_areas}  "
          f"stim_vert_only={vert_only if args.align=='stim' else False}")

    built = 0
    skipped = 0
    for sid in sids:
        areas = list_areas(args.root, sid)
        if len(areas) < args.min_areas:
            print(f"[skip] {sid}: only {len(areas)} valid areas present ({areas})")
            skipped += 1
            continue

        saved = build_cache_for_session(
            root=args.root, sid=sid, align=args.align,
            t0=t0, t1=t1, bin_s=bin_ms/1000.0,
            out_root=args.out_root,
            correct_only=True,
            stim_targets_vert_only=vert_only
        )
        if not saved:
            print(f"[warn] {sid}: nothing saved (filters may have removed all trials)")
        else:
            print(f"[ok] {sid}: wrote {len(saved)} caches")
            built += 1

    print(f"[done] align={args.align}  built_for={built}  skipped={skipped}  total={len(sids)}")

if __name__ == "__main__":
    main()
