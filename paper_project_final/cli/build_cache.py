#!/usr/bin/env python
from __future__ import annotations
import argparse, os
from paperflow.io import list_sessions
from paperflow.binning import build_cache_for_session

def main():
    ap = argparse.ArgumentParser(description="Build binned caches for a session (stim- or saccade-aligned).")
    ap.add_argument("--root", default=os.environ.get("PAPER_DATA",""), help="RCT_02 root")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc","targ"], required=True)
    ap.add_argument("--sid", default=None, help="Session id (8 digits). If absent, use --index.")
    ap.add_argument("--index", type=int, default=None, help="Index into discovered sessions")

    # ALIGN-SPECIFIC DEFAULTS (override with --t0/--t1/--bin_ms if passed)
    ap.add_argument("--stim_t0", type=float, default=-0.30)
    ap.add_argument("--stim_t1", type=float, default=+0.80)
    ap.add_argument("--stim_bin_ms", type=float, default=5.0)
    ap.add_argument("--stim_targets_vert_only", action="store_true", default=False)

    ap.add_argument("--sacc_t0", type=float, default=-0.40)
    ap.add_argument("--sacc_t1", type=float, default=+0.20)
    ap.add_argument("--sacc_bin_ms", type=float, default=5.0)

    ap.add_argument("--targ_t0", type=float, default=-0.20)
    ap.add_argument("--targ_t1", type=float, default=+0.35)
    ap.add_argument("--targ_bin_ms", type=float, default=5.0)
    ap.add_argument("--targ_targets_vert_only", action="store_true", default=False)

    # OPTIONAL explicit overrides
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--bin_ms", type=float, default=None)

    ap.add_argument("--no_correct_filter", action="store_true", help="Keep incorrect trials too")
    args = ap.parse_args()

    if not args.root:
        raise SystemExit("Provide --root or set $PAPER_DATA")

    # resolve session
    if args.sid:
        sid = args.sid
    else:
        if args.index is None:
            raise SystemExit("Provide --sid or --index")
        sids = list_sessions(args.root)
        if not (0 <= args.index < len(sids)):
            raise SystemExit(f"--index out of range 0..{len(sids)-1}")
        sid = sids[args.index]

    # pick defaults by alignment, then override if explicit t0/t1/bin_ms given
    if args.align == "stim":
        t0 = args.stim_t0 if args.t0 is None else args.t0
        t1 = args.stim_t1 if args.t1 is None else args.t1
        bin_ms = args.stim_bin_ms if args.bin_ms is None else args.bin_ms
        vert_only = args.stim_targets_vert_only
    elif args.align == "sacc":
        t0 = args.sacc_t0 if args.t0 is None else args.t0
        t1 = args.sacc_t1 if args.t1 is None else args.t1
        bin_ms = args.sacc_bin_ms if args.bin_ms is None else args.bin_ms
        vert_only = False  # never drop orientation at cache time for sacc-align
    else:  # targ
        t0 = args.targ_t0 if args.t0 is None else args.t0
        t1 = args.targ_t1 if args.t1 is None else args.t1
        bin_ms = args.targ_bin_ms if args.bin_ms is None else args.bin_ms
        vert_only = args.targ_targets_vert_only

    saved = build_cache_for_session(
        root=args.root, sid=sid, align=args.align,
        t0=t0, t1=t1, bin_s=bin_ms/1000.0,
        out_root=args.out_root,
        correct_only=(not args.no_correct_filter),
        stim_targets_vert_only=vert_only
    )
    if not saved:
        print(f"[warn] nothing saved for sid={sid} ({args.align})")
    else:
        print(f"[ok] sid={sid} ({args.align}) wrote {len(saved)} caches (t0={t0}, t1={t1}, bin_ms={bin_ms}):")
        for p in saved: print("   ", p)

if __name__ == "__main__":
    main()
