#!/usr/bin/env python
from __future__ import annotations
import argparse, os
from paperflow.io import list_sessions
from paperflow.binning import build_cache_for_session

def main():
    ap = argparse.ArgumentParser(description="Build binned caches for a session (stim- or saccade-aligned).")
    ap.add_argument("--root", default=os.environ.get("PAPER_DATA",""), help="RCT_02 root")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc"], required=True)
    ap.add_argument("--sid", default=None, help="Session id (8 digits). If absent, use --index.")
    ap.add_argument("--index", type=int, default=None, help="Index into discovered sessions")
    ap.add_argument("--t0", type=float, default=-0.40)
    ap.add_argument("--t1", type=float, default=0.20)
    ap.add_argument("--bin_ms", type=float, default=5.0)
    ap.add_argument("--no_correct_filter", action="store_true", help="Keep incorrect trials too")
    args = ap.parse_args()
    if not args.root: raise SystemExit("Provide --root or set $PAPER_DATA")

    if args.sid:
        sid = args.sid
    else:
        if args.index is None:
            raise SystemExit("Provide --sid or --index")
        sids = list_sessions(args.root)
        if not (0 <= args.index < len(sids)):
            raise SystemExit(f"--index out of range 0..{len(sids)-1}")
        sid = sids[args.index]

    saved = build_cache_for_session(
        root=args.root, sid=sid, align=args.align,
        t0=args.t0, t1=args.t1, bin_s=args.bin_ms/1000.0,
        out_root=args.out_root, correct_only=(not args.no_correct_filter)
    )
    if not saved:
        print(f"[warn] nothing saved for sid={sid} ({args.align})")
    else:
        print(f"[ok] sid={sid} ({args.align}) wrote {len(saved)} caches:")
        for p in saved: print("   ", p)

if __name__ == "__main__":
    main()
