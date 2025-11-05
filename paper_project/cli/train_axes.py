#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, numpy as np
from glob import glob
from paperflow.axes import train_axes_for_area, save_axes

def _load_cache(out_root: str, align: str, sid: str, area: str):
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    if not os.path.exists(path): raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}; cache["meta"] = meta
    return cache

def _discover_areas(out_root: str, align: str, sid: str):
    cdir = os.path.join(out_root, align, sid, "caches")
    return sorted([os.path.basename(p)[5:-4] for p in glob(os.path.join(cdir,"area_*.npz"))])

def _parse_win(s: str) -> tuple[float,float]:
    a,b = s.split(":"); return (float(a), float(b))

def main():
    ap = argparse.ArgumentParser(description="Train sC / sR / sS axes per area for one session (disentangled).")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--features", nargs="+", default=["C","R","S"], choices=["C","R","S"])

    # windows by align (UPDATED: C_stim default 0.10:0.30)
    ap.add_argument("--winC_stim", type=str, default="0.05:0.30")
    ap.add_argument("--winR_stim", type=str, default="0.05:0.30")
    ap.add_argument("--winC_sacc", type=str, default="-0.30:-0.18")
    ap.add_argument("--winS_sacc", type=str, default="-0.10:-0.03")

    # dims
    ap.add_argument("--dimC", type=int, default=1)
    ap.add_argument("--dimR", type=int, default=2)
    ap.add_argument("--dimS", type=int, default=1)

    # invariance toggles
    ap.add_argument("--c_invariance", choices=["none","holdoutR"], default="holdoutR",
                    help="Category invariance when training C in stim-align.")
    ap.add_argument("--no_S_invariant", action="store_true", help="If set, do NOT orthogonalize sS to sC.")

    # orientation filter (applies to BOTH alignments)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical",
                    help="Filter trials by target orientation; 'pooled' uses all.")

    # NEW — R Orthogonalization toggle 
    ap.add_argument(
        "--r_orthC",
        action=argparse.BooleanOptionalAction,   # gives --r_orthC / --no-r_orthC
        default=False,                            # default = do NOT orthogonalize
        help="Orthogonalize sR to sC (default: off)."
    )
    args = ap.parse_args()
    areas = args.areas or _discover_areas(args.out_root, args.align, args.sid)
    if not areas: raise SystemExit(f"No caches under {args.out_root}/{args.align}/{args.sid}/caches")

    any_cache = _load_cache(args.out_root, args.align, args.sid, areas[0])
    time_s = any_cache["time"].astype(float)

    if args.align == "stim":
        if "S" in args.features:
            print("[info] --align stim: dropping 'S'; S is saccade-aligned only.")
        feats = [f for f in args.features if f in ("C","R")]
        winC = _parse_win(args.winC_stim) if "C" in feats else None
        winR = _parse_win(args.winR_stim) if "R" in feats else None
        winS = None
        c_role = "stim"
    else:
        if "R" in args.features:
            print("[info] --align sacc: dropping 'R'; R is stimulus-aligned only.")
        feats = [f for f in args.features if f in ("C","S")]
        winC = _parse_win(args.winC_sacc) if "C" in feats else None
        winR = None
        winS = _parse_win(args.winS_sacc) if "S" in feats else None
        c_role = "sacc"

    if not feats: raise SystemExit("No features remain after alignment filtering.")
    ori = None if args.orientation == "pooled" else args.orientation

    saved = []
    for area in areas:
        cache = _load_cache(args.out_root, args.align, args.sid, area)
        pack = train_axes_for_area(
            cache=cache,
            feature_set=feats,
            time_s=time_s,
            winC=winC, winR=winR, winS=winS,
            orientation=ori,
            C_dim=args.dimC, R_dim=args.dimR, S_dim=args.dimS,
            make_S_invariant=(not args.no_S_invariant),
            c_invariance=args.c_invariance,
            r_orthC=args.r_orthC,
        )
        out_dir = os.path.join(args.out_root, args.align, args.sid, "axes")
        axis_roles = dict(C_align=c_role if ("C" in feats) else None,
                          R_align=("stim" if ("R" in feats and args.align=="stim") else None),
                          S_align=("sacc" if ("S" in feats and args.align=="sacc") else None))
        meta_extra = dict(
            sid=args.sid, area=area, align=args.align,
            winC=winC, winR=winR, winS=winS,
            dimC=int(args.dimC), dimR=int(args.dimR), dimS=int(args.dimS),
            orientation=args.orientation, features=feats,
            axis_roles=axis_roles, c_invariance=args.c_invariance
        )
        path = save_axes(out_dir, area, pack, meta_extra)
        print(f"[{args.sid}][{area}] wrote {path}")
        saved.append(path)

    with open(os.path.join(args.out_root, args.align, args.sid, "axes_summary.json"), "w") as f:
        json.dump(dict(sid=args.sid, align=args.align, areas=areas, axes_files=saved), f, indent=2)
    print(f"[ok] session summary → {args.out_root}/{args.align}/{args.sid}/axes_summary.json")

if __name__ == "__main__":
    main()
