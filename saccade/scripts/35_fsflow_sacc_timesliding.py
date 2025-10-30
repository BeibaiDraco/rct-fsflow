#!/usr/bin/env python
import argparse, os, json
from glob import glob
from saccflow.flow import compute_saccade_flow_for_pair, save_flow_npz

def _discover_areas(out_root: str, sid: str):
    cdir = os.path.join(out_root, sid, "caches")
    return sorted([os.path.basename(p)[5:-4] for p in glob(os.path.join(cdir, "area_*.npz"))])

def _ordered_pairs(areas):
    return [(a,b) for a in areas for b in areas if a!=b]

def main():
    ap = argparse.ArgumentParser(description="Saccade FS flow (time-sliding only, induced, category-conditioned).")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical")
    ap.add_argument("--out_root", default="results_sacc")
    ap.add_argument("--axes_dir", default=None)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--all_pairs", action="store_true")
    ap.add_argument("--pair", nargs=2, metavar=("AREA_A","AREA_B"))
    ap.add_argument("--lags_ms", type=float, default=30.0)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no_induced", action="store_true")
    ap.add_argument("--no_condC", action="store_true")
    ap.add_argument("--pt_cov", action="store_true")
    ap.add_argument("--tag", default="sacc_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    areas = args.areas or _discover_areas(args.out_root, args.sid)
    if not areas:
        raise RuntimeError(f"No caches found in {args.out_root}/{args.sid}/caches")

    pairs = _ordered_pairs(areas) if args.all_pairs else [tuple(args.pair)]
    out_dir = os.path.join(args.out_root, args.sid, "saccflow", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    for (A, B) in pairs:
        fr = compute_saccade_flow_for_pair(
            out_root=args.out_root, sid=args.sid, areaA=A, areaB=B,
            axes_dir=args.axes_dir, orientation=args.orientation,
            lags_ms=args.lags_ms, permutations=args.perms, alpha=args.alpha,
            use_induced=(not args.no_induced), condition_on_C=(not args.no_condC),
            include_PT_cov=args.pt_cov, rng_seed=args.seed
        )
        out_npz = os.path.join(out_dir, f"induced_flow_S_{A}to{B}.npz" if not args.no_induced else f"raw_flow_S_{A}to{B}.npz")
        save_flow_npz(fr, out_npz)
        print(f"[{args.sid}][{A}->{B}] wrote {out_npz}")

    with open(os.path.join(out_dir, "runmeta.json"), "w") as f:
        json.dump(dict(
            sid=args.sid, orientation=args.orientation, areas=areas, pairs=pairs,
            lags_ms=args.lags_ms, perms=args.perms, alpha=args.alpha,
            induced=(not args.no_induced), condC=(not args.no_condC), pt_cov=args.pt_cov,
            tag=args.tag
        ), f, indent=2)

if __name__ == "__main__":
    main()
