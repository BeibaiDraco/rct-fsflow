#!/usr/bin/env python
import argparse, os, json
from glob import glob
from saccflow.flow import compute_sflow_pairdiff_for_pair, save_pairdiff_npz

def _areas_from_caches(out_root, sid):
    cdir = os.path.join(out_root, sid, "caches")
    paths = sorted(glob(os.path.join(cdir, "area_*.npz")))
    # 'area_MFEF.npz' -> 'MFEF'
    return [os.path.basename(p)[5:-4] for p in paths]

def _all_pairs(areas):
    return [(a,b) for a in areas for b in areas if a!=b]

def main():
    ap = argparse.ArgumentParser(description="Per-session paired-null pair-diff for S-flow (time-series only).")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical")
    ap.add_argument("--out_root", default="results_sacc")
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
    args = ap.parse_args()

    areas = args.areas or _areas_from_caches(args.out_root, args.sid)
    pairs = _all_pairs(areas) if args.all_pairs else [tuple(args.pair)]

    out_dir = os.path.join(args.out_root, args.sid, "saccflow", args.tag, "pairdiff")
    os.makedirs(out_dir, exist_ok=True)

    for (A,B) in pairs:
        pdres = compute_sflow_pairdiff_for_pair(
            out_root=args.out_root, sid=args.sid, areaA=A, areaB=B,
            orientation=args.orientation, lags_ms=args.lags_ms, permutations=args.perms,
            alpha=args.alpha, use_induced=(not args.no_induced),
            condition_on_C=(not args.no_condC), include_PT_cov=args.pt_cov
        )
        out_npz = os.path.join(out_dir, f"pairdiff_S_{A}to{B}.npz")
        save_pairdiff_npz(pdres, out_npz)
        print(f"[{args.sid}][{A}->{B}] wrote {out_npz}")

if __name__ == "__main__":
    main()
