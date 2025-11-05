#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, numpy as np
from glob import glob
from paperflow.qc import qc_curves_for_area, save_qc, save_qc_json

def _load_cache(out_root: str, align: str, sid: str, area: str):
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    if not os.path.exists(path): raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}; cache["meta"] = meta
    return cache

def _load_axes(out_root: str, align: str, sid: str, area: str):
    path = os.path.join(out_root, align, sid, "axes", f"axes_{area}.npz")
    if not os.path.exists(path): raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    axes = {k: d[k] for k in d.files if k != "meta"}
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    return axes, meta

def _areas(out_root: str, align: str, sid: str):
    cdir = os.path.join(out_root, align, sid, "caches")
    return sorted([os.path.basename(p)[5:-4] for p in glob(os.path.join(cdir,"area_*.npz"))])

def main():
    ap = argparse.ArgumentParser(description="QC AUC/ACC curves for trained axes.")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical",
                    help="Filter by orientation for BOTH stim- and sacc-align; pooled uses all.")
    ap.add_argument("--thr", type=float, default=0.75); ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--qc_r_residC",
        action=argparse.BooleanOptionalAction,   # gives --qc_r_residC / --no-qc_r_residC
        default=False,                           # default OFF (unconstrained view)
        help="Residualize QC features vs sC before evaluating R (default: off).")
    ap.add_argument(
        "--decoder_R",
        action=argparse.BooleanOptionalAction,  # gives --decoder_R / --no-decoder_R
        default=True,                           # default ON to see the comparison
        help="Compute raw time-resolved CV decoder accuracy for direction (like paper)."
    )
    args = ap.parse_args()

    areas = args.areas or _areas(args.out_root, args.align, args.sid)
    if not areas: raise SystemExit(f"No caches found for {args.sid}")
    # time vector
    any_cache = _load_cache(args.out_root, args.align, args.sid, areas[0])
    time_s = any_cache["time"].astype(float)

    ori = None if args.orientation == "pooled" else args.orientation
    for area in areas:
        cache = _load_cache(args.out_root, args.align, args.sid, area)
        axes, _ = _load_axes(args.out_root, args.align, args.sid, area)
        curves = qc_curves_for_area(cache=cache, axes=axes, align=args.align,
                                    compute_decoder_R=args.decoder_R,
                                    time_s=time_s, orientation=ori, thr=args.thr, k_bins=args.k,qc_r_residC=args.qc_r_residC)
        out_dir = os.path.join(args.out_root, args.align, args.sid, "qc")
        os.makedirs(out_dir, exist_ok=True)
        out_pdf = os.path.join(out_dir, f"qc_axes_{area}.pdf")
        save_qc(curves, out_pdf, area)
        save_qc_json(curves, os.path.join(out_dir, f"qc_axes_{area}.json"))
        print(f"[{args.sid}][{area}] wrote {out_pdf} (+ .png, .json)")

if __name__ == "__main__":
    main()
