#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from typing import Dict, List, Tuple
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _sessions_with_any_area(out_root: str, align: str, min_areas: int) -> List[str]:
    root = os.path.join(out_root, align)
    sids = [s for s in os.listdir(root) if s.isdigit() and os.path.isdir(os.path.join(root, s))]
    keep = []
    for sid in sorted(sids):
        cdir = os.path.join(root, sid, "caches")
        areas = [f for f in os.listdir(cdir) if f.startswith("area_") and f.endswith(".npz")] if os.path.isdir(cdir) else []
        if len(areas) >= min_areas:
            keep.append(sid)
    return keep

def _load_one_area_npz(out_root: str, align: str, sid: str, area: str) -> Dict:
    p = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    return np.load(p, allow_pickle=True)

def _collect_PT(out_root: str, align: str, sids: List[str],
                correct_only: bool, orientation: str|None) -> np.ndarray:
    pts = []
    for sid in sids:
        cdir = os.path.join(out_root, align, sid, "caches")
        if not os.path.isdir(cdir): continue
        for f in os.listdir(cdir):
            if not (f.startswith("area_") and f.endswith(".npz")): continue
            d = np.load(os.path.join(cdir, f), allow_pickle=True)
            if "lab_PT_ms" not in d: continue
            ok = np.ones(d["lab_PT_ms"].shape[0], dtype=bool)
            if correct_only and "lab_is_correct" in d:
                ok &= d["lab_is_correct"].astype(bool)
            if orientation is not None and "lab_orientation" in d:
                ok &= (d["lab_orientation"].astype(str) == orientation)
            PT = d["lab_PT_ms"].astype(float)[ok]
            pts.append(PT)
            # take PT once per session; break to avoid duplicating trials across areas
            break
    if not pts:
        return np.array([], dtype=float)
    return np.concatenate(pts, axis=0)

def main():
    ap = argparse.ArgumentParser(description="Processing time (PT) histograms from cached trials.")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc"], default="stim")
    ap.add_argument("--min_areas", type=int, default=2, help="Only include sessions with >= this many cached areas.")
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="pooled")
    ap.add_argument("--correct_only", action="store_true", default=True)
    ap.add_argument("--bins_ms", type=int, default=25)
    ap.add_argument("--xmax_ms", type=int, default=600)
    args = ap.parse_args()

    sids = _sessions_with_any_area(args.out_root, args.align, args.min_areas)
    if not sids:
        raise SystemExit(f"No sessions with >= {args.min_areas} areas under {args.out_root}/{args.align}")

    ori = None if args.orientation=="pooled" else args.orientation
    PT = _collect_PT(args.out_root, args.align, sids, args.correct_only, ori)
    out_dir = os.path.join(args.out_root, "qc", "pt_hist", args.align)
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6.8,3.2))
    edges = np.arange(0, args.xmax_ms+args.bins_ms, args.bins_ms, dtype=float)
    plt.hist(PT, bins=edges, color="C0", alpha=0.8, edgecolor="none")
    plt.axvline(200, ls="--", c="k", lw=1.2, label="PT = 200 ms")
    plt.xlabel("Processing time (ms)")
    plt.ylabel("# trials")
    ttl = f"PT histogram ({args.align}, {args.orientation}, correct_only={args.correct_only})\nN={PT.size}, sessions={len(sids)}"
    plt.title(ttl)
    plt.tight_layout()
    pdf = os.path.join(out_dir, f"pt_hist_{args.orientation}_minA{args.min_areas}.pdf")
    plt.savefig(pdf); plt.savefig(pdf.replace(".pdf",".png"), dpi=300); plt.close()
    print(f"[ok] wrote {pdf} (+ .png). N={PT.size}")

if __name__ == "__main__":
    main()
