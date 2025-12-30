#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, numpy as np
from glob import glob
from typing import List, Tuple, Optional
from paperflow.axes import train_axes_for_area, save_axes, make_window_grid
from paperflow.norm import parse_win

def _parse_range_arg(val):
    """
    Accepts:
      - a list of two floats: [-0.30, -0.18]
      - a list with one string "a:b": ["-0.30:-0.18"]
      - a string "a:b"
    Returns tuple(float, float)
    """
    if isinstance(val, (list, tuple)):
        if len(val) == 2:
            return float(val[0]), float(val[1])
        if len(val) == 1 and isinstance(val[0], str) and ":" in val[0]:
            a, b = val[0].split(":")
            return float(a), float(b)
    if isinstance(val, str) and ":" in val:
        a, b = val.split(":")
        return float(a), float(b)
    raise ValueError(f"Bad range arg: {val}")

def _load_cache(out_root: str, align: str, sid: str, area: str):
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    return cache

def _discover_areas(out_root: str, align: str, sid: str):
    cdir = os.path.join(out_root, align, sid, "caches")
    return sorted([os.path.basename(p)[5:-4] for p in glob(os.path.join(cdir,"area_*.npz"))])

def _parse_win(s: str) -> tuple[float,float]:
    a,b = s.split(":"); return float(a), float(b)


def _make_window_candidates(
    search_range: str,
    lens_ms: List[float],
    step_ms: float,
) -> List[Tuple[float, float]]:
    """Generate candidate windows from CLI args."""
    t0, t1 = _parse_range_arg(search_range)
    return make_window_grid((t0, t1), lens_ms, step_ms)


def main():
    ap = new_argparser()
    args = ap.parse_args()

    areas = args.areas or _discover_areas(args.out_root, args.align, args.sid)
    if not areas:
        raise SystemExit(f"No caches under {args.out_root}/{args.align}/{args.sid}/caches")

    any_cache = _load_cache(args.out_root, args.align, args.sid, areas[0])
    time_s = any_cache["time"].astype(float)

    # choose windows by alignment
    feats = list(dict.fromkeys([f for f in args.features]))  # unique preserve order
    
    # === Normalization setup ===
    norm = args.norm
    baseline_win = None
    if norm == "baseline":
        if args.baseline_win:
            baseline_win = _parse_range_arg(args.baseline_win)
        else:
            # Use alignment-specific defaults
            if args.align == "stim":
                baseline_win = (-0.20, 0.00)
            elif args.align == "sacc":
                baseline_win = (-0.35, -0.20)
            elif args.align == "targ":
                baseline_win = (-0.15, 0.00)
            else:
                baseline_win = (-0.20, 0.00)
    
    # === C_grid setup ===
    C_grid = [float(c) for c in args.C_grid]
    
    # === Window search setup ===
    winC_candidates = None
    winS_candidates = None
    winT_candidates = None
    winR_candidates = None
    
    if args.searchC:
        search_range = args.search_range_C or _get_default_search_range(args.align, "C")
        winC_candidates = _make_window_candidates(
            search_range, args.search_len_ms, args.search_step_ms
        )
    if args.searchS:
        search_range = args.search_range_S or _get_default_search_range(args.align, "S")
        winS_candidates = _make_window_candidates(
            search_range, args.search_len_ms, args.search_step_ms
        )
    if args.searchT:
        search_range = args.search_range_T or _get_default_search_range(args.align, "T")
        winT_candidates = _make_window_candidates(
            search_range, args.search_len_ms, args.search_step_ms
        )
    if args.searchR:
        search_range = args.search_range_R or _get_default_search_range(args.align, "R")
        winR_candidates = _make_window_candidates(
            search_range, args.search_len_ms, args.search_step_ms
        )

    # choose windows by alignment
    if args.align == "stim":
        feats = [f for f in feats if f in ("C","R","T","O")]  # allow T for stim
        winC = _parse_range_arg(args.winC_stim) if (("C" in feats) or ("T" in feats) or ("O" in feats)) else None
        winR = _parse_range_arg(args.winR_stim) if "R" in feats else None
        winS = None
        winT = None  # Use winC for T in stim alignment
        pt_min = (None if args.no_pt_filter else args.pt_min_ms_stim)
    elif args.align == "sacc":
        feats = [f for f in feats if f in ("C","S","T","O")]  # allow T for sacc if desired
        winC = _parse_range_arg(args.winC_sacc) if ("C" in feats or "O" in feats) else None
        winR = None
        winS = _parse_range_arg(args.winS_sacc) if "S" in feats else None
        winT = None  # Use winC for T in sacc alignment
        pt_min = (None if args.no_pt_filter else args.pt_min_ms_sacc)
    else:  # targ
        feats = [f for f in feats if f in ("T","O")]  # T is main feature for targ alignment
        winC = None
        winR = None
        winS = None
        winT = _parse_range_arg(args.winT_targ) if "T" in feats else None
        pt_min = (None if args.no_pt_filter else args.pt_min_ms_stim)  # use stim PT threshold


    if not feats:
        raise SystemExit("No features to train for this alignment; check --features/--align.")

    saved = []
    for area in areas:
        cache = _load_cache(args.out_root, args.align, args.sid, area)
        pack = train_axes_for_area(
            cache=cache,
            feature_set=feats,
            time_s=time_s,
            winC=winC, winR=winR, winS=winS, winT=winT,
            orientation=(None if args.orientation == "pooled" else args.orientation),
            C_dim=args.dimC, R_dim=args.dimR, S_dim=args.dimS,
            make_S_invariant=(not args.no_S_invariant),
            C_grid=C_grid,
            select_mode=(args.select_mode if args.select_mode != "none" else None),
            select_frac=args.select_frac,
            pt_min_ms=pt_min,
            # === NEW: normalization ===
            norm=norm,
            baseline_win=baseline_win,
            # === NEW: classifier ===
            clf_binary=args.clf_binary,
            lda_shrinkage=args.lda_shrinkage,
            # === NEW: window search ===
            winC_candidates=winC_candidates,
            winS_candidates=winS_candidates,
            winT_candidates=winT_candidates,
            winR_candidates=winR_candidates,
        )
        axes_dir = (os.path.join(args.out_root, args.align, args.sid, "axes", args.tag)
                    if args.tag else
                    os.path.join(args.out_root, args.align, args.sid, "axes"))
        os.makedirs(axes_dir, exist_ok=True)
        path = save_axes(axes_dir, area, pack)
        print(f"[{args.sid}][{area}] wrote {path}")

    summary = dict(
        sid=args.sid, align=args.align, features=feats,
        winC=winC, winR=winR, winS=winS, winT=winT,
        pt_min_ms=pt_min,
        orientation=args.orientation,
        areas=areas,
        tag=args.tag,
        # === NEW: normalization summary ===
        norm=norm,
        baseline_win=list(baseline_win) if baseline_win else None,
        # === NEW: classifier summary ===
        clf_binary=args.clf_binary,
        C_grid=C_grid,
        lda_shrinkage=args.lda_shrinkage if args.clf_binary == "lda" else None,
        # === NEW: window search summary ===
        searchC=args.searchC,
        searchS=args.searchS,
        searchT=args.searchT,
        searchR=args.searchR,
        search_step_ms=args.search_step_ms if any([args.searchC, args.searchS, args.searchT, args.searchR]) else None,
        search_len_ms=args.search_len_ms if any([args.searchC, args.searchS, args.searchT, args.searchR]) else None,
    )
    summary_dir = axes_dir  # write summary next to the axes
    with open(os.path.join(summary_dir, "axes_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] session summary → {os.path.join(summary_dir, 'axes_summary.json')}")


def _get_default_search_range(align: str, feature: str) -> str:
    """Get default search range for a feature in a given alignment."""
    defaults = {
        "stim": {
            "C": "0.00:0.50",
            "R": "0.05:0.40",
            "T": "0.00:0.50",
        },
        "sacc": {
            "C": "-0.30:-0.05",
            "S": "-0.20:0.05",
            "T": "-0.30:0.10",
        },
        "targ": {
            "T": "0.00:0.35",
        },
    }
    return defaults.get(align, {}).get(feature, "0.00:0.30")


def new_argparser():
    ap = argparse.ArgumentParser(description="Train C/R/S/T/O subspaces per area for one session.")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."), "out"), help="Where to write outputs (default: $PAPER_HOME/out)")
    ap.add_argument("--align", choices=["stim","sacc","targ"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--features", nargs="+", default=["C","R","S","T","O"], choices=["C","R","S","T","O"])
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--tag", default=None,
                    help="Optional tag; if set, write axes to out/<align>/<sid>/axes/<tag>/")
    
    # === Normalization args ===
    ap.add_argument("--norm", choices=["global", "baseline", "none"], default="global",
                    help="Normalization mode: 'global' (cache Z), 'baseline' (z-score from baseline), 'none' (raw X)")
    ap.add_argument("--baseline_win", default=None,
                    help="Baseline window 'a:b' in seconds (required if --norm baseline, else uses defaults)")
    
    # === Binary classifier args ===
    ap.add_argument("--clf_binary", choices=["logreg", "svm", "lda"], default="logreg",
                    help="Binary classifier: logreg, svm, or lda (default: logreg)")
    ap.add_argument("--C_grid", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0],
                    help="Regularization grid for logreg/svm (default: 0.1 0.3 1 3 10)")
    ap.add_argument("--lda_shrinkage", choices=["auto", "none"], default="auto",
                    help="LDA shrinkage: 'auto' or 'none' (default: auto)")
    
    # === Window search args ===
    ap.add_argument("--searchC", action="store_true", help="Enable window search for C axis")
    ap.add_argument("--searchS", action="store_true", help="Enable window search for S axis")
    ap.add_argument("--searchT", action="store_true", help="Enable window search for T axis")
    ap.add_argument("--searchR", action="store_true", help="Enable window search for R axis")
    ap.add_argument("--search_step_ms", type=float, default=20.0,
                    help="Step size for window search in ms (default: 20)")
    ap.add_argument("--search_len_ms", nargs="+", type=float, default=[50, 80, 120],
                    help="Window lengths to try in ms (default: 50 80 120)")
    ap.add_argument("--search_range_C", default=None,
                    help="Search range for C 'a:b' in seconds (uses alignment defaults if not specified)")
    ap.add_argument("--search_range_S", default=None,
                    help="Search range for S 'a:b' in seconds (uses alignment defaults if not specified)")
    ap.add_argument("--search_range_T", default=None,
                    help="Search range for T 'a:b' in seconds (uses alignment defaults if not specified)")
    ap.add_argument("--search_range_R", default=None,
                    help="Search range for R 'a:b' in seconds (uses alignment defaults if not specified)")
    
    # windows (fixed, used when not searching)
    ap.add_argument("--winC_stim", nargs="+", default=["0.10:0.30"])
    ap.add_argument("--winR_stim", nargs="+", default=["0.05:0.20"])
    ap.add_argument("--winC_sacc", nargs="+", default=["-0.30:-0.18"])
    ap.add_argument("--winS_sacc", nargs="+", default=["-0.10:-0.03"])
    # targ-aligned windows (for T axis training: 50ms to 200ms after targets onset)
    ap.add_argument("--winT_targ", nargs="+", default=["0.05:0.20"])

    # dims
    ap.add_argument("--dimC", type=int, default=1)
    ap.add_argument("--dimR", type=int, default=4)
    ap.add_argument("--dimS", type=int, default=1)
    # invariance
    ap.add_argument("--no_S_invariant", action="store_true")
    # PT gating
    ap.add_argument("--pt_min_ms_sacc", type=float, default=200.0, help="PT≥ threshold for sacc-align; ms")
    ap.add_argument("--pt_min_ms_stim", type=float, default=200.0, help="PT≥ for stim-align (None = no PT gate)")
    ap.add_argument("--no_pt_filter", action="store_true", help="Disable PT gating")
    # optional unit selection in training (not applied at projection)
    ap.add_argument("--select_mode", choices=["none","C","R","S"], default="none",
                    help="Restrict features used in training: none, C-based, R-based (variance proxy), or S-based")
    ap.add_argument("--select_frac", type=float, default=1.0)
    return ap

if __name__ == "__main__":
    main()
