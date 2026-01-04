#!/usr/bin/env python
from __future__ import annotations
import argparse, os, numpy as np
from glob import glob
from typing import List, Tuple, Optional
from paperflow.flow import compute_flow_timecourse_for_pair
from paperflow.norm import rebin_cache_data
import json

def _areas(out_root: str, align: str, sid: str) -> List[str]:
    cdir = os.path.join(out_root, align, sid, "caches")
    if not os.path.isdir(cdir): return []
    return sorted([os.path.basename(p)[5:-4] for p in glob(os.path.join(cdir, "area_*.npz"))])

def _load_cache(out_root: str, align: str, sid: str, area: str, rebin_factor: int = 1):
    p = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    d = np.load(p, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    
    # Apply rebinning if requested
    if rebin_factor > 1:
        cache, _ = rebin_cache_data(cache, rebin_factor)
        cache["meta"]["rebin_factor"] = rebin_factor
        orig_bin_s = meta.get("bin_s", 0.01)
        cache["meta"]["bin_s"] = orig_bin_s * rebin_factor
    
    return cache

def _load_axes(out_root: str, align: str, sid: str, area: str, axes_tag: Optional[str], flow_tag: Optional[str]):
    candidates = []
    if axes_tag:  # highest priority
        candidates.append(os.path.join(out_root, align, sid, "axes", axes_tag, f"axes_{area}.npz"))
    if flow_tag:  # use the flow tag if axes_tag not provided
        candidates.append(os.path.join(out_root, align, sid, "axes", flow_tag, f"axes_{area}.npz"))
    candidates.append(os.path.join(out_root, align, sid, "axes", f"axes_{area}.npz"))  # legacy fallback
    for p in candidates:
        if os.path.exists(p):
            return np.load(p, allow_pickle=True)
    raise FileNotFoundError(f"axes not found for {area}; tried: {candidates}")

def _ordered_pairs(areas: List[str], subset: Optional[List[str]] = None) -> List[Tuple[str,str]]:
    A = subset if subset else areas
    return [(A[i], A[j]) for i in range(len(A)) for j in range(len(A)) if i != j]

def _parse_range_arg(val) -> Optional[Tuple[float, float]]:
    """Parse a:b into (float, float) or return None."""
    if val is None:
        return None
    if isinstance(val, str) and ":" in val:
        a, b = val.split(":")
        return float(a), float(b)
    return None

def main():
    ap = argparse.ArgumentParser(description="Compute session-level flow for all ordered pairs (or a chosen pair).")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc","targ"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--feature", choices=["C","R","S","T","O"], required=True)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--lags_ms", type=float, default=50.0)
    ap.add_argument("--verbose", action="store_true", help="Print per-pair shapes and masks")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--no_induced", action="store_true")
    ap.add_argument("--no_B_lags", action="store_true")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--perm-within",
        default="CR",
        choices=["CR", "other", "C", "R", "none"],
        help="Stratification for trial-shuffle null. Default: CR (shuffle within joint category×direction).")
    ap.add_argument("--null_method",
        default="trial_shuffle",
        choices=["trial_shuffle", "circular_shift", "phase_randomize"],
        help="Null method for flow (trial_shuffle, circular_shift, phase_randomize).")
    ap.add_argument("--standardize_mode",
        default="none",
        choices=["none", "zscore_regressors"],
        help="Standardization for regressors: none, or zscore_regressors.")
    ap.add_argument("--evoked_subtract", action="store_true",
                    help="Subtract global evoked PSTH (mean across trials per time bin) before flow.")
    ap.add_argument("--evoked_sigma_ms", type=float, default=0.0,
                    help="Gaussian smoothing sigma for evoked PSTH (ms). Default 0 (no smoothing).")
    ap.add_argument("--flow_tag_base", default=None,
        help="Base name for flow tag; final tag = <flow_tag_base>-<std>-<null>. "
             "Default: use --tag.")
    ap.add_argument("--pair", nargs=2, default=None, metavar=("AREA_A","AREA_B"))
    ap.add_argument("--tag", default="flow_v1")
    ap.add_argument("--axes_tag", default=None,
                    help="If set, read axes from axes/<axes_tag>/... "
                         "(default: try axes/<tag>/ then legacy axes/)")
    ap.add_argument("--save_null_samples", action="store_true",
                    help="Save per-permutation null samples into the flow_*.npz "
                         "(needed for old-style group DIFF p(t)). Warning: large files.")
    
    # === NEW: normalization args ===
    ap.add_argument("--norm", choices=["auto", "global", "baseline", "none"], default="auto",
                    help="Normalization mode: 'auto' (use axes meta), 'global', 'baseline', 'none'")
    ap.add_argument("--baseline_win", default=None,
                    help="Baseline window 'a:b' in seconds (only used if --norm baseline)")
    
    # === Time rebinning ===
    ap.add_argument("--rebin_factor", type=int, default=1,
                    help="Number of adjacent time bins to combine (default: 1 = no rebinning). "
                         "Must match the rebin_factor used in train_axes.py and qc_axes.py.")
    
    args = ap.parse_args()

    areas = _areas(args.out_root, args.align, args.sid)
    if not areas:
        raise SystemExit(f"No caches under {args.out_root}/{args.align}/{args.sid}/caches")

    pairs = [(args.pair[0], args.pair[1])] if args.pair else _ordered_pairs(areas)

    # Parse normalization
    norm = None if args.norm == "auto" else args.norm
    baseline_win = _parse_range_arg(args.baseline_win)

    # Get rebin factor
    rebin_factor = args.rebin_factor
    if rebin_factor > 1:
        print(f"[rebin] Combining {rebin_factor} adjacent bins (e.g., 5ms → {5*rebin_factor}ms)")

    for (A, B) in pairs:
        cA = _load_cache(args.out_root, args.align, args.sid, A, rebin_factor=rebin_factor)
        cB = _load_cache(args.out_root, args.align, args.sid, B, rebin_factor=rebin_factor)
        aA = _load_axes(args.out_root, args.align, args.sid, A, axes_tag=args.axes_tag, flow_tag=args.tag)
        aB = _load_axes(args.out_root, args.align, args.sid, B, axes_tag=args.axes_tag, flow_tag=args.tag)
        
        if args.verbose:
            import json
            metaA = cA["meta"]
            if not isinstance(metaA, dict):
                metaA = json.loads(metaA.item() if hasattr(metaA, "item") else str(metaA))
            time = cA["time"].astype(float)
            bin_s = float(metaA.get("bin_s", (time[1]-time[0] if time.size>1 else 0.01)))

            def _K(ax, feat):
                if feat == "C":
                    v = ax.get("sC", np.array([]))
                elif feat == "R":
                    v = ax.get("sR", np.array([[]]))
                elif feat == "O":
                    v = ax.get("sO", np.array([]))
                else:  # S
                    v = ax.get("sS_inv", np.array([]))
                    if v.size == 0:
                        v = ax.get("sS_raw", np.array([]))
                if v.size == 0:
                    return 0
                v = np.array(v)
                return v.shape[1] if v.ndim == 2 else 1

            print(f"[dbg] {args.sid} {A}->{B} feature={args.feature} "
                f"bin_s={bin_s*1000:.1f}ms lags_ms={args.lags_ms} "
                f"K_A={_K(aA, args.feature)} K_B={_K(aB, args.feature)} "
                f"perm_within={args.perm_within} norm={args.norm}")


        try:
            res = compute_flow_timecourse_for_pair(
                cacheA=cA, cacheB=cB, axesA=aA, axesB=aB,
                feature=args.feature, align=args.align,
                orientation=(None if args.orientation=="pooled" else args.orientation),
                pt_min_ms=(None if args.pt_min_ms is None else float(args.pt_min_ms)),
                lags_ms=float(args.lags_ms), ridge=float(args.ridge),
                perms=int(args.perms), induced=(not args.no_induced),
                include_B_lags=(not args.no_B_lags), seed=0,
                perm_within=args.perm_within,
                null_method=args.null_method,
                standardize_mode=args.standardize_mode,
                evoked_subtract=bool(args.evoked_subtract),
                evoked_sigma_ms=float(args.evoked_sigma_ms),
                save_null_samples=bool(args.save_null_samples),
                # === NEW: normalization ===
                norm=norm,
                baseline_win=baseline_win,
            )
        except ValueError as e:
            print(f"[skip] {A}->{B}: {e}")
            continue

        # build flow tag from base + standardization + null
        null_short = {"trial_shuffle": "trial",
                      "circular_shift": "circ",
                      "phase_randomize": "phase"}[args.null_method]
        std_short = {"none": "none",
                     "zscore_regressors": "zreg"}[args.standardize_mode]
        flow_tag_base = args.flow_tag_base or args.tag
        flow_tag = f"{flow_tag_base}-{std_short}-{null_short}"

        out_dir = os.path.join(args.out_root, args.align, args.sid, "flow", flow_tag, args.feature)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"flow_{args.feature}_{A}to{B}.npz")
        np.savez_compressed(out_path, **res)
        print(f"[{args.sid}][{A}->{B}] [{flow_tag}] wrote {out_path}")

if __name__ == "__main__":
    main()
