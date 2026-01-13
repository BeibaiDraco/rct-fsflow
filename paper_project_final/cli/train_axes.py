#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, numpy as np
from glob import glob
from datetime import datetime
from typing import List, Tuple, Optional
from paperflow.axes import (
    train_axes_for_area, save_axes, make_window_grid,
    train_time_resolved_axis, save_time_resolved_axes
)
from paperflow.norm import parse_win, rebin_cache_data, sliding_window_cache_data


def _generate_auto_tag(args) -> str:
    """
    Generate a descriptive tag based on training settings.
    
    Format: [method_components]-<align>-<orientation>
    
    Method components (only non-default settings included):
    - timeresolved: included if --time_resolved is set
    - norm: 'baseline' or 'none' (skip if 'global' which is default)
    - clf: 'svm' or 'lda' (skip if 'logreg' which is default)
    - winsearch: included if any --search* flag is set
    - peakbin/meanbin: included if search_score_mode is not 'cv_auc'
    
    Examples:
    - Default settings: "default-stim-vertical"
    - Window search only: "winsearch-stim-vertical"
    - Window search with peak-bin scoring: "winsearch-peakbin-stim-vertical"
    - Time-resolved: "timeresolved-stim-vertical"
    - Baseline + SVM: "baseline-svm-stim-vertical"
    """
    parts = []
    
    # Time-resolved mode (highest priority in naming)
    if getattr(args, 'time_resolved', False):
        parts.append("timeresolved")
    
    # Normalization (skip default 'global')
    if args.norm != "global":
        parts.append(args.norm)
    
    # Classifier (skip default 'logreg')
    if args.clf_binary != "logreg":
        parts.append(args.clf_binary)
    
    # Window search (not applicable for time-resolved)
    if not getattr(args, 'time_resolved', False):
        if any([args.searchC, args.searchS, args.searchT, args.searchR]):
            parts.append("winsearch")
            # Include score mode if not default
            if args.search_score_mode == "peak_bin_auc":
                parts.append("peakbin")
            elif args.search_score_mode == "mean_bin_auc":
                parts.append("meanbin")
    
    # If no special settings, use 'default'
    if not parts:
        parts.append("default")
    
    # Always include alignment and orientation
    parts.append(args.align)
    parts.append(args.orientation)
    
    return "-".join(parts)


def _save_full_config(axes_dir: str, args, computed_values: dict):
    """
    Save complete training configuration for reproducibility.
    
    This includes:
    - All CLI arguments (raw values)
    - Computed/resolved values (e.g., auto-generated tag, resolved windows)
    - Metadata (timestamp, version)
    """
    # Convert args namespace to dict
    args_dict = vars(args).copy()
    
    # Convert any non-serializable types
    for k, v in args_dict.items():
        if isinstance(v, np.ndarray):
            args_dict[k] = v.tolist()
    
    config = {
        "_meta": {
            "created": datetime.now().isoformat(),
            "script": "cli/train_axes.py",
            "version": "2.0",  # Version with auto-tag support
        },
        "cli_args": args_dict,
        "resolved": computed_values,
    }
    
    config_path = os.path.join(axes_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    return config_path


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

def _load_cache(out_root: str, align: str, sid: str, area: str, 
                rebin_factor: int = 1,
                sliding_window_bins: int = 0, sliding_step_bins: int = 0):
    """
    Load cache with optional rebinning or sliding window.
    
    Parameters
    ----------
    rebin_factor : int
        If > 1, apply non-overlapping rebinning (legacy mode).
    sliding_window_bins : int
        If > 0, apply sliding window with this many bins per window.
    sliding_step_bins : int
        If sliding_window_bins > 0, step by this many bins.
    
    Note: sliding window takes precedence over rebin_factor if both specified.
    """
    path = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
    orig_bin_s = meta.get("bin_s", 0.01)
    
    # Sliding window takes precedence over rebinning
    if sliding_window_bins > 0 and sliding_step_bins > 0:
        cache, _ = sliding_window_cache_data(cache, sliding_window_bins, sliding_step_bins)
        # Update meta to record sliding window params
        cache["meta"]["sliding_window_bins"] = sliding_window_bins
        cache["meta"]["sliding_step_bins"] = sliding_step_bins
        cache["meta"]["window_ms"] = sliding_window_bins * orig_bin_s * 1000
        cache["meta"]["step_ms"] = sliding_step_bins * orig_bin_s * 1000
        cache["meta"]["bin_s"] = orig_bin_s * sliding_step_bins  # effective bin spacing
    elif rebin_factor > 1:
        cache, _ = rebin_cache_data(cache, rebin_factor)
        # Update meta to record rebinning
        cache["meta"]["rebin_factor"] = rebin_factor
        cache["meta"]["bin_s"] = orig_bin_s * rebin_factor
    
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

    # === Auto-generate tag if not provided ===
    if args.tag is None and not args.no_auto_tag:
        args.tag = _generate_auto_tag(args)
        print(f"[auto-tag] Generated tag: {args.tag}")
    elif args.tag is None and args.no_auto_tag:
        print("[warning] No tag specified and --no_auto_tag set. Writing to legacy location (may overwrite).")

    areas = args.areas or _discover_areas(args.out_root, args.align, args.sid)
    if not areas:
        raise SystemExit(f"No caches under {args.out_root}/{args.align}/{args.sid}/caches")

    # Get rebin factor and sliding window params
    rebin_factor = args.rebin_factor
    sliding_window_bins = 0
    sliding_step_bins = 0
    
    # Load one cache to get native bin size for sliding window calculation
    temp_path = os.path.join(args.out_root, args.align, args.sid, "caches", f"area_{areas[0]}.npz")
    temp_d = np.load(temp_path, allow_pickle=True)
    temp_meta = json.loads(temp_d["meta"].item()) if "meta" in temp_d else {}
    native_bin_s = temp_meta.get("bin_s", 0.01)
    native_bin_ms = native_bin_s * 1000.0
    
    # Sliding window takes precedence over rebinning
    if args.sliding_window_ms > 0 and args.sliding_step_ms > 0:
        # Compute bin counts from ms
        if args.sliding_window_ms % native_bin_ms != 0:
            raise SystemExit(f"sliding_window_ms ({args.sliding_window_ms}) must be multiple of native bin ({native_bin_ms}ms)")
        if args.sliding_step_ms % native_bin_ms != 0:
            raise SystemExit(f"sliding_step_ms ({args.sliding_step_ms}) must be multiple of native bin ({native_bin_ms}ms)")
        
        sliding_window_bins = int(round(args.sliding_window_ms / native_bin_ms))
        sliding_step_bins = int(round(args.sliding_step_ms / native_bin_ms))
        print(f"[sliding-window] window={args.sliding_window_ms}ms ({sliding_window_bins} bins), "
              f"step={args.sliding_step_ms}ms ({sliding_step_bins} bins), native={native_bin_ms}ms")
        rebin_factor = 1  # Disable rebinning when using sliding window
    elif rebin_factor > 1:
        print(f"[rebin] Combining {rebin_factor} adjacent bins")

    any_cache = _load_cache(args.out_root, args.align, args.sid, areas[0], 
                            rebin_factor=rebin_factor,
                            sliding_window_bins=sliding_window_bins,
                            sliding_step_bins=sliding_step_bins)
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
    
    # === TIME-RESOLVED MODE (special case: train axis per time bin) ===
    if args.time_resolved:
        print(f"[time-resolved] Training separate axis for each time bin")
        
        # Only support single feature for time-resolved
        if len(feats) != 1:
            raise SystemExit("Time-resolved mode requires exactly one feature (e.g., --features C)")
        feature = feats[0]
        if feature not in ("C", "S", "T"):
            raise SystemExit(f"Time-resolved only supports C, S, or T features, not {feature}")
        
        # Time range (optional)
        time_range = None
        if args.time_resolved_range:
            time_range = _parse_range_arg(args.time_resolved_range)
        
        # Orientation
        ori = None if args.orientation == "pooled" else args.orientation
        
        # PT threshold
        if args.no_pt_filter:
            pt_min = None
        elif args.align == "sacc":
            pt_min = args.pt_min_ms_sacc
        else:
            pt_min = args.pt_min_ms_stim
        
        per_area_results = {}
        axes_dir = (os.path.join(args.out_root, args.align, args.sid, "axes", args.tag)
                    if args.tag else
                    os.path.join(args.out_root, args.align, args.sid, "axes"))
        os.makedirs(axes_dir, exist_ok=True)
        
        for area in areas:
            cache = _load_cache(args.out_root, args.align, args.sid, area, 
                                rebin_factor=rebin_factor,
                                sliding_window_bins=sliding_window_bins,
                                sliding_step_bins=sliding_step_bins)
            result = train_time_resolved_axis(
                cache=cache,
                time_s=time_s,
                feature=feature,
                orientation=ori,
                pt_min_ms=pt_min,
                norm=norm,
                baseline_win=baseline_win,
                clf_binary=args.clf_binary,
                C_grid=C_grid,
                lda_shrinkage=args.lda_shrinkage,
                time_range=time_range,
            )
            path = save_time_resolved_axes(axes_dir, area, result)
            print(f"[{args.sid}][{area}] wrote {path}")
            print(f"    → mean CV AUC: {result['meta']['mean_cv_auc']:.4f}, "
                  f"max: {result['meta']['max_cv_auc']:.4f} at {result['meta']['peak_time_ms']:.0f}ms")
            
            per_area_results[area] = {
                "mean_cv_auc": result["meta"]["mean_cv_auc"],
                "max_cv_auc": result["meta"]["max_cv_auc"],
                "peak_time_ms": result["meta"]["peak_time_ms"],
                "n_trials": result["meta"]["n_trials"],
            }
        
        # Save summary
        summary = dict(
            sid=args.sid, align=args.align, feature=feature,
            time_resolved=True,
            time_range=list(time_range) if time_range else None,
            pt_min_ms=pt_min,
            orientation=args.orientation,
            areas=areas,
            tag=args.tag,
            norm=norm,
            baseline_win=list(baseline_win) if baseline_win else None,
            clf_binary=args.clf_binary,
            C_grid=C_grid,
            per_area_results=per_area_results,
            # === Sliding window ===
            sliding_window_ms=args.sliding_window_ms if sliding_window_bins > 0 else None,
            sliding_step_ms=args.sliding_step_ms if sliding_step_bins > 0 else None,
            sliding_window_bins=sliding_window_bins if sliding_window_bins > 0 else None,
            sliding_step_bins=sliding_step_bins if sliding_step_bins > 0 else None,
        )
        summary_path = os.path.join(axes_dir, "axes_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ok] axes_summary.json → {summary_path}")
        
        # Save config
        computed_values = dict(
            auto_tag_generated=(args.tag == _generate_auto_tag(args)) if not args.no_auto_tag else False,
            final_tag=args.tag,
            time_resolved=True,
            feature=feature,
            areas_processed=areas,
        )
        config_path = _save_full_config(axes_dir, args, computed_values)
        print(f"[ok] config.json → {config_path}")
        
        return  # Exit after time-resolved training
    
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

    # Collect per-area results for summary
    per_area_results = {}
    
    for area in areas:
        cache = _load_cache(args.out_root, args.align, args.sid, area, 
                            rebin_factor=rebin_factor,
                            sliding_window_bins=sliding_window_bins,
                            sliding_step_bins=sliding_step_bins)
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
            # === NEW: window search scoring and tiebreak ===
            search_score_mode=args.search_score_mode,
            search_tiebreak=args.search_tiebreak,
            search_tol=args.search_tol,
            search_event_time_s=0.0,  # always 0.0 for now (saccade time reference)
        )
        axes_dir = (os.path.join(args.out_root, args.align, args.sid, "axes", args.tag)
                    if args.tag else
                    os.path.join(args.out_root, args.align, args.sid, "axes"))
        os.makedirs(axes_dir, exist_ok=True)
        path = save_axes(axes_dir, area, pack)
        print(f"[{args.sid}][{area}] wrote {path}")
        
        # Collect per-area metadata for summary
        area_info = {}
        meta = pack.meta if hasattr(pack, 'meta') else {}
        
        # Window selections (if search was used)
        if meta.get("winC_selected"):
            area_info["winC_selected"] = meta["winC_selected"]
            area_info["winC_cv_best"] = meta.get("winC_cv_best")
            area_info["winC_best_param"] = meta.get("winC_best_param")
            # Store peak-bin info if available
            if meta.get("winC_peak_time_ms") is not None:
                area_info["winC_peak_time_ms"] = meta["winC_peak_time_ms"]
                area_info["winC_peak_auc"] = meta.get("winC_peak_auc")
                area_info["winC_mean_bin_auc"] = meta.get("winC_mean_bin_auc")
                area_info["winC_cv_auc_original"] = meta.get("winC_cv_auc_original")
        if meta.get("winS_selected"):
            area_info["winS_selected"] = meta["winS_selected"]
            area_info["winS_cv_best"] = meta.get("winS_cv_best")
            if meta.get("winS_peak_time_ms") is not None:
                area_info["winS_peak_time_ms"] = meta["winS_peak_time_ms"]
                area_info["winS_peak_auc"] = meta.get("winS_peak_auc")
        if meta.get("winT_selected"):
            area_info["winT_selected"] = meta["winT_selected"]
            area_info["winT_cv_best"] = meta.get("winT_cv_best")
            if meta.get("winT_peak_time_ms") is not None:
                area_info["winT_peak_time_ms"] = meta["winT_peak_time_ms"]
                area_info["winT_peak_auc"] = meta.get("winT_peak_auc")
        if meta.get("winR_selected"):
            area_info["winR_selected"] = meta["winR_selected"]
            area_info["winR_cv_best"] = meta.get("winR_cv_best")
        
        # CV scores for each axis
        if meta.get("sC_auc_mean") is not None:
            area_info["sC_auc"] = meta["sC_auc_mean"]
        if meta.get("sS_inv_auc_mean") is not None:
            area_info["sS_inv_auc"] = meta["sS_inv_auc_mean"]
        if meta.get("sT_auc_mean") is not None:
            area_info["sT_auc"] = meta["sT_auc_mean"]
        if meta.get("acc_R_macro") is not None:
            area_info["acc_R_macro"] = meta["acc_R_macro"]
        
        # Trial counts
        if meta.get("sC_n") is not None:
            area_info["n_trials_C"] = meta["sC_n"]
        
        if area_info:
            per_area_results[area] = area_info
            # Print per-area window selection if search was used
            if "winC_selected" in area_info:
                score_str = f"score: {area_info.get('winC_cv_best', 'N/A'):.4f}" if isinstance(area_info.get('winC_cv_best'), (int, float)) else ""
                peak_str = ""
                if area_info.get("winC_peak_time_ms") is not None:
                    peak_str = f", peak@{area_info['winC_peak_time_ms']:.0f}ms"
                    if area_info.get("winC_cv_auc_original") is not None:
                        peak_str += f" (cv_auc={area_info['winC_cv_auc_original']:.4f})"
                print(f"    → winC_selected: {area_info['winC_selected']}, {score_str}{peak_str}")

    # === Build summary (backward-compatible format) ===
    has_search = any([args.searchC, args.searchS, args.searchT, args.searchR])
    summary = dict(
        sid=args.sid, align=args.align, features=feats,
        winC=list(winC) if winC else None,
        winR=list(winR) if winR else None,
        winS=list(winS) if winS else None,
        winT=list(winT) if winT else None,
        pt_min_ms=pt_min,
        orientation=args.orientation,
        areas=areas,
        tag=args.tag,
        # === Normalization ===
        norm=norm,
        baseline_win=list(baseline_win) if baseline_win else None,
        # === Classifier ===
        clf_binary=args.clf_binary,
        C_grid=C_grid,
        lda_shrinkage=args.lda_shrinkage if args.clf_binary == "lda" else None,
        # === Window search ===
        searchC=args.searchC,
        searchS=args.searchS,
        searchT=args.searchT,
        searchR=args.searchR,
        search_step_ms=args.search_step_ms if has_search else None,
        search_len_ms=args.search_len_ms if has_search else None,
        search_range_C=args.search_range_C if args.searchC else None,
        search_range_S=args.search_range_S if args.searchS else None,
        search_range_T=args.search_range_T if args.searchT else None,
        search_range_R=args.search_range_R if args.searchR else None,
        # === Window search scoring and tiebreak ===
        search_score_mode=args.search_score_mode if has_search else None,
        search_tiebreak=args.search_tiebreak if has_search else None,
        search_tol=args.search_tol if has_search else None,
        # === Per-area results (window selections, CV scores) ===
        per_area_results=per_area_results if per_area_results else None,
        # === Time rebinning ===
        rebin_factor=rebin_factor if rebin_factor > 1 else None,
        # === Sliding window ===
        sliding_window_ms=args.sliding_window_ms if sliding_window_bins > 0 else None,
        sliding_step_ms=args.sliding_step_ms if sliding_step_bins > 0 else None,
        sliding_window_bins=sliding_window_bins if sliding_window_bins > 0 else None,
        sliding_step_bins=sliding_step_bins if sliding_step_bins > 0 else None,
    )
    
    # Write axes_summary.json (backward-compatible)
    summary_path = os.path.join(axes_dir, "axes_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] axes_summary.json → {summary_path}")
    
    # === Save full config (new detailed format) ===
    computed_values = dict(
        auto_tag_generated=(args.tag == _generate_auto_tag(args)) if not args.no_auto_tag else False,
        final_tag=args.tag,
        resolved_baseline_win=list(baseline_win) if baseline_win else None,
        resolved_winC=list(winC) if winC else None,
        resolved_winR=list(winR) if winR else None,
        resolved_winS=list(winS) if winS else None,
        resolved_winT=list(winT) if winT else None,
        resolved_pt_min_ms=pt_min,
        features_trained=feats,
        areas_processed=areas,
        winC_candidates_count=len(winC_candidates) if winC_candidates else 0,
        winS_candidates_count=len(winS_candidates) if winS_candidates else 0,
        winT_candidates_count=len(winT_candidates) if winT_candidates else 0,
        winR_candidates_count=len(winR_candidates) if winR_candidates else 0,
        # === Sliding window ===
        sliding_window_bins=sliding_window_bins if sliding_window_bins > 0 else None,
        sliding_step_bins=sliding_step_bins if sliding_step_bins > 0 else None,
        native_bin_ms=native_bin_ms,
    )
    config_path = _save_full_config(axes_dir, args, computed_values)
    print(f"[ok] config.json → {config_path}")


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
                    help="Output tag; if not set, auto-generates from settings (e.g., 'baseline-svm-winsearch-stim-vertical')")
    ap.add_argument("--no_auto_tag", action="store_true",
                    help="Disable auto-tag generation; write to legacy location if --tag not provided")
    
    # === Time-resolved mode ===
    ap.add_argument("--time_resolved", action="store_true",
                    help="Train separate axis for each time bin (time-resolved decoding). "
                         "Requires exactly one feature (e.g., --features C).")
    ap.add_argument("--time_resolved_range", default=None,
                    help="Optional time range 'a:b' in seconds to restrict time-resolved training")
    
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
    ap.add_argument("--search_len_ms", nargs="+", type=float, default=[30, 50, 100, 150, 200, 250],
                    help="Window lengths to try in ms (default: 30 50 100 150 200 250)")
    ap.add_argument("--search_range_C", default=None,
                    help="Search range for C 'a:b' in seconds (uses alignment defaults if not specified)")
    ap.add_argument("--search_range_S", default=None,
                    help="Search range for S 'a:b' in seconds (uses alignment defaults if not specified)")
    ap.add_argument("--search_range_T", default=None,
                    help="Search range for T 'a:b' in seconds (uses alignment defaults if not specified)")
    ap.add_argument("--search_range_R", default=None,
                    help="Search range for R 'a:b' in seconds (uses alignment defaults if not specified)")
    
    # === Window search scoring and tiebreak ===
    ap.add_argument("--search_score_mode",
                    choices=["cv_auc", "peak_bin_auc", "mean_bin_auc"],
                    default="cv_auc",
                    help="How to score candidate windows during search: "
                         "'cv_auc' (default, CV AUC from window-mean), "
                         "'peak_bin_auc' (max per-bin AUC within window), "
                         "'mean_bin_auc' (mean per-bin AUC within window)")
    ap.add_argument("--search_tiebreak",
                    choices=["none", "shortest_then_earliest", "shortest_then_closest0", "earliest"],
                    default="none",
                    help="Tie-break rule among windows within --search_tol of the best score: "
                         "'none' (just pick best), "
                         "'shortest_then_earliest' (prefer shorter windows, then earlier start), "
                         "'shortest_then_closest0' (prefer shorter, then center closest to 0), "
                         "'earliest' (prefer earliest start)")
    ap.add_argument("--search_tol", type=float, default=0.0,
                    help="Tolerance for tie-break (absolute AUC units). "
                         "E.g. 0.01 means consider windows within 0.01 of best score.")
    
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
    
    # === Time rebinning ===
    ap.add_argument("--rebin_factor", type=int, default=1,
                    help="Number of adjacent time bins to combine (default: 1 = no rebinning). "
                         "Set to 2 for sacc alignment to convert 5ms bins to 10ms bins.")
    
    # === Sliding window (alternative to rebinning) ===
    ap.add_argument("--sliding_window_ms", type=float, default=0.0,
                    help="Sliding window width in ms (e.g., 20). If > 0, uses sliding window "
                         "instead of rebinning. Window averages over this duration.")
    ap.add_argument("--sliding_step_ms", type=float, default=0.0,
                    help="Sliding window step in ms (e.g., 10). Output bins are spaced by this amount. "
                         "For STIM (10ms native): step_ms=10 → step_bins=1. "
                         "For SACC (5ms native): step_ms=10 → step_bins=2.")
    return ap

if __name__ == "__main__":
    main()
