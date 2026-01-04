#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, numpy as np
from glob import glob
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paperflow.qc import qc_curves_for_area
from paperflow.norm import rebin_cache_data

def _load_cache(out_root, align, sid, area, rebin_factor: int = 1):
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

def _load_axes(out_root, align, sid, area, tag=None):
    candidates = []
    if tag:
        candidates.append(os.path.join(out_root, align, sid, "axes", tag, f"axes_{area}.npz"))
    candidates.append(os.path.join(out_root, align, sid, "axes", f"axes_{area}.npz"))  # legacy fallback
    for p in candidates:
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            meta = json.loads(d["meta"].item()) if "meta" in d else {}
            axes = {k: d[k] for k in d.files if k != "meta"}
            axes["meta"] = meta
            return axes
    raise FileNotFoundError(f"axes not found for {area} (tried: {candidates})")

def _areas(out_root, align, sid):
    cdir = os.path.join(out_root, align, sid, "caches")
    return sorted([os.path.basename(p)[5:-4] for p in glob(os.path.join(cdir, "area_*.npz"))])

def _save_json(curves, out_json):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = dict(
        time=curves.time.tolist(),
        auc_C=None if curves.auc_C is None else curves.auc_C.tolist(),
        auc_S_raw=None if curves.auc_S_raw is None else curves.auc_S_raw.tolist(),
        auc_S_inv=None if curves.auc_S_inv is None else curves.auc_S_inv.tolist(),
        acc_R_macro=None if curves.acc_R_macro is None else curves.acc_R_macro.tolist(),
        auc_T=None if curves.auc_T is None else curves.auc_T.tolist(),
        latencies_ms=dict(C=curves.lat_C_ms, S_raw=curves.lat_S_raw_ms, S_inv=curves.lat_S_inv_ms, T=curves.lat_T_ms),
        meta=curves.meta,
    )
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

def _plot_curves(curves, out_pdf, area):
    tms = curves.time * 1000.0
    plt.figure(figsize=(7.6, 3.8))
    plt.axvline(0, ls="--", c="k", lw=0.8)
    if curves.auc_C is not None:
        h = plt.plot(tms, curves.auc_C, lw=2.2, label="AUC(C | sC)")[0]
        plt.axhline(0.5, ls=":", c=h.get_color(), lw=1.0)
    if curves.auc_S_inv is not None:
        h = plt.plot(tms, curves.auc_S_inv, lw=2.2, label="AUC(S | sS inv)")[0]
        plt.axhline(0.5, ls=":", c=h.get_color(), lw=1.0)
    if curves.auc_S_raw is not None:
        plt.plot(tms, curves.auc_S_raw, lw=1.6, ls="--", label="AUC(S | sS raw)")
    if curves.acc_R_macro is not None:
        h = plt.plot(tms, curves.acc_R_macro, lw=2.2, label="ACC(R | sR) (within C)")[0]
        plt.axhline(1.0/3.0, ls=":", c=h.get_color(), lw=1.0)
    if curves.auc_T is not None:
        h = plt.plot(tms, curves.auc_T, lw=2.2, label="AUC(T | sT)")[0]
        plt.axhline(0.5, ls=":", c=h.get_color(), lw=1.0)

    # Build title with normalization info
    norm_str = curves.meta.get('norm', 'global')
    title = f"{area} — QC ({curves.meta.get('align','?')}, ori={curves.meta.get('orientation')}, PT≥{curves.meta.get('pt_min_ms')}, norm={norm_str})"
    
    plt.xlabel("Time (ms)")
    plt.ylabel("AUC / Accuracy")
    plt.title(title)
    plt.legend(loc="lower right", ncol=2, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf)
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300)
    plt.close()


def _parse_range_arg(val) -> Optional[Tuple[float, float]]:
    """Parse a:b into (float, float) or return None."""
    if val is None:
        return None
    if isinstance(val, str) and ":" in val:
        a, b = val.split(":")
        return float(a), float(b)
    return None


# ==================== TIME-RESOLVED QC ====================

def _is_time_resolved(axes):
    """Check if axes are time-resolved (have sC_time_resolved)."""
    return "sC_time_resolved" in axes and axes["sC_time_resolved"].ndim == 2

def _qc_time_resolved(cache, axes, time_s, orientation, pt_min_ms, align):
    """
    QC for time-resolved axes: compute AUC at each time bin using
    the axis trained at that same time bin (diagonal decoding).
    """
    from sklearn.metrics import roc_auc_score
    from paperflow.norm import get_Z, baseline_stats, get_axes_norm, get_axes_norm_mode, get_axes_baseline_win
    
    meta = axes.get("meta", {})
    if isinstance(meta, str):
        meta = json.loads(meta)
    
    feature = meta.get("feature", "C")
    
    # Time-resolved axes: (n_bins, n_units)
    axes_matrix = axes["sC_time_resolved"]
    n_bins, n_units = axes_matrix.shape
    
    # Build trial mask
    N_total = cache["Z"].shape[0]
    C_raw = cache.get("lab_C", np.full(N_total, np.nan)).astype(np.float64)
    S_raw = cache.get("lab_S", np.full(N_total, np.nan)).astype(np.float64)
    if "lab_T" in cache:
        T_raw = cache.get("lab_T", np.full(N_total, np.nan)).astype(np.float64)
    else:
        T_raw = np.sign(C_raw) * np.sign(S_raw)
        T_raw[~(np.isfinite(C_raw) & np.isfinite(S_raw))] = np.nan
    OR_raw = cache.get("lab_orientation", np.array(["pooled"] * N_total, dtype=object))
    PT_raw = cache.get("lab_PT_ms", None)
    IC_raw = cache.get("lab_is_correct", np.ones(N_total, dtype=bool))
    
    keep = np.ones(N_total, dtype=bool)
    keep &= IC_raw
    if orientation is not None and "lab_orientation" in cache:
        keep &= (OR_raw.astype(str) == orientation)
    if pt_min_ms is not None and PT_raw is not None:
        keep &= np.isfinite(PT_raw) & (PT_raw >= float(pt_min_ms))
    
    # Get normalization from axes meta
    norm_mode = meta.get("norm", "global")
    baseline_win = meta.get("baseline_win")
    if baseline_win:
        baseline_win = tuple(baseline_win)
    
    # Get normalized data
    axes_norm = None
    if "norm_mu" in axes and len(axes["norm_mu"]) > 0:
        axes_norm = {"mu": axes["norm_mu"], "sd": axes["norm_sd"]}
    
    Z, _ = get_Z(cache, time_s, keep, norm_mode, baseline_win, axes_norm=axes_norm)
    
    # Select label
    if feature == "C":
        y_raw = C_raw[keep]
    elif feature == "S":
        y_raw = S_raw[keep]
    elif feature == "T":
        y_raw = T_raw[keep]
    else:
        y_raw = C_raw[keep]
    
    valid = np.isfinite(y_raw)
    y = y_raw[valid]
    Z_valid = Z[valid]
    
    # Compute AUC at each time bin using diagonal decoding
    auc_diagonal = np.full(n_bins, np.nan)
    y_binary = (y > 0).astype(int)
    
    if np.unique(y_binary).size >= 2:
        for t_idx in range(n_bins):
            axis_t = axes_matrix[t_idx]
            if np.linalg.norm(axis_t) < 1e-10:
                continue
            # Project activity at time t onto axis trained at time t
            proj = Z_valid[:, t_idx, :] @ axis_t
            try:
                auc_diagonal[t_idx] = roc_auc_score(y_binary, proj)
            except:
                pass
    
    return {
        "time": time_s,
        "auc_diagonal": auc_diagonal,
        "cv_auc_train": axes.get("cv_auc", np.full(n_bins, np.nan)),
        "meta": {
            "time_resolved": True,
            "feature": feature,
            "align": align,
            "orientation": orientation,
            "pt_min_ms": pt_min_ms,
            "norm": norm_mode,
            "n_trials": int(valid.sum()),
        }
    }

def _plot_time_resolved_qc(result, out_pdf, area):
    """Plot time-resolved QC: CV AUC from training (honest estimate)."""
    tms = result["time"] * 1000.0
    
    plt.figure(figsize=(8, 4))
    plt.axvline(0, ls="--", c="k", lw=0.8)
    plt.axhline(0.5, ls=":", c="gray", lw=1.0)
    
    # CV AUC from training (this is the honest cross-validated estimate)
    cv_auc = result.get("cv_auc_train", result.get("auc_diagonal"))
    if cv_auc is not None and np.any(np.isfinite(cv_auc)):
        plt.plot(tms, cv_auc, lw=2.5, label="CV AUC (per time bin)", color="C0")
        
        # Mark peak
        peak_idx = np.nanargmax(cv_auc)
        peak_auc = cv_auc[peak_idx]
        peak_time = tms[peak_idx]
        plt.scatter([peak_time], [peak_auc], s=100, c="C0", zorder=5, 
                    marker="*", label=f"Peak: {peak_auc:.3f} @ {peak_time:.0f}ms")
    
    feature = result["meta"].get("feature", "C")
    norm_str = result["meta"].get("norm", "global")
    n_trials = result["meta"].get("n_trials", "?")
    title = (f"{area} — Time-Resolved Decoding ({feature})\n"
             f"align={result['meta'].get('align')}, ori={result['meta'].get('orientation')}, "
             f"PT≥{result['meta'].get('pt_min_ms')}, norm={norm_str}, n={n_trials}")
    
    plt.xlabel("Time (ms)")
    plt.ylabel("CV AUC (5-fold)")
    plt.title(title)
    plt.legend(loc="lower right", frameon=False)
    plt.ylim(0.4, 1.0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf)
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300)
    plt.close()

def _save_time_resolved_json(result, out_json):
    """Save time-resolved QC results to JSON."""
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "time": result["time"].tolist(),
        "auc_diagonal": result["auc_diagonal"].tolist(),
        "cv_auc_train": result["cv_auc_train"].tolist() if "cv_auc_train" in result else None,
        "meta": result["meta"],
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="QC curves for trained axes (per area).")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc","targ"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--tag", default=None,
                    help="Optional tag; if set, read axes from axes/<tag>/ and write QC to qc/<tag>/")
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pt_min_ms_sacc", type=float, default=200.0)
    ap.add_argument("--pt_min_ms_stim", type=float, default=200.0)
    ap.add_argument("--pt_min_ms_targ", type=float, default=200.0)
    ap.add_argument("--no_pt_filter", action="store_true")
    
    # === NEW: normalization args ===
    ap.add_argument("--norm", choices=["auto", "global", "baseline", "none"], default="auto",
                    help="Normalization mode: 'auto' (use axes meta), 'global', 'baseline', 'none'")
    ap.add_argument("--baseline_win", default=None,
                    help="Baseline window 'a:b' in seconds (only used if --norm baseline)")
    ap.add_argument("--force_norm_mismatch", action="store_true",
                    help="Allow normalization mode to differ from axes training (not recommended)")
    
    # === Time rebinning ===
    ap.add_argument("--rebin_factor", type=int, default=1,
                    help="Number of adjacent time bins to combine (default: 1 = no rebinning). "
                         "Must match the rebin_factor used in train_axes.py.")
    
    args = ap.parse_args()

    areas = args.areas or _areas(args.out_root, args.align, args.sid)
    if not areas:
        raise SystemExit(f"No caches found under {args.out_root}/{args.align}/{args.sid}/caches")

    # Get rebin factor
    rebin_factor = args.rebin_factor
    if rebin_factor > 1:
        print(f"[rebin] Combining {rebin_factor} adjacent bins (e.g., 5ms → {5*rebin_factor}ms)")

    any_cache = _load_cache(args.out_root, args.align, args.sid, areas[0], rebin_factor=rebin_factor)
    time_s = any_cache["time"].astype(float)

    ori = None if args.orientation == "pooled" else args.orientation
    if args.no_pt_filter:
        pt_thr = None
    elif args.align == "sacc":
        pt_thr = args.pt_min_ms_sacc
    elif args.align == "targ":
        pt_thr = args.pt_min_ms_targ
    else:
        pt_thr = args.pt_min_ms_stim

    for area in areas:
        cache = _load_cache(args.out_root, args.align, args.sid, area, rebin_factor=rebin_factor)
        axes  = _load_axes(args.out_root, args.align, args.sid, area, tag=args.tag)

        # === Check for time-resolved axes ===
        if _is_time_resolved(axes):
            print(f"[{args.sid}][{area}] Detected time-resolved axes...")
            result = _qc_time_resolved(cache, axes, time_s, ori, pt_thr, args.align)
            
            qc_dir = (os.path.join(args.out_root, args.align, args.sid, "qc", args.tag)
                      if args.tag else
                      os.path.join(args.out_root, args.align, args.sid, "qc"))
            os.makedirs(qc_dir, exist_ok=True)
            
            pdf = os.path.join(qc_dir, f"qc_axes_{area}.pdf")
            _plot_time_resolved_qc(result, pdf, area)
            _save_time_resolved_json(result, os.path.join(qc_dir, f"qc_axes_{area}.json"))
            
            # Report CV AUC from training (the honest estimate)
            cv_auc = result.get("cv_auc_train", result.get("auc_diagonal"))
            if cv_auc is not None and np.any(np.isfinite(cv_auc)):
                peak_auc = np.nanmax(cv_auc)
                peak_idx = np.nanargmax(cv_auc)
                peak_time = result["time"][peak_idx] * 1000
                print(f"[{args.sid}][{area}] wrote {pdf} (+ .png, .json)   "
                      f"peak CV AUC={peak_auc:.3f} at {peak_time:.0f}ms")
            else:
                print(f"[{args.sid}][{area}] wrote {pdf} (+ .png, .json)")
            continue  # Skip standard QC

        # === Determine normalization ===
        if args.norm == "auto":
            # Use axes meta - pass None to let qc_curves_for_area handle it
            norm = None
            baseline_win = None
        else:
            norm = args.norm
            baseline_win = _parse_range_arg(args.baseline_win)
            
            # Check for mismatch with axes training
            axes_norm = axes.get("meta", {}).get("norm", "global")
            if norm != axes_norm and not args.force_norm_mismatch:
                import warnings
                warnings.warn(
                    f"Normalization mode '{norm}' differs from axes training mode '{axes_norm}'. "
                    f"This may lead to inconsistent results. Use --force_norm_mismatch to override."
                )

        curves = qc_curves_for_area(
            cache=cache, axes=axes, align=args.align,
            time_s=time_s, orientation=ori,
            thr=args.thr, k_bins=args.k, pt_min_ms=pt_thr,
            norm=norm, baseline_win=baseline_win
        )

        qc_dir = (os.path.join(args.out_root, args.align, args.sid, "qc", args.tag)
                  if args.tag else
                  os.path.join(args.out_root, args.align, args.sid, "qc"))
        os.makedirs(qc_dir, exist_ok=True)
        pdf = os.path.join(qc_dir, f"qc_axes_{area}.pdf")
        _plot_curves(curves, pdf, area)
        _save_json(curves, os.path.join(qc_dir, f"qc_axes_{area}.json"))
        print(f"[{args.sid}][{area}] wrote {pdf} (+ .png, .json)   "
              f"lat(ms): C={curves.lat_C_ms}, Sraw={curves.lat_S_raw_ms}, Sinv={curves.lat_S_inv_ms}, T={curves.lat_T_ms}")

if __name__ == "__main__":
    main()
