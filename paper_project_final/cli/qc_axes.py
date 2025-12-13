#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, numpy as np
from glob import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paperflow.qc import qc_curves_for_area

def _load_cache(out_root, align, sid, area):
    p = os.path.join(out_root, align, sid, "caches", f"area_{area}.npz")
    d = np.load(p, allow_pickle=True)
    meta = json.loads(d["meta"].item()) if "meta" in d else {}
    cache = {k: d[k] for k in d.files}
    cache["meta"] = meta
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

    plt.xlabel("Time (ms)")
    plt.ylabel("AUC / Accuracy")
    plt.title(f"{area} — QC ({curves.meta.get('align','?')}, ori={curves.meta.get('orientation')}, PT≥{curves.meta.get('pt_min_ms')})")
    plt.legend(loc="lower right", ncol=2, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf)
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="QC curves for trained axes (per area).")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--tag", default=None,
                    help="Optional tag; if set, read axes from axes/<tag>/ and write QC to qc/<tag>/")
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pt_min_ms_sacc", type=float, default=200.0)
    ap.add_argument("--pt_min_ms_stim", type=float, default=200.0)
    ap.add_argument("--no_pt_filter", action="store_true")
    args = ap.parse_args()

    areas = args.areas or _areas(args.out_root, args.align, args.sid)
    if not areas:
        raise SystemExit(f"No caches found under {args.out_root}/{args.align}/{args.sid}/caches")

    any_cache = _load_cache(args.out_root, args.align, args.sid, areas[0])
    time_s = any_cache["time"].astype(float)

    ori = None if args.orientation == "pooled" else args.orientation
    pt_thr = None if args.no_pt_filter else (args.pt_min_ms_sacc if args.align=="sacc" else args.pt_min_ms_stim)

    for area in areas:
        cache = _load_cache(args.out_root, args.align, args.sid, area)
        axes  = _load_axes(args.out_root, args.align, args.sid, area, tag=args.tag)

        curves = qc_curves_for_area(
            cache=cache, axes=axes, align=args.align,
            time_s=time_s, orientation=ori,
            thr=args.thr, k_bins=args.k, pt_min_ms=pt_thr
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
