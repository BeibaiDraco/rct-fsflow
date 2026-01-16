#!/usr/bin/env python
import argparse, os, json, numpy as np, matplotlib.pyplot as plt
from saccflow.io import load_area_cache
from numpy.typing import NDArray

def _first_threshold_crossing(curve: NDArray[np.float64], thr: float, k: int) -> int:
    consec = 0
    for i, v in enumerate(curve):
        consec = consec + 1 if v > thr else 0
        if consec >= k:
            return int(i - k + 1)
    return -1

def _load_axes(path: str):
    d = np.load(path, allow_pickle=True)
    meta = {}
    if "meta" in d:
        m = d["meta"]
        if isinstance(m, np.ndarray) and m.dtype == object:
            m = m.item()
        try:
            meta = json.loads(m)
        except Exception:
            meta = {}
    sC     = d["sC"]     if "sC"     in d else np.array([])
    sS_raw = d["sS_raw"] if "sS_raw" in d else np.array([])
    sS_inv = d["sS_inv"] if "sS_inv" in d else np.array([])
    return sC.astype(np.float64), sS_raw.astype(np.float64), sS_inv.astype(np.float64), meta

def _auc_from_proj(scores: NDArray[np.float64], y_pm1: NDArray[np.float64]) -> float:
    y = (y_pm1 > 0).astype(int)
    if np.unique(y).size < 2:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos = (y == 1)
    n1 = np.sum(pos); n0 = np.sum(~pos)
    if n1 == 0 or n0 == 0:
        return np.nan
    u = np.sum(ranks[pos]) - n1*(n1-1)/2
    return float(u / (n1*n0))

def main():
    ap = argparse.ArgumentParser(description="QC: AUC curves & latencies for sC and sS (raw & invariant).")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical",
                    help="Match the configuration used for training.")
    ap.add_argument("--out_root", default="results_sacc")
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    axes_dir = os.path.join(args.out_root, args.sid)
    if args.areas is None:
        args.areas = sorted([p.replace("axes_","").replace(".npz","")
                             for p in os.listdir(axes_dir) if p.startswith("axes_") and p.endswith(".npz")])

    for area in args.areas:
        cache = load_area_cache(args.out_root, args.sid, area)
        Z  = cache["Z"].astype(np.float64)
        tm = cache["time"].astype(np.float64)
        C  = cache["C"]; S = cache["S"]
        ori = cache["orientation"].astype(str)
        ic = cache["is_correct"]

        keep = (~np.isnan(C)) & (~np.isnan(S)) & (ori == args.orientation) & (ic if ic is not None else True)
        Z, C, S = Z[keep], C[keep], S[keep]

        sC, sS_raw, sS_inv, meta = _load_axes(os.path.join(axes_dir, f"axes_{area}.npz"))

        curves = {}
        lats   = {}

        for name, axis_vec, ylab in [("C", sC, C), ("S_raw", sS_raw, S), ("S_inv", sS_inv, S)]:
            if axis_vec.size == 0:
                curves[name] = np.full(tm.shape, np.nan)
                lats[name] = np.nan
                continue
            auc_t = np.zeros_like(tm, dtype=float)
            for bi in range(len(tm)):
                s = Z[:, bi, :] @ axis_vec
                auc_t[bi] = _auc_from_proj(s, ylab)
            curves[name] = auc_t
            idx = _first_threshold_crossing(auc_t, args.thr, args.k)
            lats[name] = (tm[idx]*1000.0) if idx >= 0 else np.nan

        out_fig = os.path.join(axes_dir, f"qc_axes_{area}.png")
        plt.figure(figsize=(6.5, 3.2))
        plt.axhline(0.5, ls="--", lw=1, color="k")
        plt.axvline(0.0, ls="--", lw=1, color="k")
        if np.isfinite(curves["C"]).any():
            plt.plot(tm*1000, curves["C"], label="AUC(C)", lw=2)
        if np.isfinite(curves["S_inv"]).any():
            plt.plot(tm*1000, curves["S_inv"], label="AUC(S inv)", lw=2)
        if np.isfinite(curves["S_raw"]).any():
            plt.plot(tm*1000, curves["S_raw"], label="AUC(S raw)", lw=1.5, ls="--")
        plt.xlabel("Time from Saccade Onset (ms)")
        plt.ylabel("AUC")
        plt.ylim(0.35, 1.01)
        plt.xlim(tm[0]*1000, tm[-1]*1000)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_fig)
        plt.close()

        lat_json = {
            "sid": args.sid, "area": area, "orientation": args.orientation,
            "thr": args.thr, "k": args.k,
            "latency_ms_C": float(lats["C"]) if np.isfinite(lats["C"]) else None,
            "latency_ms_S_inv": float(lats["S_inv"]) if np.isfinite(lats["S_inv"]) else None,
            "latency_ms_S_raw": float(lats["S_raw"]) if np.isfinite(lats["S_raw"]) else None
        }
        with open(os.path.join(axes_dir, f"qc_axes_{area}.json"), "w") as f:
            json.dump(lat_json, f, indent=2)
        print(f"[{args.sid}][{area}] wrote {out_fig} and qc_axes_{area}.json")

if __name__ == "__main__":
    main()
