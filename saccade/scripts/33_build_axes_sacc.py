#!/usr/bin/env python
import argparse, os, json, numpy as np
from saccflow.io import load_manifest, list_areas_from_manifest, load_area_cache
from saccflow.axes import build_axes_for_area, auc_from_axis_projection

def _infer_areas(root: str, sid: str) -> list:
    man = load_manifest(root)
    areas = list_areas_from_manifest(man, sid)
    if not areas:
        cdir = os.path.join("results_sacc", sid, "caches")
        if os.path.isdir(cdir):
            areas = sorted([p.split("_",1)[1].replace(".npz","") for p in os.listdir(cdir) if p.startswith("area_")])
    return areas

def main():
    ap = argparse.ArgumentParser(description="Train sC and sS axes per area (saccade-aligned).")
    ap.add_argument("--root", required=True, help="RCT_02 root (to infer areas if needed)")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--areas", nargs="*", default=None)
    ap.add_argument("--trainC", type=str, default="-0.25:-0.10", help="C window (s), e.g. -0.25:-0.10")
    ap.add_argument("--trainS", type=str, default="-0.20:-0.05", help="S window (s), e.g. -0.20:-0.05")
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical",
                    help="Use only this target configuration (matches your original pipeline).")
    ap.add_argument("--no_invariant", action="store_true", help="If set, do NOT orthogonalize sS to sC.")
    ap.add_argument("--out_root", default="results_sacc")
    args = ap.parse_args()

    areas = args.areas or _infer_areas(args.root, args.sid)
    if not areas:
        raise RuntimeError("No areas found")
    tC = tuple(float(x) for x in args.trainC.split(":"))
    tS = tuple(float(x) for x in args.trainS.split(":"))

    out_dir = os.path.join(args.out_root, args.sid)
    os.makedirs(out_dir, exist_ok=True)

    summary = {"sid": args.sid, "trainC": tC, "trainS": tS,
               "orientation": args.orientation, "areas": {}}

    for area in areas:
        cache = load_area_cache(args.out_root, args.sid, area)
        pack = build_axes_for_area(
            cache, win_C=tC, win_S=tS,
            correct_only=True,
            orientation=args.orientation,
            make_invariant=(not args.no_invariant)
        )

        # Leak tests (use the same orientation subset)
        Z, time = cache["Z"].astype(np.float64), cache["time"].astype(np.float64)
        C, S = cache["C"], cache["S"]
        ori = cache["orientation"].astype(str)
        ic = cache["is_correct"]
        keep = (~np.isnan(C)) & (~np.isnan(S)) & (ori == args.orientation) & (ic if ic is not None else True)
        Zk, Ck, Sk = Z[keep], C[keep], S[keep]
        mC = (time >= tC[0]) & (time <= tC[1])
        mS = (time >= tS[0]) & (time <= tS[1])

        sC_vec   = pack["sC"].vec        if pack["sC"]      else None
        sS_raw_v = pack["sS_raw"].vec    if pack["sS_raw"]  else None
        sS_inv_v = pack["sS_inv"].vec    if pack["sS_inv"]  else None

        auc_S_from_sC    = auc_from_axis_projection(Zk, sC_vec,    Sk, mS)
        auc_C_from_sSraw = auc_from_axis_projection(Zk, sS_raw_v,  Ck, mC)
        auc_C_from_sSinv = auc_from_axis_projection(Zk, sS_inv_v,  Ck, mC) if sS_inv_v is not None else np.nan

        # Save per-area axes (single file per area; includes both variants)
        axes_npz = os.path.join(out_dir, f"axes_{area}.npz")
        np.savez_compressed(
            axes_npz,
            sC=(sC_vec if sC_vec is not None else np.array([])),
            sS_raw=(sS_raw_v if sS_raw_v is not None else np.array([])),
            sS_inv=(sS_inv_v if sS_inv_v is not None else np.array([])),
            meta=json.dumps({
                "sid": args.sid,
                "area": area,
                "trainC": tC,
                "trainS": tS,
                "orientation": args.orientation,
                "correct_only": True,
                "make_invariant": (not args.no_invariant),
                "sC_auc_mean": (pack["sC"].auc_mean if pack["sC"] else None),
                "sC_auc_std":  (pack["sC"].auc_std  if pack["sC"] else None),
                "sSraw_auc_mean": (pack["sS_raw"].auc_mean if pack["sS_raw"] else None),
                "sSraw_auc_std":  (pack["sS_raw"].auc_std  if pack["sS_raw"] else None),
                "cos_sSraw_sC":   (pack["cos_sSraw_sC"]),
                "auc_S_from_sC":    auc_S_from_sC,
                "auc_C_from_sSraw": auc_C_from_sSraw,
                "auc_C_from_sSinv": auc_C_from_sSinv
            })
        )
        print(f"[{args.sid}][{area}] wrote {axes_npz}")
        summary["areas"][area] = {
            "sC_auc_mean": pack["sC"].auc_mean if pack["sC"] else None,
            "sSraw_auc_mean": pack["sS_raw"].auc_mean if pack["sS_raw"] else None,
            "cos_sSraw_sC":   pack["cos_sSraw_sC"],
            "auc_C_from_sSraw": float(auc_C_from_sSraw) if np.isfinite(auc_C_from_sSraw) else None,
            "auc_C_from_sSinv": float(auc_C_from_sSinv) if np.isfinite(auc_C_from_sSinv) else None
        }

    with open(os.path.join(out_dir, "axes_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {os.path.join(out_dir,'axes_summary.json')}")

if __name__ == "__main__":
    main()
