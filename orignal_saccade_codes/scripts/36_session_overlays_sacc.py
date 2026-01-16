#!/usr/bin/env python
import argparse, os, json, numpy as np
from glob import glob
from saccflow.plotting import overlay_plot

def main():
    ap = argparse.ArgumentParser(description="Per-session overlays for S-flow (A↔B) with null means.")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--tag", required=True, help="Subfolder under results_sacc/<sid>/saccflow/<tag>/")
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical")
    ap.add_argument("--smooth_bins", type=int, default=3)
    ap.add_argument("--shade_null", action="store_true", help="Fill ±1σ around null means")
    ap.add_argument("--out_root", default="results_sacc")
    args = ap.parse_args()

    flow_dir = os.path.join(args.out_root, args.sid, "saccflow", args.tag)
    files = sorted(glob(os.path.join(flow_dir, "induced_flow_S_*to*.npz")))
    if not files:
        raise RuntimeError(f"No flow NPZs in {flow_dir}")

    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(d["meta"].item())
        if meta.get("orientation","vertical") != args.orientation:
            continue
        time = d["time"].astype(float)
        yAB  = d["bits_AtoB"].astype(float)
        yBA  = d["bits_BtoA"].astype(float)
        pAB  = d["p_AtoB"].astype(float) if "p_AtoB" in d else None
        pBA  = d["p_BtoA"].astype(float) if "p_BtoA" in d else None

        # null stats (present in new files; fallback-safe if missing)
        nABm = d["null_mean_AtoB"].astype(float) if "null_mean_AtoB" in d else None
        nABs = d["null_std_AtoB"].astype(float)  if "null_std_AtoB"  in d else None
        nBAm = d["null_mean_BtoA"].astype(float) if "null_mean_BtoA" in d else None
        nBAs = d["null_std_BtoA"].astype(float)  if "null_std_BtoA"  in d else None

        pair = os.path.basename(f).replace("induced_flow_S_","").replace(".npz","")
        A, B = pair.split("to")
        out = os.path.join(flow_dir, f"overlay_S_{A}to{B}.png")
        title = f"{args.sid}  {A}→{B}  ({args.orientation})"
        overlay_plot(time, yAB, pAB, yBA, pBA, nABm, nABs, nBAm, nBAs,
                     smooth_bins=args.smooth_bins, out_path=out, title=title, shade_null=args.shade_null)
        print(f"[{args.sid}][{A}->{B}] wrote {out}")

if __name__ == "__main__":
    main()
