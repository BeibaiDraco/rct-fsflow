#!/usr/bin/env python
from __future__ import annotations
import argparse, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paperflow.pseudo import pseudo_decode_direction_within_category

def main():
    ap = argparse.ArgumentParser(description="Pseudopopulation direction decoder (paper-style, within-category).")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim"], default="stim")
    ap.add_argument("--area", required=True)
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical")
    ap.add_argument("--n_pops", type=int, default=200)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=2)
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--max_units", type=int, default=200, help="Subsample to at most this many units (match areas)")
    ap.add_argument("--seed", type=int, default=0)
    # NEW: replicate Methods more closely
    ap.add_argument("--selective_only", action="store_true", help="Use only direction-selective units (approx).")
    ap.add_argument("--select_win", type=str, default="0.02:0.20", help="Selection window (s) for direction selectivity.")
    ap.add_argument("--min_rate_hz", type=float, default=10.0, help="Min rate threshold during selection (Hz).")
    ap.add_argument("--smooth_sigma_ms", type=float, default=20.0, help="Temporal smoothing sigma (ms).")
    ap.add_argument("--match_saccade", choices=["+1","-1","none"], default="none",
                    help="Fix saccade direction within category to +1 or -1; 'none' ignores saccade dir.")
    args = ap.parse_args()

    select_win = tuple(float(x) for x in args.select_win.split(":"))
    matchS = None if args.match_saccade == "none" else (+1 if args.match_saccade == "+1" else -1)

    print("="*80)
    print("Pseudopopulation Direction Decoder")
    print(f"Area: {args.area}, Orientation: {args.orientation}")
    print(f"n_pops={args.n_pops}, n_train={args.n_train}, n_test={args.n_test}, max_units={args.max_units}")
    print(f"Filters: correct, PT>{args.pt_min_ms}ms, orientation={args.orientation}, match_saccade={args.match_saccade}")
    print(f"Selection: selective_only={args.selective_only}, select_win={select_win}s, min_rate_hz={args.min_rate_hz}, smooth_sigma_ms={args.smooth_sigma_ms}")
    print("="*80)

    res = pseudo_decode_direction_within_category(
        out_root=args.out_root, align=args.align, area=args.area,
        orientation=args.orientation,
        n_pops=args.n_pops, n_train=args.n_train, n_test=args.n_test,
        pt_min_ms=args.pt_min_ms, max_units=args.max_units, seed=args.seed,
        selective_only=args.selective_only, select_win=select_win,
        min_rate_hz=args.min_rate_hz, smooth_sigma_ms=args.smooth_sigma_ms,
        match_saccade=matchS
    )

    out_dir = os.path.join(args.out_root, "pseudo", args.align, args.area)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"pseudo_R_{args.area}.npz"),
                        time=res["time"], acc_mean=res["acc_mean"], acc_std=res["acc_std"], meta=res["meta"])

    tms = res["time"]*1000.0
    plt.figure(figsize=(7.2,3.2))
    plt.axhline(1/3, ls="--", c="k", lw=0.8)
    plt.axvline(0,   ls="--", c="k", lw=0.8)
    plt.plot(tms, res["acc_mean"], lw=2, color="C4")
    plt.fill_between(tms, res["acc_mean"]-res["acc_std"], res["acc_mean"]+res["acc_std"],
                     color="C4", alpha=0.15, linewidth=0)
    plt.xlabel("Time from stimulus onset (ms)")
    plt.ylabel("Direction CV accuracy")
    plt.title(f"{args.area} — pseudopop direction (within-category, {args.orientation})")
    plt.tight_layout()
    pdff = os.path.join(out_dir, f"pseudo_R_{args.area}.pdf")
    plt.savefig(pdff); plt.savefig(pdff.replace(".pdf", ".png"), dpi=300)
    plt.close()
    print(f"[ok] wrote {pdff} (+ .png, .npz)")

if __name__ == "__main__":
    main()
