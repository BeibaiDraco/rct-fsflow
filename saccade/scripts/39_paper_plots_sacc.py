#!/usr/bin/env python
"""python scripts/39_paper_plots_sacc.py \
  --tag sacc_v1 \
  --orientation vertical \
  --run_all \
  --out_root results_sacc"""
import argparse, os, numpy as np, matplotlib.pyplot as plt

def load_group(base, mk, pair):
    d1 = np.load(os.path.join(base, f"{mk}_{pair}", "group_overlay.npz"))
    d2 = np.load(os.path.join(base, f"{mk}_{pair}", "group_pairdiff.npz"))
    return d1, d2

def panel_two(base, mk, pair, outpdf, alpha=0.05):
    ol, pd = load_group(base, mk, pair)
    t  = ol["time"]; tms = t*1000
    td = pd["time"]; tdms = td*1000

    fig = plt.figure(figsize=(7.2,5.2))

    # ---------------- Panel A: pair-diff ----------------
    ax1 = fig.add_subplot(2,1,1)
    ax1.axhline(0, ls="--", c="k", lw=0.8); ax1.axvline(0, ls="--", c="k", lw=0.8)
    md = pd["mean_DIFF"]; sd = pd["sem_DIFF"]
    ax1.plot(tdms, md, color="C3", lw=2, label="(A→B − B→A)")
    ax1.fill_between(tdms, md-sd, md+sd, color="C3", alpha=0.15, linewidth=0)

    # --- significance dots (p_DIFF < alpha) ---
    if "p_DIFF" in pd:
        p = pd["p_DIFF"]
        if p.shape == md.shape:
            sigmask = np.isfinite(p) & (p < alpha)
            if np.any(sigmask):
                # place dots slightly below the lowest value for visibility
                rng = float(np.nanmax(md) - np.nanmin(md))
                if not np.isfinite(rng) or rng <= 0: rng = 1e-6
                ybar = float(np.nanmin(md) - 0.05*rng)
                ax1.plot(tdms[sigmask], np.full(sigmask.sum(), ybar), ".", ms=6, color="k")

    ax1.set_ylabel("ΔLL bits"); ax1.set_title(f"{mk} {pair} — Pair-diff")

    # ---------------- Panel B: overlay ----------------
    ax2 = fig.add_subplot(2,1,2)
    ax2.axhline(0, ls="--", c="k", lw=0.8); ax2.axvline(0, ls="--", c="k", lw=0.8)
    mAB, sAB = ol["mean_AtoB"], ol["sem_AtoB"]
    mBA, sBA = ol["mean_BtoA"], ol["sem_BtoA"]
    ax2.plot(tms, mAB, lw=2, color="C0", label="A→B")
    ax2.fill_between(tms, mAB-sAB, mAB+sAB, color="C0", alpha=0.15, linewidth=0)
    ax2.plot(tms, mBA, lw=2, color="C1", label="B→A")
    ax2.fill_between(tms, mBA-sBA, mBA+sBA, color="C1", alpha=0.15, linewidth=0)
    ax2.set_xlabel("Time from saccade onset (ms)"); ax2.set_ylabel("ΔLL bits"); ax2.legend()

    fig.tight_layout()
    # save PDF + PNG
    fig.savefig(outpdf)
    outpng = os.path.splitext(outpdf)[0] + ".png"
    fig.savefig(outpng, dpi=300)
    plt.close(fig)

def list_group_pairs(base):
    pairs = []
    for d in os.listdir(base):
        if "_" not in d: continue
        mk, pair = d.split("_", 1)
        pdir = os.path.join(base, d)
        if os.path.isdir(pdir):
            pairs.append((mk, pair))
    return sorted(set(pairs))

def main():
    ap = argparse.ArgumentParser(description="Two-panel paper figure for S-flow group results.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical")
    ap.add_argument("--pair", default=None)
    ap.add_argument("--monkey", choices=["M","S"], default=None)
    ap.add_argument("--run_all", action="store_true", help="Render for ALL pairs & monkeys")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for dots on Panel A")
    ap.add_argument("--out_root", default="results_sacc")
    args = ap.parse_args()

    base = os.path.join(args.out_root, "group_sacc", args.tag, args.orientation)
    os.makedirs(base, exist_ok=True)

    if args.run_all:
        for mk, pair in list_group_pairs(base):
            outpdf = os.path.join(base, f"{mk}_{pair}", f"paper_S_{mk}_{pair}.pdf")
            panel_two(base, mk, pair, outpdf, alpha=args.alpha)
            print(f"[paper] wrote {outpdf} and PNG")
        return

    if not args.pair or not args.monkey:
        raise SystemExit("Provide --pair and --monkey, or use --run_all")

    outpdf = os.path.join(base, f"{args.monkey}_{args.pair}", f"paper_S_{args.monkey}_{args.pair}.pdf")
    panel_two(base, args.monkey, args.pair, outpdf, alpha=args.alpha)
    print(f"[paper] wrote {outpdf} and PNG")

if __name__ == "__main__":
    main()
