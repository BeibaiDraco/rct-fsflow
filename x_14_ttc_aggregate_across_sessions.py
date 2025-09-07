#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, glob, json, numpy as np, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_session", default="results/session")
    ap.add_argument("--out_root", default="results/ttc")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stats_script", default="12_ttc_stats.py")
    ap.add_argument("--fef", default="MFEF")
    ap.add_argument("--lip", default="MLIP")
    ap.add_argument("--sc", default="MSC")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.root_session, "*", f"ttc_{args.tag}.npz")))
    if not files:
        print("[err] no session npz found for tag", args.tag, file=sys.stderr); sys.exit(2)

    out_csv = os.path.join(args.out_root, f"ttc_summary_{args.tag}.csv")
    out_json = os.path.join(args.out_root, f"ttc_summary_{args.tag}.json")
    cmd = [sys.executable, args.stats_script, "--files", *files,
           "--out_csv", out_csv, "--out_json", out_json, "--fef", args.fef, "--lip", args.lip, "--sc", args.sc]
    print("[run stats]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(out_json, "r") as f:
        rows = json.load(f)

    pairs_cat = [("FEF<LIP", "Frac_cat_FEF<LIP", "MedLead_ms_cat_FEF-LIP"),
                 ("FEF<SC", "Frac_cat_FEF<SC", "MedLead_ms_cat_FEF-SC"),
                 ("SC<LIP", "Frac_cat_SC<LIP", "MedLead_ms_cat_SC-LIP")]
    pairs_dir = [("LIP<FEF", "Frac_dir_LIP<FEF", "MedLead_ms_dir_LIP-FEF"),
                 ("LIP<SC", "Frac_dir_LIP<SC", "MedLead_ms_dir_LIP-SC"),
                 ("FEF<SC", "Frac_dir_FEF<SC", "MedLead_ms_dir_FEF-SC")]

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    labels = [p[0] for p in pairs_cat]
    fracs = [np.nanmean([r.get(p[1], np.nan) for r in rows]) for p in pairs_cat]
    meds  = [np.nanmedian([r.get(p[2], np.nan) for r in rows]) for p in pairs_cat]
    ax1.bar(np.arange(len(labels)), fracs)
    ax1.set_xticks(np.arange(len(labels))); ax1.set_xticklabels(labels)
    ax1.set_ylim(0,1); ax1.set_ylabel("Lead fraction (Category)")
    for i, m in enumerate(meds):
        ax1.text(i, fracs[i]+0.02, f"medΔ={m:.0f} ms", ha="center", fontsize=9)
    ax1.set_title(f"Category lead across sessions (tag={args.tag})")

    ax2 = fig.add_subplot(1,2,2)
    labels = [p[0] for p in pairs_dir]
    fracs = [np.nanmean([r.get(p[1], np.nan) for r in rows]) for p in pairs_dir]
    meds  = [np.nanmedian([r.get(p[2], np.nan) for r in rows]) for p in pairs_dir]
    ax2.bar(np.arange(len(labels)), fracs)
    ax2.set_xticks(np.arange(len(labels))); ax2.set_xticklabels(labels)
    ax2.set_ylim(0,1); ax2.set_ylabel("Lead fraction (Direction)")
    for i, m in enumerate(meds):
        ax2.text(i, fracs[i]+0.02, f"medΔ={m:.0f} ms", ha="center", fontsize=9)
    ax2.set_title(f"Direction lead across sessions (tag={args.tag})")

    fig.tight_layout()
    out_png = os.path.join(args.out_root, f"ttc_summary_{args.tag}.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("[ok] wrote", out_csv, out_json, out_png)

if __name__ == "__main__":
    main()
