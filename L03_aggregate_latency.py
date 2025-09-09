#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L03_aggregate_latency.py

Aggregate per-trial latency outputs across sessions and create pooled plots per pair, split by monkey:
  • Pooled ΔT histograms (C and R panels)
  • Pooled 2-D latency scatter (C and R panels; A on x, B on y; y=x line)
  • Save pooled arrays to NPZ for post-hoc analysis

Reads per-session outputs from:
  results/session/<sid>/<out_tag>/latency_C_<AREA>.npy
  results/session/<sid>/<out_tag>/latency_R_<AREA>.npy

Monkey split:
  M: session id starts with '2020' (areas 'M...')
  S: session id starts with '2023' (areas 'S...')
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID = {"MFEF","MLIP","MSC","SFEF","SLIP","SSC"}

def detect_sessions(sess_root: Path) -> list[int]:
    s=[]
    for p in sorted(sess_root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            s.append(int(p.name))
    return s

def find_areas(sess_dir: Path, out_tag: str) -> list[str]:
    base = sess_dir/out_tag
    if not base.exists(): return []
    st=set()
    for f in base.glob("latency_C_*.npy"):
        st.add(f.stem.split("latency_C_")[1])
    for f in base.glob("latency_R_*.npy"):
        st.add(f.stem.split("latency_R_")[1])
    return sorted(a for a in st if a in VALID)

def load_pair(sess_dir: Path, out_tag: str, A: str, B: str):
    base = sess_dir/out_tag
    fCA=base/f"latency_C_{A}.npy"; fCB=base/f"latency_C_{B}.npy"
    fRA=base/f"latency_R_{A}.npy"; fRB=base/f"latency_R_{B}.npy"
    if not (fCA.exists() and fCB.exists() and fRA.exists() and fRB.exists()):
        return None
    return np.load(fCA), np.load(fCB), np.load(fRA), np.load(fRB)

def pair_list(monkey: str):
    ASET = ["MFEF","MLIP","MSC"] if monkey=="M" else ["SFEF","SLIP","SSC"]
    return [(a,b) for a in ASET for b in ASET if a!=b]

def pooled_hist(pair, dC, dR, out_png, title):
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    for ax, (lbl, arr) in zip(axes, [("ΔT_C (B - A)", dC), ("ΔT_R (B - A)", dR)]):
        ok = np.isfinite(arr)
        if ok.sum() < 20:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center"); ax.set_axis_off(); continue
        ax.hist(arr[ok], bins=60, color="steelblue", alpha=0.85)
        med = float(np.nanmedian(arr)); mean = float(np.nanmean(arr)); prop = float(np.nanmean(arr > 0))
        ax.axvline(0, color="k", lw=1)
        ax.axvline(med, color="green", lw=2, ls="--", label=f"median={med:.3f}s")
        ax.axvline(mean, color="red", lw=2, ls="-", label=f"mean={mean:.3f}s")
        ax.set_title(lbl); ax.set_xlabel("seconds"); ax.legend(frameon=False)
        ax.text(0.02,0.95,f"P(B>A)={prop:.2f}  n={ok.sum()}", transform=ax.transAxes, va="top")
    fig.suptitle(f"{title} — {pair[0]}→{pair[1]}")
    fig.tight_layout(rect=[0,0.03,1,0.95]); fig.savefig(out_png, dpi=150); plt.close(fig)

def pooled_scatter(pair, tC_A, tC_B, tR_A, tR_B, out_png, title):
    mC = np.isfinite(tC_A) & np.isfinite(tC_B)
    mR = np.isfinite(tR_A) & np.isfinite(tR_B)
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    panels = [("T_C  A vs B", tC_A[mC], tC_B[mC]), ("T_R  A vs B", tR_A[mR], tR_B[mR])]
    for ax, (ttl,x,y) in zip(axes, panels):
        if x.size < 20:
            ax.text(0.5,0.5,"Insufficient data", ha="center", va="center"); ax.set_axis_off(); continue
        lim_min = float(np.nanmin(np.concatenate([x,y])))
        lim_max = float(np.nanmax(np.concatenate([x,y])))
        pad = 0.05*(lim_max - lim_min + 1e-6)
        ax.scatter(x, y, s=10, alpha=0.6, color="steelblue", edgecolor="none")
        ax.plot([lim_min-pad, lim_max+pad], [lim_min-pad, lim_max+pad], 'k-', lw=1)
        
        # Plot mean with red marker
        mean_x = float(np.nanmean(x))
        mean_y = float(np.nanmean(y))
        ax.scatter(mean_x, mean_y, s=80, color="red", marker="o", edgecolor="darkred", linewidth=2, zorder=5, label="Mean")
        
        ax.set_xlim(lim_min-pad, lim_max+pad); ax.set_ylim(lim_min-pad, lim_max+pad)
        ax.set_aspect('equal','box'); ax.set_title(ttl); ax.set_xlabel(f"{pair[0]} (s)"); ax.set_ylabel(f"{pair[1]} (s)")
        if x.size>=2:
            r = np.corrcoef(x,y)[0,1]; ax.text(0.02,0.95,f"n={x.size}  r={r:.2f}", transform=ax.transAxes, va="top")
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle(f"{title} — {pair[0]}→{pair[1]}")
    fig.tight_layout(rect=[0,0.03,1,0.95]); fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess_root", type=Path, default=Path("results/session"))
    ap.add_argument("--out_tag", type=str, default="latency_auto")
    ap.add_argument("--out_dir", type=Path, default=Path("results/group_latency"))
    args = ap.parse_args()

    sids = detect_sessions(args.sess_root)
    if not sids:
        print("[info] no sessions found"); return

    (args.out_dir/"monkey_M").mkdir(parents=True, exist_ok=True)
    (args.out_dir/"monkey_S").mkdir(parents=True, exist_ok=True)

    for mk in ("M","S"):
        pairs = pair_list(mk)
        base = args.out_dir / ("monkey_M" if mk=="M" else "monkey_S")
        for pair in pairs:
            A,B = pair
            pool_dC=[]; pool_dR=[]; pool_tC_A=[]; pool_tC_B=[]; pool_tR_A=[]; pool_tR_B=[]
            for sid in sids:
                sid_str = str(sid)
                if (mk=="M" and not sid_str.startswith("2020")) or (mk=="S" and not sid_str.startswith("2023")):
                    continue
                sdir = args.sess_root / sid_str
                areas = find_areas(sdir, args.out_tag)
                if A not in areas or B not in areas:
                    continue
                lat = load_pair(sdir, args.out_tag, A, B)
                if lat is None: 
                    continue
                tC_A, tC_B, tR_A, tR_B = lat
                # Exclude trials > 350ms (0.35s) and keep only finite values
                mC = np.isfinite(tC_A) & np.isfinite(tC_B) & (tC_A <= 0.35) & (tC_B <= 0.35)
                mR = np.isfinite(tR_A) & np.isfinite(tR_B) & (tR_A <= 0.35) & (tR_B <= 0.35)
                if mC.sum()>0:
                    pool_dC.append(tC_B[mC] - tC_A[mC])
                    pool_tC_A.append(tC_A[mC]); pool_tC_B.append(tC_B[mC])
                if mR.sum()>0:
                    pool_dR.append(tR_B[mR] - tR_A[mR])
                    pool_tR_A.append(tR_A[mR]); pool_tR_B.append(tR_B[mR])

            # concatenate
            dC = np.concatenate(pool_dC) if pool_dC else np.array([])
            dR = np.concatenate(pool_dR) if pool_dR else np.array([])
            tC_A_all = np.concatenate(pool_tC_A) if pool_tC_A else np.array([])
            tC_B_all = np.concatenate(pool_tC_B) if pool_tC_B else np.array([])
            tR_A_all = np.concatenate(pool_tR_A) if pool_tR_A else np.array([])
            tR_B_all = np.concatenate(pool_tR_B) if pool_tR_B else np.array([])

            # save pooled arrays
            np.savez_compressed(base/f"pooled_{A}to{B}.npz",
                                dC=dC, dR=dR,
                                tC_A=tC_A_all, tC_B=tC_B_all,
                                tR_A=tR_A_all, tR_B=tR_B_all)

            # plots
            pooled_hist(pair, dC, dR, base/f"pooled_hist_{A}to{B}.png",
                        title=f"All sessions ({'Monkey M' if mk=='M' else 'Monkey S'})")
            pooled_scatter(pair, tC_A_all, tC_B_all, tR_A_all, tR_B_all,
                           base/f"pooled_scatter_{A}to{B}.png",
                           title=f"All sessions ({'Monkey M' if mk=='M' else 'Monkey S'})")
    print(f"[done] pooled outputs → {args.out_dir}")

if __name__ == "__main__":
    main()
