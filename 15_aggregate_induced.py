#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_aggregate_induced.py

Aggregate induced (evoked-removed) flow results produced by:
  12_induced_fsflow_timesliding.py

Folder layout expected per session:
  results/session/<sid>/<TAG>/induced_flow_<A>to<B>.npz
where <TAG> starts with "induced" (e.g., "induced_k2_win016_p500").

This script:
  • Scans results/session/<sid>/<TAG> across sessions for the SAME <TAG> (pass via --tag)
  • Splits by monkey (M: sid starts '2020'; S: sid starts '2023')
  • Aggregates FORWARD curves:
      - Sliding-integrated forward series (C_fwd_sl/R_fwd_sl)
      - Falls back to raw forward if needed (rare)
  • Interpolates to common time grid; saves mean ± 95% CI plots per pair
  • Writes manifest CSV and band-integral CSV (IC/pIC, IR/pIR) if present

Usage:
  python 15_aggregate_induced.py --tag induced_k2_win016_p500
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SESS_ROOT = BASE / "results" / "session"
OUT_ROOT  = BASE / "results" / "group_induced"

M_AREAS = ["MFEF","MLIP","MSC"]
S_AREAS = ["SFEF","SLIP","SSC"]

def sid_monkey(sid: int) -> str|None:
    s = str(sid)
    if s.startswith("2020"): return "M"
    if s.startswith("2023"): return "S"
    return None

def load_induced_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    rec = {
        "tC": z["tC"] if "tC" in z.files else None,
        "tR": z["tR"] if "tR" in z.files else None,
        "C_series": z["C_fwd_sl"] if "C_fwd_sl" in z.files else z.get("C_fwd", None),
        "R_series": z["R_fwd_sl"] if "R_fwd_sl" in z.files else z.get("R_fwd", None),
        "IC": float(z["IC_fwd"]) if "IC_fwd" in z.files else np.nan,
        "pIC": float(z["pC_fwd"]) if "pC_fwd" in z.files else np.nan,
        "IR": float(z["IR_fwd"]) if "IR_fwd" in z.files else np.nan,
        "pIR": float(z["pR_fwd"]) if "pR_fwd" in z.files else np.nan,
    }
    return rec

def interp_to(x_src, y_src, x_ref):
    if x_src is None or y_src is None or len(x_src) < 2:
        return np.full_like(x_ref, np.nan, dtype=float)
    y = np.array(y_src, float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(x_ref, np.nan, dtype=float)
    return np.interp(x_ref, x_src[mask], y[mask], left=np.nan, right=np.nan)

def mean_ci(y_mat: np.ndarray):
    mu = np.nanmean(y_mat, axis=0)
    n  = np.sum(np.isfinite(y_mat), axis=0).astype(float)
    sd = np.nanstd(y_mat, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(np.maximum(n,1)), out=np.zeros_like(sd), where=(n>0))
    lo = mu - 1.96*sem
    hi = mu + 1.96*sem
    return mu, lo, hi, n

def area_pairs(monkey: str):
    areas = M_AREAS if monkey=="M" else S_AREAS
    return [(a,b) for a in areas for b in areas if a!=b]

def plot_pair(monkey, pair, x_ref, C_mat, R_mat, out_png):
    fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)
    if C_mat is not None and C_mat.size:
        C_mu, C_lo, C_hi, nC = mean_ci(C_mat)
        ax = axes[0]
        ax.fill_between(x_ref, C_lo, C_hi, color="grey", alpha=0.2, label="95% CI (forward)")
        ax.plot(x_ref, C_mu, lw=2, label="Mean forward (C)")
        ax.axvline(0.0, color="k", ls=":", lw=1); ax.set_ylabel("Integrated GC (C)"); ax.legend(frameon=False)
    else:
        axes[0].text(0.5,0.5,"No C data", ha="center", va="center"); axes[0].set_ylabel("Integrated GC (C)")
    if R_mat is not None and R_mat.size:
        R_mu, R_lo, R_hi, nR = mean_ci(R_mat)
        ax = axes[1]
        ax.fill_between(x_ref, R_lo, R_hi, color="grey", alpha=0.2, label="95% CI (forward)")
        ax.plot(x_ref, R_mu, lw=2, label="Mean forward (R)")
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_xlabel("time (s) from cat_stim_on"); ax.set_ylabel("Integrated GC (R)"); ax.legend(frameon=False)
    else:
        axes[1].text(0.5,0.5,"No R data", ha="center", va="center")
        axes[1].set_xlabel("time (s) from cat_stim_on"); axes[1].set_ylabel("Integrated GC (R)")
    fig.suptitle(f"Induced aggregate — Monkey {monkey}: {pair[0]}→{pair[1]}")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Subfolder name starting with 'induced' (e.g., induced_k2_win016_p500)")
    ap.add_argument("--out_dir", default=str(OUT_ROOT), help="Output root directory")
    ap.add_argument("--save_csv", action="store_true", help="Write per-pair CSV summaries")
    args = ap.parse_args()

    tag = args.tag
    OUT = Path(args.out_dir) / tag
    (OUT/"monkey_M").mkdir(parents=True, exist_ok=True)
    (OUT/"monkey_S").mkdir(parents=True, exist_ok=True)

    buckets = {"M": {}, "S": {}}
    manifests = {"M": [], "S": []}
    scalars   = {"M": [], "S": []}

    # Walk sessions
    for sdir in sorted(SESS_ROOT.glob("*")):
        if not sdir.is_dir() or not sdir.name.isdigit(): continue
        sid = int(sdir.name)
        mk = sid_monkey(sid)
        if mk is None: continue
        sub = sdir / tag
        if not sub.exists(): continue

        for npz in sorted(sub.glob("induced_flow_*to*.npz")):
            name = npz.stem[len("induced_flow_"):]
            if "to" not in name: continue
            A, B = name.split("to")
            # enforce same-monkey areas
            if mk=="M" and (not A.startswith("M") or not B.startswith("M")): continue
            if mk=="S" and (not A.startswith("S") or not B.startswith("S")): continue

            rec = load_induced_npz(npz)
            buckets[mk].setdefault((A,B), []).append((sid, rec["tC"], rec["C_series"], rec["tR"], rec["R_series"]))
            manifests[mk].append({"sid": sid, "pair": f"{A}->{B}", "file": npz.name})
            scalars[mk].append({"sid": sid, "pair": f"{A}->{B}", "IC": rec["IC"], "pIC": rec["pIC"], "IR": rec["IR"], "pIR": rec["pIR"]})

    # Write manifests & scalars
    for mk in ("M","S"):
        mdir = OUT / f"monkey_{mk}"
        if manifests[mk]:
            pd.DataFrame(manifests[mk]).sort_values(["pair","sid"]).to_csv(mdir/"manifest.csv", index=False)
        if args.save_csv and scalars[mk]:
            pd.DataFrame(scalars[mk]).sort_values(["pair","sid"]).to_csv(mdir/"band_integrals.csv", index=False)

    # Aggregate & plot
    for mk in ("M","S"):
        mdir = OUT / f"monkey_{mk}"
        for pair in area_pairs(mk):
            recs = buckets[mk].get(pair, [])
            if not recs: continue
            # reference time axis
            x_ref = None
            for _, tC, Cs, tR, Rs in recs:
                if tC is not None and len(tC)>1:
                    x_ref = tC; break
                if tR is not None and len(tR)>1:
                    x_ref = tR; break
            if x_ref is None: continue

            C_list, R_list = [], []
            for sid, tC, Cser, tR, Rser in recs:
                if Cser is not None and tC is not None and len(tC)>1:
                    C_list.append(interp_to(tC, Cser, x_ref))
                if Rser is not None and tR is not None and len(tR)>1:
                    R_list.append(interp_to(tR, Rser, x_ref))
            C_mat = np.vstack(C_list) if C_list else None
            R_mat = np.vstack(R_list) if R_list else None

            out_png = mdir / f"agg_{pair[0]}to{pair[1]}.png"
            plot_pair(mk, pair, x_ref, C_mat, R_mat, out_png)

            if args.save_csv:
                if C_mat is not None:
                    C_mu, C_lo, C_hi, nC = mean_ci(C_mat)
                    pd.DataFrame({"t": x_ref, "C_mu": C_mu, "C_lo": C_lo, "C_hi": C_hi, "nC": nC}).to_csv(
                        mdir / f"agg_C_{pair[0]}to{pair[1]}.csv", index=False)
                if R_mat is not None:
                    R_mu, R_lo, R_hi, nR = mean_ci(R_mat)
                    pd.DataFrame({"t": x_ref, "R_mu": R_mu, "R_lo": R_lo, "R_hi": R_hi, "nR": nR}).to_csv(
                        mdir / f"agg_R_{pair[0]}to{pair[1]}.csv", index=False)

    print(f"[done] Induced aggregation complete → {OUT}")

if __name__ == "__main__":
    main()
