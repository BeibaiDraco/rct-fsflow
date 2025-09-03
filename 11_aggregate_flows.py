#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_aggregate_flows.py  (verbose aggregator)

Walks results/session/<sid>/flow_timeseries_<A>to<B>.npz files,
groups by monkey (M: MFEF/MLIP/MSC; S: SFEF/SLIP/SSC),
and builds session-averaged curves (mean ± 95% CI) for each ordered pair.

Also prints lots of intermediate info and writes a manifest CSV per monkey.

Usage:
  python 11_aggregate_flows.py
  python 11_aggregate_flows.py --save_csv
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SESS_ROOT = BASE / "results" / "session"
OUT_ROOT  = BASE / "results" / "group"

M_SET = ["MFEF", "MLIP", "MSC"]
S_SET = ["SFEF", "SLIP", "SSC"]

def log(msg: str):
    print(msg, flush=True)

def detect_sessions() -> list[int]:
    if not SESS_ROOT.exists():
        return []
    sids = []
    for p in sorted(SESS_ROOT.iterdir()):
        if p.is_dir() and p.name.isdigit():
            sids.append(int(p.name))
    return sids

def list_pair_files_for_session(sid: int) -> list[Path]:
    sdir = SESS_ROOT / str(sid)
    return sorted(sdir.glob("flow_timeseries_*to*.npz"))

def area_monkey(area: str) -> str | None:
    if area.startswith("M"): return "M"
    if area.startswith("S"): return "S"
    return None

def load_pair_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    # Category
    tC   = z.get("tC", None)
    C_f  = z.get("C_fwd", None)
    C_r  = z.get("C_rev", None)
    # Direction
    tR   = z.get("tR", None)
    R_f  = z.get("R_fwd", None)
    R_r  = z.get("R_rev", None)
    return dict(tC=tC, C_fwd=C_f, C_rev=C_r, tR=tR, R_fwd=R_f, R_rev=R_r, meta=meta)

def interp_to_grid(x_src: np.ndarray, y_src: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    if x_src is None or y_src is None or len(x_src) == 0:
        return np.full_like(x_ref, np.nan, dtype=float)
    ys = np.asarray(y_src, dtype=float).copy()
    ys[~np.isfinite(ys)] = np.nan
    valid = ~np.isnan(ys)
    if valid.sum() < 2:
        val = np.nanmean(ys)
        return np.full_like(x_ref, val if np.isfinite(val) else np.nan, dtype=float)
    return np.interp(x_ref, x_src[valid], ys[valid], left=np.nan, right=np.nan)

def stack_on_grid(records: list[dict], feature: str):
    if feature == "C":
        xs = [r["tC"] for r in records]
        ys_f = [r["C_fwd"] for r in records]
        ys_r = [r["C_rev"] for r in records]
    else:
        xs = [r["tR"] for r in records]
        ys_f = [r["R_fwd"] for r in records]
        ys_r = [r["R_rev"] for r in records]
    # reference grid: first non-empty
    ref = None
    for x in xs:
        if x is not None and len(x) > 1:
            ref = x
            break
    if ref is None:
        return None, None, None
    Yf, Yr = [], []
    for x, yf, yr in zip(xs, ys_f, ys_r):
        if x is None or yf is None or yr is None or len(x) < 2:
            continue
        Yf.append(interp_to_grid(x, yf, ref))
        Yr.append(interp_to_grid(x, yr, ref))
    if not Yf:
        return None, None, None
    return ref, np.vstack(Yf), np.vstack(Yr)

def mean_ci(y: np.ndarray):
    mu = np.nanmean(y, axis=0)
    n  = np.sum(~np.isnan(y), axis=0).astype(float)
    sd = np.nanstd(y, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(np.maximum(n, 1.0)), out=np.zeros_like(sd), where=(n>0))
    lo = mu - 1.96 * sem
    hi = mu + 1.96 * sem
    return mu, lo, hi, n

def plot_agg_pair(monkey: str, pair: tuple[str,str], recs: list[dict], outdir: Path, save_csv: bool=False):
    A, B = pair
    log(f"  ↳ Aggregating pair {A}→{B} with {len(recs)} session curves")

    # Category
    tC, C_f, C_r = stack_on_grid(recs, "C")
    # Direction
    tR, R_f, R_r = stack_on_grid(recs, "R")

    if tC is None and tR is None:
        log("    (no alignable time grid for this pair; skipping)")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- Category ---
    if tC is not None:
        C_f_mu, C_f_lo, C_f_hi, nC = mean_ci(C_f)
        C_r_mu, C_r_lo, C_r_hi, _  = mean_ci(C_r)
        ax = axes[0]
        ax.fill_between(tC, C_f_lo, C_f_hi, color="grey", alpha=0.18, label="mean ± 95% CI (forward)")
        ax.plot(tC, C_f_mu, lw=2, label=f"{A}→{B} (C)")
        ax.plot(tC, C_r_mu, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (C, rev)")
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_ylabel("GC bits/bin (C)")
        ax.legend(frameon=False)
        if save_csv:
            dfC = pd.DataFrame({"t": tC, "mu_fwd": C_f_mu, "lo_fwd": C_f_lo, "hi_fwd": C_f_hi,
                                "mu_rev": C_r_mu, "lo_rev": C_r_lo, "hi_rev": C_r_hi, "n": nC})
            dfC.to_csv(outdir / f"flow_avg_C_{A}to{B}.csv", index=False)
        log(f"    Category: n_sessions per t (min–max) = {int(np.nanmin(nC))}–{int(np.nanmax(nC))}")
    else:
        axes[0].text(0.5, 0.5, "No Category data", ha="center", va="center")
        axes[0].set_ylabel("GC bits/bin (C)")

    # --- Direction ---
    if tR is not None:
        R_f_mu, R_f_lo, R_f_hi, nR = mean_ci(R_f)
        R_r_mu, R_r_lo, R_r_hi, _  = mean_ci(R_r)
        ax = axes[1]
        ax.fill_between(tR, R_f_lo, R_f_hi, color="grey", alpha=0.18, label="mean ± 95% CI (forward)")
        ax.plot(tR, R_f_mu, lw=2, label=f"{A}→{B} (R)")
        ax.plot(tR, R_r_mu, lw=1.5, ls="--", color="#d98400", label=f"{B}→{A} (R, rev)")
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_xlabel("time (s) from cat_stim_on")
        ax.set_ylabel("GC bits/bin (R)")
        ax.legend(frameon=False)
        if save_csv:
            dfR = pd.DataFrame({"t": tR, "mu_fwd": R_f_mu, "lo_fwd": R_f_lo, "hi_fwd": R_f_hi,
                                "mu_rev": R_r_mu, "lo_rev": R_r_lo, "hi_rev": R_r_hi, "n": nR})
            dfR.to_csv(outdir / f"flow_avg_R_{A}to{B}.csv", index=False)
        log(f"    Direction: n_sessions per t (min–max) = {int(np.nanmin(nR))}–{int(np.nanmax(nR))}")
    else:
        axes[1].text(0.5, 0.5, "No Direction data", ha="center", va="center")
        axes[1].set_xlabel("time (s) from cat_stim_on")
        axes[1].set_ylabel("GC bits/bin (R)")

    fig.suptitle(f"Session-averaged flow — Monkey {monkey}: {A}→{B}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_png = outdir / f"flow_avg_{A}to{B}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log(f"    [ok] wrote {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_csv", action="store_true", help="Also write CSV alongside PNGs")
    args = ap.parse_args()

    sids = detect_sessions()
    log(f"[info] Found {len(sids)} session folders under {SESS_ROOT}")
    if not sids:
        log("[warn] No sessions found. Exiting.")
        return

    # Buckets and manifests
    by_pair_M: dict[tuple[str,str], list[dict]] = {}
    by_pair_S: dict[tuple[str,str], list[dict]] = {}
    rows_manifest_M, rows_manifest_S = [], []

    # Walk sessions and collect files
    for sid in sids:
        pair_files = list_pair_files_for_session(sid)
        log(f"[info] Session {sid}: found {len(pair_files)} flow npz files")
        if not pair_files:
            continue
        # quick monkey hint by year prefix (your note: M≈2020*, S≈2023*), purely informational
        hint = "M?" if str(sid).startswith("2020") else ("S?" if str(sid).startswith("2023") else "?")
        log(f"       (year hint: {hint})")

        for npz in pair_files:
            name = npz.stem  # flow_timeseries_<A>to<B>
            if not name.startswith("flow_timeseries_") or "to" not in name:
                continue
            AtoB = name[len("flow_timeseries_"):]
            try:
                A, B = AtoB.split("to")
            except ValueError:
                log(f"       [skip] cannot parse pair from {npz.name}")
                continue
            mkA = area_monkey(A); mkB = area_monkey(B)
            if mkA is None or mkB is None or mkA != mkB:
                log(f"       [skip] cross-monkey or unknown areas: {A}->{B}")
                continue

            rec = load_pair_npz(npz)
            if rec["tC"] is None and rec["tR"] is None:
                log(f"       [skip] empty curves in {npz.name}")
                continue

            key = (A, B)
            if mkA == "M":
                by_pair_M.setdefault(key, []).append(rec)
                rows_manifest_M.append({"sid": sid, "pair": f"{A}->{B}", "file": str(npz.name)})
            else:
                by_pair_S.setdefault(key, []).append(rec)
                rows_manifest_S.append({"sid": sid, "pair": f"{A}->{B}", "file": str(npz.name)})

    # Report what we collected
    log(f"[info] Monkey M: collected {sum(len(v) for v in by_pair_M.values())} session-pair entries "
        f"across {len(by_pair_M)} pairs")
    log(f"[info] Monkey S: collected {sum(len(v) for v in by_pair_S.values())} session-pair entries "
        f"across {len(by_pair_S)} pairs")

    # Write manifests
    (OUT_ROOT / "monkey_M").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "monkey_S").mkdir(parents=True, exist_ok=True)
    if rows_manifest_M:
        pd.DataFrame(rows_manifest_M).sort_values(["pair","sid"]).to_csv(
            OUT_ROOT/"monkey_M"/"manifest_pairs.csv", index=False)
        log(f"[ok] wrote {OUT_ROOT/'monkey_M'/'manifest_pairs.csv'}")
    if rows_manifest_S:
        pd.DataFrame(rows_manifest_S).sort_values(["pair","sid"]).to_csv(
            OUT_ROOT/"monkey_S"/"manifest_pairs.csv", index=False)
        log(f"[ok] wrote {OUT_ROOT/'monkey_S'/'manifest_pairs.csv'}")

    # Aggregate & plot for each bucket
    for monkey, bucket in [("M", by_pair_M), ("S", by_pair_S)]:
        outdir_base = OUT_ROOT / f"monkey_{monkey}"
        if not bucket:
            log(f"[info] No pairs to aggregate for Monkey {monkey}")
            continue
        # Iterate over pairs actually present (not all 6 by default)
        for pair, recs in sorted(bucket.items()):
            plot_agg_pair(monkey, pair, recs, outdir_base, save_csv=args.save_csv)

    log("[done] Aggregation complete.")

if __name__ == "__main__":
    main()