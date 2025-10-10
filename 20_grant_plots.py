#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate three grant-ready plots with consistent styling:

(1) Group metrics (TOP PANEL ONLY = Category S bits), overlaying A→B and B→A:
    results/group_pairdiff/<group_tag>/monkey_<monkey>/
       metrics_BIDIR_C_<bidir_pair>_raw.png

(2) Group DIFF overlay (Category):
    results/group_pairdiff/<group_tag>/monkey_<monkey>/
       overlay_DIFF_C_<diff_pair>_raw.png

(3) Single-session overlay (Category, no p-marks):
    results/session/<sid>/<session_tag>/overlay_C_MFEF_MLIP_raw.png
    (We replot from data: induced_flow_MFEFtoMLIP.npz)

Outputs:
  <outdir>/grant_group_metrics_category_sbits_<bidir_pair>_raw.{svg,png,pdf}
  <outdir>/grant_group_overlay_diff_category_<diff_pair>_raw.{svg,png,pdf}
  <outdir>/grant_session_overlay_category_MFEF_MLIP_raw_<sid>.{svg,png,pdf}
  and CSVs with exact data used under <outdir>/data/
"""
from __future__ import annotations
from pathlib import Path
import argparse, csv, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------- small utils ----------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def as1d(a): return np.asarray(a, dtype=float).ravel()

def sem_band(arr2d: np.ndarray):
    """mean ± SEM along axis=0"""
    m = np.nanmean(arr2d, axis=0)
    n = np.sum(np.isfinite(arr2d), axis=0).astype(float)
    sd = np.nanstd(arr2d, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(np.maximum(n, 1.0)), out=np.zeros_like(sd), where=n>0)
    lo = m - sem; hi = m + sem
    return m, lo, hi, n

def ms(x): return 1000.0 * as1d(x)

def area_short(name: str) -> str:
    # Strip monkey prefix M/S
    return re.sub(r"^[MS]", "", name)

def dir_labels_colors(A: str, B: str):
    """Return ((label_fwd,color_fwd),(label_rev,color_rev)) mapping so that
       FEF→LIP is blue, LIP→FEF is red, regardless of A/B order."""
    Acore, Bcore = area_short(A), area_short(B)
    lab_fwd = f"{Acore} → {Bcore}"
    lab_rev = f"{Bcore} → {Acore}"
    col_fwd = "tab:blue"; col_rev = "tab:red"
    if ("FEF" in Acore and "LIP" in Bcore):
        col_fwd, col_rev = "tab:blue", "tab:red"
    elif ("FEF" in Bcore and "LIP" in Acore):
        col_fwd, col_rev = "tab:red", "tab:blue"
    return (lab_fwd, col_fwd), (lab_rev, col_rev)

def unit_factor(flow_unit: str) -> float:
    """Return multiplicative factor to convert flow from nats to desired unit."""
    if flow_unit == "bits":
        return 1.0 / np.log(2.0)   # nats -> bits
    return 1.0                     # nats (identity)


# ---------------------- 1) GROUP METRICS (S bits) ----------------------
def plot_group_metrics_sbits(repo: Path, group_tag: str, monkey: str, bidir_pair: str,
                             outdir: Path, figsize):
    m = re.match(r"([A-Za-z0-9]+)to([A-Za-z0-9]+)$", bidir_pair)
    if not m:
        raise ValueError(f"bidir_pair should look like AtoB, got: {bidir_pair}")
    A, B = m.group(1), m.group(2)

    session_root = repo / "results" / "session"
    sids = sorted([d.name for d in session_root.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}", d.name)])
    S_fwd_list, S_rev_list, t_ref = [], [], None

    for sid in sids:
        npz = session_root / sid / group_tag / "pairdiff" / f"{A}to{B}" / "pairdiff_timeseries_metrics.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        C = z["C"].item()
        t = C["t_raw"]
        if t_ref is None:
            t_ref = t
        elif t.size != t_ref.size or not np.allclose(t, t_ref, atol=1e-9):
            continue
        S_fwd_list.append(as1d(C["S_fwd_raw"]))
        S_rev_list.append(as1d(C["S_rev_raw"]))

    if not S_fwd_list or not S_rev_list:
        raise RuntimeError(f"No per-session NPZ found for {bidir_pair} @ {group_tag}")

    S_fwd = np.vstack(S_fwd_list)
    S_rev = np.vstack(S_rev_list)
    tm = ms(t_ref)

    (lab_fwd, col_fwd), (lab_rev, col_rev) = dir_labels_colors(A, B)
    mf, lf, hf, nf = sem_band(S_fwd)
    mr, lr, hr, nr = sem_band(S_rev)

    ddir = ensure_dir(outdir / "data")
    csv_path = ddir / f"data_group_metrics_category_Sbits_{A}_{B}_raw.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "mean_S_AtoB", "lo_S_AtoB", "hi_S_AtoB", "mean_S_BtoA", "lo_S_BtoA", "hi_S_BtoA",
                    "n_AtoB", "n_BtoA", "label_AtoB", "label_BtoA"])
        for i in range(tm.size):
            w.writerow([float(tm[i]), float(mf[i]), float(lf[i]), float(hf[i]),
                        float(mr[i]), float(lr[i]), float(hr[i]),
                        int(nf[i]), int(nr[i]), lab_fwd, lab_rev])

    plt.figure(figsize=figsize, dpi=200)
    ax = plt.gca()
    ax.fill_between(tm, lf, hf, color=col_fwd, alpha=0.20, linewidth=0)
    ax.plot(tm, mf, color=col_fwd, lw=2.2, label=lab_fwd)
    ax.fill_between(tm, lr, hr, color=col_rev, alpha=0.20, linewidth=0)
    ax.plot(tm, mr, color=col_rev, lw=2.2, label=lab_rev)
    ax.axvline(0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("Time from Stimulus Onset (ms)")
    ax.set_ylabel("Category S bits (−log₂ p)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.15)
    plt.tight_layout()

    base = outdir / f"grant_group_metrics_category_sbits_{A}_{B}_raw"
    for ext in ("svg", "png", "pdf"):
        plt.savefig(f"{base}.{ext}")
    plt.close()


# ---------------------- 2) GROUP DIFF OVERLAY ----------------------
def plot_group_overlay_diff(repo: Path, group_tag: str, monkey: str, diff_pair: str,
                            outdir: Path, alpha: float, figsize, flow_unit: str):
    """
    Re-plot group DIFF with consistent style from CSV written by 19_aggregate_pairdiff_metrics.py:
      results/group_pairdiff/<group_tag>/monkey_<monkey>/group_DIFF_C_<A>_<B>_raw.csv
    Adds black dots below the curve where significant (prefer group-null p_display).
    Converts flows to requested unit (bits or nats).
    """
    A, B = diff_pair.split("to")
    group_csv = repo / "results" / "group_pairdiff" / group_tag / f"monkey_{monkey}" / \
                f"group_DIFF_C_{A}_{B}_raw.csv"
    if not group_csv.exists():
        raise RuntimeError(f"Group CSV not found: {group_csv}")

    t, mean_diff, lo, hi, null_mu = [], [], [], [], []
    pdisp_list, sig_list, pmeta_list = [], [], []
    with open(group_csv, "r") as f:
        r = csv.DictReader(f)
        has_pdisp = "p_display" in r.fieldnames
        has_sig   = "sig"       in r.fieldnames
        has_pmeta = "p_meta"    in r.fieldnames
        for row in r:
            t.append(float(row["t"]))
            mean_diff.append(float(row["mean_diff"]))
            lo.append(float(row["ci_lo"]))
            hi.append(float(row["ci_hi"]))
            null_mu.append(float(row["null_mu_mean"]))
            if has_pdisp:
                try: pdisp_list.append(float(row["p_display"]))
                except: pdisp_list.append(np.nan)
            if has_sig:
                s = row["sig"].strip().lower()
                sig_list.append(1 if s in ("1","true","t","yes") else 0)
            if has_pmeta:
                try: pmeta_list.append(float(row["p_meta"]))
                except: pmeta_list.append(np.nan)

    t = ms(np.array(t)); mean_diff = as1d(mean_diff); lo = as1d(lo); hi = as1d(hi); null_mu = as1d(null_mu)

    # convert flow units if needed
    k = unit_factor(flow_unit)
    mean_diff *= k; lo *= k; hi *= k; null_mu *= k

    # decide significance mask
    sigmask = None
    if pdisp_list:
        pdisp = np.array(pdisp_list, dtype=float)
        sigmask = np.isfinite(pdisp) & (pdisp < alpha)
    elif sig_list:
        sigmask = np.array(sig_list, dtype=bool)
    elif pmeta_list:
        pmeta = np.array(pmeta_list, dtype=float)
        sigmask = np.isfinite(pmeta) & (pmeta < alpha)

    # Save data used for plotting (with converted units)
    ddir = ensure_dir(outdir / "data")
    csv_out = ddir / f"data_group_overlay_diff_category_{A}_{B}_raw_{flow_unit}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", f"mean_diff_{flow_unit}", f"band_lo_{flow_unit}",
                    f"band_hi_{flow_unit}", f"null_mu_mean_{flow_unit}", "sig"])
        for i in range(t.size):
            w.writerow([float(t[i]), float(mean_diff[i]), float(lo[i]), float(hi[i]),
                        float(null_mu[i]), int(sigmask[i]) if (sigmask is not None and i < sigmask.size) else 0])

    # Plot
    plt.figure(figsize=figsize, dpi=200)
    ax = plt.gca()
    ax.fill_between(t, lo, hi, color="tab:blue", alpha=0.20, linewidth=0, label="mean ± SEM")
    ax.plot(t, mean_diff, color="tab:blue", lw=2.4, label="Directional difference (FEF↔LIP)")
    ax.plot(t, null_mu, color="tab:gray", lw=1.5, ls="--", alpha=0.9, label="Mean null μ (difference)")
    ax.axvline(0, color="k", ls="--", lw=1.0)
    ax.axhline(0, color="k", ls=":",  lw=1.0)

    # black dots below the curve when significant
    if sigmask is not None and sigmask.size == t.size and np.any(sigmask):
        rng = np.nanmax(mean_diff) - np.nanmin(mean_diff)
        if not np.isfinite(rng) or rng <= 0: rng = 1e-6
        ybar = float(np.nanmin(mean_diff) - 0.05*rng)
        ax.plot(t[sigmask], np.full(sigmask.sum(), ybar), ".", ms=6, color="k", label=f"p < {alpha:g}")

    ax.set_xlabel("Time from Stimulus Onset (ms)")
    ylab_unit = "bits" if flow_unit == "bits" else "nats"
    ax.set_ylabel(f"Category flow difference ({ylab_unit})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    plt.tight_layout()

    base = outdir / f"grant_group_overlay_diff_category_{A}_{B}_raw_{flow_unit}"
    for ext in ("svg", "png", "pdf"):
        plt.savefig(f"{base}.{ext}")
    plt.close()


# ---------------------- 3) SINGLE SESSION OVERLAY ----------------------
def plot_session_overlay(repo: Path, sid: str, session_tag: str, outdir: Path, figsize, flow_unit: str):
    pair_npz = repo / "results" / "session" / sid / session_tag / "induced_flow_MFEFtoMLIP.npz"
    if not pair_npz.exists():
        raise RuntimeError(f"Pair NPZ not found: {pair_npz}")
    z = np.load(pair_npz, allow_pickle=True)
    t = z["tC"]; fwd = z["C_fwd"]; rev = z["C_rev"]
    fnull = z["C_fnull"]; rnull = z["C_rnull"]
    mu_f = np.nanmean(fnull, axis=0); mu_r = np.nanmean(rnull, axis=0)

    # convert flows
    k = unit_factor(flow_unit)
    fwd = as1d(fwd) * k; rev = as1d(rev) * k
    mu_f = as1d(mu_f) * k; mu_r = as1d(mu_r) * k

    tm = ms(t)
    col_fwd, col_rev = "tab:blue", "tab:red"
    lab_fwd, lab_rev = "MFEF → MLIP", "MLIP → MFEF"

    ddir = ensure_dir(outdir / "data")
    csv_out = ddir / f"data_session_overlay_category_MFEF_MLIP_raw_{sid}_{flow_unit}.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_ms", f"fwd_{flow_unit}", f"rev_{flow_unit}", f"null_mu_fwd_{flow_unit}", f"null_mu_rev_{flow_unit}"])
        for i in range(tm.size):
            w.writerow([float(tm[i]), float(fwd[i]), float(rev[i]), float(mu_f[i]), float(mu_r[i])])

    plt.figure(figsize=figsize, dpi=200)
    ax = plt.gca()
    ax.plot(tm, mu_f, ls="--", lw=1.2, color="tab:gray", alpha=0.9, label="null μ (MFEF → MLIP)")
    ax.plot(tm, mu_r, ls="--", lw=1.2, color="lightcoral", alpha=0.9, label="null μ (MLIP → MFEF)")
    ax.plot(tm, fwd, color=col_fwd, lw=2.4, label=lab_fwd)
    ax.plot(tm, rev, color=col_rev, lw=2.4, label=lab_rev)
    ax.axvline(0, color="k", ls="--", lw=1.0)
    ax.axhline(0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("Time from Stimulus Onset (ms)")
    ylab_unit = "bits" if flow_unit == "bits" else "nats"
    ax.set_ylabel(f"Category flow ({ylab_unit})")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.15)
    plt.tight_layout()

    base = outdir / f"grant_session_overlay_category_MFEF_MLIP_raw_{sid}_{flow_unit}"
    for ext in ("svg", "png", "pdf"):
        plt.savefig(f"{base}.{ext}")
    plt.close()


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/project/bdoiron/dracoxu/rct-fsflow", type=str)
    ap.add_argument("--group-tag", default="induced_k5_win016_p500", type=str)
    ap.add_argument("--monkey", default="M", choices=["M","S"])
    ap.add_argument("--bidir-pair", default="MLIPtoMFEF", type=str, help="AtoB for metrics (S bits) overlay")
    ap.add_argument("--diff-pair", default="MFEFtoMLIP", type=str, help="AtoB for DIFF overlay (Category)")
    ap.add_argument("--sid", default="20200327", type=str)
    ap.add_argument("--session-tag", default="induced_k5_win016_p500", type=str)
    ap.add_argument("--outdir", default="grant_plot", type=str)
    ap.add_argument("--figsize", default="8.6,3.6",
                    help="Figure size in inches as W,H (default 8.6,3.6)")
    ap.add_argument("--flow-unit", choices=["bits","nats"], default="bits",
                    help="Unit for flow curves (default: bits; nats are natural-log units)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    try:
        _w, _h = [float(x) for x in args.figsize.split(",")]
    except Exception:
        raise ValueError("Use --figsize like '8.6,3.6'")
    FIGSIZE = (_w, _h)

    repo = Path(args.repo)
    outdir = ensure_dir(Path(args.outdir))

    # 1) Group metrics S-bits (already in bits; no conversion)
    plot_group_metrics_sbits(repo, args.group_tag, args.monkey, args.bidir_pair, outdir, FIGSIZE)

    # 2) Group DIFF overlay (convert to chosen unit)
    plot_group_overlay_diff(repo, args.group_tag, args.monkey, args.diff_pair,
                            outdir, alpha=args.alpha, figsize=FIGSIZE, flow_unit=args.flow_unit)

    # 3) Single-session overlay (convert to chosen unit)
    plot_session_overlay(repo, args.sid, args.session_tag, outdir, FIGSIZE, args.flow_unit)

    print(f"[ok] wrote grant plots + data under: {outdir}")

if __name__ == "__main__":
    main()
