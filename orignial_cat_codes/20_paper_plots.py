#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figures (batch):
For each monkey (M/S), for each pair (A,B), and for each feature (C/R) and series (raw/int),
create a two-panel figure:
  Panel A: group DIFF (A→B − B→A) with SEM band + significance dots (group-null)
  Panel B: group S-bits overlay (A→B vs B→A, mean±SEM)

Inputs (already produced by your 18_ and 19_ scripts):
  - results/group_pairdiff/<group_tag>/monkey_<M|S>/group_DIFF_{C|R}_{A}_{B}_{raw|int}.csv
  - results/session/<sid>/<group_tag>/pairdiff/<A>to<B>/pairdiff_timeseries_metrics.npz (for S-bits)

Outputs:
  results/pre_paper_figs/<group_tag>/monkey_<M|S>/
    paper_{category|direction}_{A}_{B}_{raw|int}_{bits|nats}.{svg,png,pdf}
    python 20_paper_plots.py \
  --repo /project/bdoiron/dracoxu/rct-fsflow \
  --group-tag induced_k5_win016_p500 \
  --monkeys M \
  --series raw int \
  --out-root results/pre_paper_figs

"""

from __future__ import annotations
from pathlib import Path
import argparse, csv, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- style constants (you can tweak these) --------
AX_BOX_TOP = (0.14, 0.57, 0.82, 0.36)   # top panel (DIFF)
AX_BOX_BOT = (0.14, 0.13, 0.82, 0.36)   # bottom panel (S-bits)
DIFF_LINE_COLOR = "#1f6feb"             # custom blue for Δ flow
DIFF_BAND_COLOR = "#9ec1ff"             # lighter band
FWD_COLOR = "tab:blue"                  # A→B in S-bits panel (FEF→LIP will resolve to blue)
REV_COLOR = "tab:red"                   # B→A in S-bits panel (LIP→FEF red)

# ---------------- utils ----------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def as1d(a): return np.asarray(a, dtype=float).ravel()

def ms(x): return 1000.0 * as1d(x)

def sem_band(arr2d: np.ndarray):
    m = np.nanmean(arr2d, axis=0)
    n = np.sum(np.isfinite(arr2d), axis=0).astype(float)
    sd = np.nanstd(arr2d, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(np.maximum(n, 1.0)), out=np.zeros_like(sd), where=n>0)
    lo = m - sem; hi = m + sem
    return m, lo, hi, n

def area_short(name: str) -> str:
    return re.sub(r"^[MS]", "", name)

def dir_labels_colors(A: str, B: str):
    Acore, Bcore = area_short(A), area_short(B)
    lab_fwd = f"{Acore} → {Bcore}"
    lab_rev = f"{Bcore} → {Acore}"
    # FEF→LIP blue, LIP→FEF red
    if ("FEF" in Acore and "LIP" in Bcore):
        col_fwd, col_rev = FWD_COLOR, REV_COLOR
    elif ("FEF" in Bcore and "LIP" in Acore):
        col_fwd, col_rev = REV_COLOR, FWD_COLOR
    else:
        col_fwd, col_rev = FWD_COLOR, REV_COLOR
    return (lab_fwd, col_fwd), (lab_rev, col_rev)

def unit_factor(flow_unit: str) -> float:
    return 1.0/np.log(2.0) if flow_unit == "bits" else 1.0

def feat_name(feat: str) -> str:
    return "category" if feat.upper()=="C" else "direction"

# ------------- gather pairs from group CSVs -------------
def list_pairs_for_monkey(repo: Path, group_tag: str, monkey: str, feat: str, series: str) -> list[tuple[str,str]]:
    gdir = repo / "results" / "group_pairdiff" / group_tag / f"monkey_{monkey}"
    if not gdir.exists(): return []
    pairs = []
    pat = re.compile(rf"group_DIFF_{feat.upper()}_([A-Za-z0-9]+)_([A-Za-z0-9]+)_{series}\.csv$")
    for p in gdir.glob(f"group_DIFF_{feat.upper()}_*_*_{series}.csv"):
        m = pat.match(p.name)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return sorted(set(pairs))

# ------------- read group DIFF CSV -------------
def read_group_diff_csv(repo: Path, group_tag: str, monkey: str, feat: str, A: str, B: str, series: str):
    p = repo / "results" / "group_pairdiff" / group_tag / f"monkey_{monkey}" / f"group_DIFF_{feat.upper()}_{A}_{B}_{series}.csv"
    if not p.exists(): return None
    t = []; mean_diff=[]; lo=[]; hi=[]; null_mu=[]; pdisp=[]; sig=[]; pmeta=[]
    with open(p, "r") as f:
        r = csv.DictReader(f)
        has_pdisp = "p_display" in r.fieldnames
        has_sig   = "sig" in r.fieldnames
        has_pmeta = "p_meta" in r.fieldnames
        for row in r:
            t.append(float(row["t"]))
            mean_diff.append(float(row["mean_diff"]))
            lo.append(float(row["ci_lo"]))
            hi.append(float(row["ci_hi"]))
            null_mu.append(float(row["null_mu_mean"]))
            if has_pdisp:
                try: pdisp.append(float(row["p_display"]))
                except: pdisp.append(np.nan)
            if has_sig:
                s = row["sig"].strip().lower()
                sig.append(1 if s in ("1","true","t","yes") else 0)
            if has_pmeta:
                try: pmeta.append(float(row["p_meta"]))
                except: pmeta.append(np.nan)
    return dict(t=ms(np.array(t)), diff=as1d(mean_diff), lo=as1d(lo), hi=as1d(hi),
                null_mu=as1d(null_mu), pdisp=np.array(pdisp) if pdisp else None,
                sig=np.array(sig) if sig else None, pmeta=np.array(pmeta) if pmeta else None)

# ------------- aggregate S-bits across sessions -------------
def aggregate_sbits(repo: Path, group_tag: str, A: str, B: str, feat: str, series: str):
    session_root = repo / "results" / "session"
    S_fwd_list=[]; S_rev_list=[]; t_ref=None
    key = {"C":"C", "R":"R"}[feat.upper()]
    kser = "raw" if series=="raw" else "int"
    for d in sorted(session_root.iterdir()):
        if not (d.is_dir() and re.fullmatch(r"\d{8}", d.name)): continue
        npz = session_root / d.name / group_tag / "pairdiff" / f"{A}to{B}" / "pairdiff_timeseries_metrics.npz"
        if not npz.exists(): continue
        z = np.load(npz, allow_pickle=True); D = z[key].item()
        t = D[f"t_{kser}"]
        if t_ref is None:
            t_ref = t
        elif t.size != t_ref.size or not np.allclose(t, t_ref, atol=1e-9):
            continue
        S_fwd_list.append(as1d(D[f"S_fwd_{kser}"]))
        S_rev_list.append(as1d(D[f"S_rev_{kser}"]))
    if not S_fwd_list or not S_rev_list: return None
    return dict(t=ms(t_ref), S_fwd=np.vstack(S_fwd_list), S_rev=np.vstack(S_rev_list))

# ------------- common x-limits -------------
def compute_xlim_from(t_arrays: list[np.ndarray]):
    arrs = [a for a in t_arrays if a is not None and a.size]
    if not arrs: return None
    return (float(np.nanmin([a.min() for a in arrs])),
            float(np.nanmax([a.max() for a in arrs])))

# ------------- one figure per pair/feat/series -------------
def make_paper_figure(repo: Path, outdir: Path, group_tag: str, monkey: str,
                      A: str, B: str, feat: str, series: str, flow_unit: str, alpha: float, figsize):
    # Read group diff
    gd = read_group_diff_csv(repo, group_tag, monkey, feat, A, B, series)
    if gd is None: return False
    # Aggregate S-bits
    sb = aggregate_sbits(repo, group_tag, A, B, feat, series)
    if sb is None: return False

    # Units for DIFF
    k = unit_factor(flow_unit)
    t_diff = gd["t"]; diff = gd["diff"]*k; lo = gd["lo"]*k; hi = gd["hi"]*k; null_mu = gd["null_mu"]*k
    sigmask = None
    if gd["pdisp"] is not None and gd["pdisp"].size: sigmask = np.isfinite(gd["pdisp"]) & (gd["pdisp"] < alpha)
    elif gd["sig"] is not None and gd["sig"].size: sigmask = gd["sig"].astype(bool)
    elif gd["pmeta"] is not None and gd["pmeta"].size: sigmask = np.isfinite(gd["pmeta"]) & (gd["pmeta"] < alpha)

    # S-bits
    t_s = sb["t"]; S_fwd = sb["S_fwd"]; S_rev = sb["S_rev"]
    (lab_fwd, col_fwd), (lab_rev, col_rev) = dir_labels_colors(A, B)
    mf, lf, hf, _ = sem_band(S_fwd)
    mr, lr, hr, _ = sem_band(S_rev)

    # Common x-limits for both panels
    xlim = compute_xlim_from([t_diff, t_s])

    # Figure
    fig = plt.figure(figsize=figsize, dpi=200)

    # Panel A: DIFF
    ax1 = fig.add_axes(AX_BOX_TOP)
    ax1.fill_between(t_diff, lo, hi, color=DIFF_BAND_COLOR, alpha=0.20, linewidth=0, label="mean ± SEM")
    ax1.plot(t_diff, diff, color=DIFF_LINE_COLOR, lw=2.4, label="Directional difference (FEF↔LIP)")
    ax1.plot(t_diff, null_mu, color="tab:gray", lw=1.5, ls="--", alpha=0.9, label="Mean null μ")
    ax1.axvline(0, color="k", ls="--", lw=1.0); ax1.axhline(0, color="k", ls=":", lw=1.0)
    if xlim: ax1.set_xlim(xlim)
    if sigmask is not None and sigmask.size == t_diff.size and np.any(sigmask):
        rng = np.nanmax(diff) - np.nanmin(diff); rng = max(rng, 1e-6)
        ybar = float(np.nanmin(diff) - 0.05*rng)
        ax1.plot(t_diff[sigmask], np.full(sigmask.sum(), ybar), ".", ms=6, color="k", label=f"p < {alpha:g}")
    yunit = "bits" if flow_unit=="bits" else "nats"
    ax1.set_ylabel(f"Δ {feat_name(feat).capitalize()} flow ({yunit})")
    ax1.legend(frameon=False); ax1.grid(alpha=0.15)

    # Panel B: S-bits overlay
    ax2 = fig.add_axes(AX_BOX_BOT)
    ax2.fill_between(t_s, lf, hf, color=col_fwd, alpha=0.20, linewidth=0)
    ax2.plot(t_s, mf, color=col_fwd, lw=2.2, label=lab_fwd)
    ax2.fill_between(t_s, lr, hr, color=col_rev, alpha=0.20, linewidth=0)
    ax2.plot(t_s, mr, color=col_rev, lw=2.2, label=lab_rev)
    ax2.axvline(0, color="k", ls="--", lw=1.0)
    if xlim: ax2.set_xlim(xlim)
    ax2.set_xlabel("Time from Stimulus Onset (ms)")
    ax2.set_ylabel("S bits (−log₂ p)")
    ax2.legend(frameon=False, ncol=2); ax2.grid(alpha=0.15)

    feat_full = feat_name(feat)
    base = outdir / f"paper_{feat_full}_{A}_{B}_{series}_{flow_unit}"
    for ext in ("svg","png","pdf"):
        fig.savefig(f"{base}.{ext}")
    plt.close(fig)
    return True

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/project/bdoiron/dracoxu/rct-fsflow", type=str)
    ap.add_argument("--group-tag", required=True, type=str)
    ap.add_argument("--monkeys", nargs="+", default=["M","S"], choices=["M","S"])
    ap.add_argument("--series", nargs="+", default=["raw"], choices=["raw","int"],
                    help="Which series to render per figure (default: raw). Use 'raw int' for both.")
    ap.add_argument("--flow-unit", choices=["bits","nats"], default="bits")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-root", default="results/pre_paper_figs", type=str)
    ap.add_argument("--figsize", default="6.6,4.8", help="Figure size W,H (inches) for each two-panel figure")
    ap.add_argument("--pairs", nargs="*", default=[], help="Optional whitelist of pairs like MFEFtoMLIP MLIPtoMSC ...")
    args = ap.parse_args()

    try:
        _w, _h = [float(x) for x in args.figsize.split(",")]
    except Exception:
        raise ValueError("Use --figsize like '6.6,4.8'")
    FIGSIZE = (_w, _h)

    repo = Path(args.repo)
    out_root = Path(args.out_root)

    feats = ["C","R"]
    total = 0
    for mk in args.monkeys:
        for feat in feats:
            for series in args.series:
                # discover pairs for this monkey/feat/series
                pairs = list_pairs_for_monkey(repo, args.group_tag, mk, feat, series)
                if args.pairs:
                    keep = set(args.pairs)
                    pairs = [p for p in pairs if f"{p[0]}to{p[1]}" in keep]
                if not pairs: continue
                outdir = ensure_dir(out_root / args.group_tag / f"monkey_{mk}")
                for (A,B) in pairs:
                    ok = make_paper_figure(repo, outdir, args.group_tag, mk, A, B, feat, series,
                                           args.flow_unit, args.alpha, FIGSIZE)
                    if ok:
                        total += 1
    print(f"[ok] wrote {total} paper figures under {out_root}")

if __name__ == "__main__":
    main()
