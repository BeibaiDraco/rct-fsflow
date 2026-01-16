#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-session pair analysis:
  • FEF→LIP, LIP→FEF, and DIFF = (FEF→LIP − LIP→FEF)
  • per-time p (raw & integrated) with paired permutation nulls
  • overlays with NULL MEANS
  • per-time "how far above null?" metrics: S bits, robust z, z_from_p, CLES
  • band/all-time summaries to CSV

Outputs in --outdir:
  overlay_diff_{C|R}_{raw|int}.png
  metrics_{C|R}_{raw|int}.png        # 4 panels: S, z_rob, z_from_p, CLES
  pvals_{C|R}_{raw|int}.png
  pairdiff_timeseries_metrics.npz    # all arrays
  pairdiff_metrics_summary.csv       # band/all-time means
"""
from pathlib import Path
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------- utils -------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def zhas(z, key: str) -> bool:
    return hasattr(z, "files") and (key in z.files)

def zget(z, key: str, default=None):
    return z[key] if zhas(z, key) else default

def as1d(a): return np.asarray(a, dtype=float).ravel()

def p_from_null(obs: np.ndarray, null_mat: np.ndarray) -> np.ndarray:
    obs = as1d(obs); null_mat = np.asarray(null_mat, dtype=float)
    if null_mat.ndim != 2 or null_mat.shape[1] != obs.size:
        null_mat = null_mat.reshape(null_mat.shape[0], -1)
        if null_mat.shape[1] != obs.size:
            raise ValueError("null_mat has incompatible shape with obs.")
    ge = (null_mat >= obs[None, :]).sum(axis=0)
    P = null_mat.shape[0]
    return (1 + ge) / (1 + P)

def null_mean(null_mat: np.ndarray) -> np.ndarray:
    return np.nanmean(np.asarray(null_mat, dtype=float), axis=0)

def sliding_integrate(series: np.ndarray, centers: np.ndarray, int_win: float) -> np.ndarray:
    series = as1d(series); centers = as1d(centers)
    if series.size == 0: return series
    step = float(np.median(np.diff(centers))) if len(centers) >= 2 else max(int_win, 1e-6)
    L = max(1, int(round(int_win / max(step, 1e-9))))
    return np.convolve(series, np.ones(L, dtype=float), mode="same")

def sliding_integrate_null(null_mat: np.ndarray, centers: np.ndarray, int_win: float) -> np.ndarray:
    null_mat = np.asarray(null_mat, dtype=float)
    out = np.empty_like(null_mat, dtype=float)
    for i in range(null_mat.shape[0]):
        out[i] = sliding_integrate(null_mat[i], centers, int_win)
    return out

def integrate_in_band(centers, series, start, end) -> float:
    centers = as1d(centers); series = as1d(series)
    m = (centers >= start) & (centers <= end)
    return float(np.nansum(series[m])) if np.any(m) else float("nan")


# ---- normal inverse CDF (Acklam approximation) to avoid SciPy dependency ----
def _norm_ppf(p):
    p = np.asarray(p, dtype=float); eps = 1e-12
    pp = np.clip(p, eps, 1 - eps)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 0.97575
    q = np.zeros_like(pp)
    lo = pp < plow; md = (pp >= plow) & (pp <= phigh); hi = pp > phigh
    if np.any(lo):
        ql = np.sqrt(-2*np.log(pp[lo]))
        q[lo] = (((((c[0]*ql + c[1])*ql + c[2])*ql + c[3])*ql + c[4])*ql + c[5]) / \
                ((((d[0]*ql + d[1])*ql + d[2])*ql + d[3])*ql + 1)
        q[lo] *= -1
    if np.any(md):
        ql = pp[md] - 0.5; r = ql*ql
        q[md] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*ql / \
                 (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    if np.any(hi):
        ql = np.sqrt(-2*np.log(1-pp[hi]))
        q[hi] = (((((c[0]*ql + c[1])*ql + c[2])*ql + c[3])*ql + c[4])*ql + c[5]) / \
                 ((((d[0]*ql + d[1])*ql + d[2])*ql + d[3])*ql + 1)
    return q


# --------- per-time "how far above null?" metrics (vectorized) ---------------
def null_effects_time(obs: np.ndarray, null_mat: np.ndarray):
    obs = as1d(obs); null_mat = np.asarray(null_mat, dtype=float)
    if null_mat.ndim != 2 or null_mat.shape[1] != obs.size:
        null_mat = null_mat.reshape(null_mat.shape[0], -1)
        if null_mat.shape[1] != obs.size:
            raise ValueError("null_mat has incompatible shape with obs.")
    p = p_from_null(obs, null_mat)
    S = -np.log2(np.maximum(p, 1e-300))
    mu = np.nanmean(null_mat, axis=0)
    sd = np.nanstd(null_mat, axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_null = (obs - mu) / sd
    med = np.nanmedian(null_mat, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(null_mat - med[None, :]), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_robust = (obs - med) / mad
    z_from_p = _norm_ppf(1.0 - p)
    cles = 1.0 - p
    return dict(p=p, S=S, z_null=z_null, z_robust=z_robust, z_from_p=z_from_p, cles=cles)


# --------------------------- plotting helpers --------------------------------
def _sigbar_y(*series):
    vals = np.concatenate([as1d(s) for s in series])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0
    rng = np.nanmax(vals) - np.nanmin(vals)
    if not np.isfinite(rng) or rng <= 0: rng = 1e-6
    return float(np.nanmin(vals) - 0.05 * rng)

def plot_overlay_mean_null(out_png: Path, title: str, x, fwd, rev, diff,
                           fnull_mu, rnull_mu, dnull_mu, p_diff, alpha=0.05):
    x = as1d(x); fwd = as1d(fwd); rev = as1d(rev); diff = as1d(diff)
    plt.figure(figsize=(8.6, 4.6), dpi=160)

    # Null means as dashed lines
    if fnull_mu is not None: plt.plot(x, as1d(fnull_mu), ls="--", lw=1.2, alpha=0.8, label="null μ (FEF→LIP)")
    if rnull_mu is not None: plt.plot(x, as1d(rnull_mu), ls="--", lw=1.2, alpha=0.8, label="null μ (LIP→FEF)")
    if dnull_mu is not None: plt.plot(x, as1d(dnull_mu), ls="--", lw=1.2, alpha=0.8, label="null μ (diff)")

    # Curves
    plt.plot(x, fwd, lw=2.2, label="FEF→LIP")
    plt.plot(x, rev, lw=2.2, label="LIP→FEF")
    plt.plot(x, diff, lw=2.2, label="(FEF→LIP) − (LIP→FEF)", alpha=0.95)
    plt.axhline(0, color="k", ls=":", lw=1)

    # Sig ticks for diff
    if p_diff is not None:
        p_diff = as1d(p_diff)
        if p_diff.size == x.size:
            sigmask = np.isfinite(p_diff) & (p_diff < alpha)
            if np.any(sigmask):
                ybar = _sigbar_y(fwd, rev, diff)
                plt.plot(x[sigmask], np.full(sigmask.sum(), ybar), ".", ms=6)

    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("Flow")
    plt.legend(frameon=False, ncol=2); plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_metrics_traces4(out_png: Path, title: str, x,
                         S_fwd, S_rev, S_diff,
                         zrob_fwd, zrob_rev, zrob_diff,
                         zfp_fwd, zfp_rev, zfp_diff,
                         cles_fwd, cles_rev, cles_diff):
    """4 stacked panels: S bits, robust z, z_from_p, CLES."""
    x = as1d(x)
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 10.0), dpi=160, sharex=True)

    # 1) S bits
    ax = axes[0]
    ax.plot(x, as1d(S_fwd), lw=2.0, label="S FEF")
    ax.plot(x, as1d(S_rev), lw=2.0, label="S LIP")
    ax.plot(x, as1d(S_diff), lw=2.0, label="S diff")
    ax.set_ylabel("S bits"); ax.legend(frameon=False, ncol=3); ax.grid(alpha=0.25)

    # 2) robust z
    ax = axes[1]
    ax.plot(x, as1d(zrob_fwd), lw=2.0, label="z_rob FEF")
    ax.plot(x, as1d(zrob_rev), lw=2.0, label="z_rob LIP")
    ax.plot(x, as1d(zrob_diff), lw=2.0, label="z_rob diff")
    ax.set_ylabel("z_rob"); ax.legend(frameon=False, ncol=3); ax.grid(alpha=0.25)

    # 3) z_from_p
    ax = axes[2]
    ax.plot(x, as1d(zfp_fwd), lw=2.0, label="z_from_p FEF")
    ax.plot(x, as1d(zfp_rev), lw=2.0, label="z_from_p LIP")
    ax.plot(x, as1d(zfp_diff), lw=2.0, label="z_from_p diff")
    ax.set_ylabel("z_from_p"); ax.legend(frameon=False, ncol=3); ax.grid(alpha=0.25)

    # 4) CLES
    ax = axes[3]
    ax.plot(x, as1d(cles_fwd), lw=2.0, label="CLES FEF")
    ax.plot(x, as1d(cles_rev), lw=2.0, label="CLES LIP")
    ax.plot(x, as1d(cles_diff), lw=2.0, label="CLES diff")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("CLES (1−p)")
    ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=3); ax.grid(alpha=0.25)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def plot_pvals_traces(out_png: Path, title: str, x, p_fwd, p_rev, p_diff, alpha=0.05):
    x = as1d(x); tolog = lambda p: -np.log10(np.maximum(as1d(p), 1e-300))
    plt.figure(figsize=(8.6, 4.6), dpi=160)
    plt.plot(x, tolog(p_fwd), lw=2.0, label="-log10 p FEF")
    plt.plot(x, tolog(p_rev), lw=2.0, label="-log10 p LIP")
    plt.plot(x, tolog(p_diff), lw=2.0, label="-log10 p diff")
    plt.axhline(-np.log10(alpha), color="k", ls=":", lw=1, label=f"α={alpha}")
    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("-log10 p")
    plt.legend(frameon=False, ncol=4); plt.tight_layout(); plt.savefig(out_png); plt.close()


# ------------------------------- core ----------------------------------------
def compute_for_prefix(z, prefix: str, alpha: float):
    out = {}
    t = z[f"t{prefix}"]
    int_win = float(z["int_win"]) if zhas(z, "int_win") else 0.16
    band = zget(z, "band", np.array([0.12, 0.28], dtype=float))
    b0, b1 = float(band[0]), float(band[1])

    fwd = z[f"{prefix}_fwd"]; rev = z[f"{prefix}_rev"]
    fnull = z[f"{prefix}_fnull"]; rnull = z[f"{prefix}_rnull"]
    P = min(fnull.shape[0], rnull.shape[0])
    diff = fwd - rev; dnull = fnull[:P, :] - rnull[:P, :]

    ef_fwd = null_effects_time(fwd, fnull)
    ef_rev = null_effects_time(rev, rnull)
    ef_diff = null_effects_time(diff, dnull)

    out.update({
        "t_raw": t, "int_win": int_win, "band": np.array([b0, b1], float),
        "fwd_raw": fwd, "rev_raw": rev, "diff_raw": diff,
        "fnull_mu_raw": null_mean(fnull), "rnull_mu_raw": null_mean(rnull), "dnull_mu_raw": null_mean(dnull),
        "p_fwd_raw": ef_fwd["p"], "p_rev_raw": ef_rev["p"], "p_diff_raw": ef_diff["p"],
        "S_fwd_raw": ef_fwd["S"], "S_rev_raw": ef_rev["S"], "S_diff_raw": ef_diff["S"],
        "znull_fwd_raw": ef_fwd["z_null"], "znull_rev_raw": ef_rev["z_null"], "znull_diff_raw": ef_diff["z_null"],
        "zrob_fwd_raw": ef_fwd["z_robust"], "zrob_rev_raw": ef_rev["z_robust"], "zrob_diff_raw": ef_diff["z_robust"],
        "zfromp_fwd_raw": ef_fwd["z_from_p"], "zfromp_rev_raw": ef_rev["z_from_p"], "zfromp_diff_raw": ef_diff["z_from_p"],
        "cles_fwd_raw": ef_fwd["cles"], "cles_rev_raw": ef_rev["cles"], "cles_diff_raw": ef_diff["cles"],
    })

    fwd_sl = zget(z, f"{prefix}_fwd_sl", sliding_integrate(fwd, t, int_win))
    rev_sl = zget(z, f"{prefix}_rev_sl", sliding_integrate(rev, t, int_win))
    fnull_sl = zget(z, f"{prefix}_null_sl", sliding_integrate_null(fnull, t, int_win))
    rnull_sl = sliding_integrate_null(rnull, t, int_win)
    diff_sl = fwd_sl - rev_sl; dnull_sl = fnull_sl - rnull_sl

    ef_fwd_sl = null_effects_time(fwd_sl, fnull_sl)
    ef_rev_sl = null_effects_time(rev_sl, rnull_sl)
    ef_diff_sl = null_effects_time(diff_sl, dnull_sl)

    out.update({
        "t_int": t,
        "fwd_int": fwd_sl, "rev_int": rev_sl, "diff_int": diff_sl,
        "fnull_mu_int": null_mean(fnull_sl), "rnull_mu_int": null_mean(rnull_sl), "dnull_mu_int": null_mean(dnull_sl),
        "p_fwd_int": ef_fwd_sl["p"], "p_rev_int": ef_rev_sl["p"], "p_diff_int": ef_diff_sl["p"],
        "S_fwd_int": ef_fwd_sl["S"], "S_rev_int": ef_rev_sl["S"], "S_diff_int": ef_diff_sl["S"],
        "znull_fwd_int": ef_fwd_sl["z_null"], "znull_rev_int": ef_rev_sl["z_null"], "znull_diff_int": ef_diff_sl["z_null"],
        "zrob_fwd_int": ef_fwd_sl["z_robust"], "zrob_rev_int": ef_rev_sl["z_robust"], "zrob_diff_int": ef_diff_sl["z_robust"],
        "zfromp_fwd_int": ef_fwd_sl["z_from_p"], "zfromp_rev_int": ef_rev_sl["z_from_p"], "zfromp_diff_int": ef_diff_sl["z_from_p"],
        "cles_fwd_int": ef_fwd_sl["cles"], "cles_rev_int": ef_rev_sl["cles"], "cles_diff_int": ef_diff_sl["cles"],
    })

    # band-integral scalars + time-averaged summaries
    I_fwd = integrate_in_band(t, fwd, b0, b1)
    I_rev = integrate_in_band(t, rev, b0, b1)
    I_diff = integrate_in_band(t, diff, b0, b1)
    I_fwd_null = np.array([integrate_in_band(t, row, b0, b1) for row in fnull])
    I_rev_null = np.array([integrate_in_band(t, row, b0, b1) for row in rnull])
    P2 = min(len(I_fwd_null), len(I_rev_null))
    I_dnull = I_fwd_null[:P2] - I_rev_null[:P2]
    pID = (1 + np.sum(I_dnull >= I_diff)) / (1 + len(I_dnull))

    mask_band = (t >= b0) & (t <= b1)
    def tmean(x, m): x = as1d(x); return float(np.nanmean(x[m])) if np.any(m) else float("nan")

    summary = dict(
        band_start=b0, band_end=b1,
        I_fwd=I_fwd, I_rev=I_rev, I_diff=I_diff, pID_diff=pID,
        mean_S_fwd_band=tmean(out["S_fwd_raw"], mask_band),
        mean_S_rev_band=tmean(out["S_rev_raw"], mask_band),
        mean_S_diff_band=tmean(out["S_diff_raw"], mask_band),
        mean_zrob_fwd_band=tmean(out["zrob_fwd_raw"], mask_band),
        mean_zrob_rev_band=tmean(out["zrob_rev_raw"], mask_band),
        mean_zrob_diff_band=tmean(out["zrob_diff_raw"], mask_band),
        mean_zfromp_fwd_band=tmean(out["zfromp_fwd_raw"], mask_band),
        mean_zfromp_rev_band=tmean(out["zfromp_rev_raw"], mask_band),
        mean_zfromp_diff_band=tmean(out["zfromp_diff_raw"], mask_band),
        mean_cles_fwd_band=tmean(out["cles_fwd_raw"], mask_band),
        mean_cles_rev_band=tmean(out["cles_rev_raw"], mask_band),
        mean_cles_diff_band=tmean(out["cles_diff_raw"], mask_band),
        frac_sig_diff_band=float(np.nanmean(out["p_diff_raw"][mask_band] < alpha)) if np.any(mask_band) else np.nan,
        mean_S_fwd_all=float(np.nanmean(out["S_fwd_raw"])),
        mean_S_rev_all=float(np.nanmean(out["S_rev_raw"])),
        mean_S_diff_all=float(np.nanmean(out["S_diff_raw"])),
        mean_zrob_fwd_all=float(np.nanmean(out["zrob_fwd_raw"])),
        mean_zrob_rev_all=float(np.nanmean(out["zrob_rev_raw"])),
        mean_zrob_diff_all=float(np.nanmean(out["zrob_diff_raw"])),
        mean_zfromp_fwd_all=float(np.nanmean(out["zfromp_fwd_raw"])),
        mean_zfromp_rev_all=float(np.nanmean(out["zfromp_rev_raw"])),
        mean_zfromp_diff_all=float(np.nanmean(out["zfromp_diff_raw"])),
        mean_cles_fwd_all=float(np.nanmean(out["cles_fwd_raw"])),
        mean_cles_rev_all=float(np.nanmean(out["cles_rev_raw"])),
        mean_cles_diff_all=float(np.nanmean(out["cles_diff_raw"])),
        frac_sig_diff_all=float(np.nanmean(out["p_diff_raw"] < alpha)),
    )
    out["summary"] = summary
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True,
                    help="Per-pair NPZ (e.g., induced_flow_MFEFtoMLIP.npz) containing fwd/rev + full nulls")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    z = np.load(args.pair, allow_pickle=True)

    C = compute_for_prefix(z, "C", args.alpha)
    R = compute_for_prefix(z, "R", args.alpha)

    # Save arrays
    np.savez_compressed(outdir / "pairdiff_timeseries_metrics.npz", C=C, R=R)

    # Overlays with NULL MEANS
    plot_overlay_mean_null(outdir / "overlay_diff_C_raw.png",
        f"C raw   band[{C['band'][0]:.2f},{C['band'][1]:.2f}]",
        C["t_raw"], C["fwd_raw"], C["rev_raw"], C["diff_raw"],
        C["fnull_mu_raw"], C["rnull_mu_raw"], C["dnull_mu_raw"],
        C["p_diff_raw"], args.alpha)
    plot_overlay_mean_null(outdir / "overlay_diff_C_int.png",
        f"C integrated (win={C['int_win']:.3f}s)   band[{C['band'][0]:.2f},{C['band'][1]:.2f}]",
        C["t_int"], C["fwd_int"], C["rev_int"], C["diff_int"],
        C["fnull_mu_int"], C["rnull_mu_int"], C["dnull_mu_int"],
        C["p_diff_int"], args.alpha)
    plot_overlay_mean_null(outdir / "overlay_diff_R_raw.png",
        f"R raw   band[{R['band'][0]:.2f},{R['band'][1]:.2f}]",
        R["t_raw"], R["fwd_raw"], R["rev_raw"], R["diff_raw"],
        R["fnull_mu_raw"], R["rnull_mu_raw"], R["dnull_mu_raw"],
        R["p_diff_raw"], args.alpha)
    plot_overlay_mean_null(outdir / "overlay_diff_R_int.png",
        f"R integrated (win={R['int_win']:.3f}s)   band[{R['band'][0]:.2f},{R['band'][1]:.2f}]",
        R["t_int"], R["fwd_int"], R["rev_int"], R["diff_int"],
        R["fnull_mu_int"], R["rnull_mu_int"], R["dnull_mu_int"],
        R["p_diff_int"], args.alpha)

    # 4-panel metrics traces
    plot_metrics_traces4(outdir / "metrics_C_raw.png", "C — per-time metrics (raw)", C["t_raw"],
        C["S_fwd_raw"], C["S_rev_raw"], C["S_diff_raw"],
        C["zrob_fwd_raw"], C["zrob_rev_raw"], C["zrob_diff_raw"],
        C["zfromp_fwd_raw"], C["zfromp_rev_raw"], C["zfromp_diff_raw"],
        C["cles_fwd_raw"], C["cles_rev_raw"], C["cles_diff_raw"])
    plot_metrics_traces4(outdir / "metrics_C_int.png", "C — per-time metrics (integrated)", C["t_int"],
        C["S_fwd_int"], C["S_rev_int"], C["S_diff_int"],
        C["zrob_fwd_int"], C["zrob_rev_int"], C["zrob_diff_int"],
        C["zfromp_fwd_int"], C["zfromp_rev_int"], C["zfromp_diff_int"],
        C["cles_fwd_int"], C["cles_rev_int"], C["cles_diff_int"])
    plot_metrics_traces4(outdir / "metrics_R_raw.png", "R — per-time metrics (raw)", R["t_raw"],
        R["S_fwd_raw"], R["S_rev_raw"], R["S_diff_raw"],
        R["zrob_fwd_raw"], R["zrob_rev_raw"], R["zrob_diff_raw"],
        R["zfromp_fwd_raw"], R["zfromp_rev_raw"], R["zfromp_diff_raw"],
        R["cles_fwd_raw"], R["cles_rev_raw"], R["cles_diff_raw"])
    plot_metrics_traces4(outdir / "metrics_R_int.png", "R — per-time metrics (integrated)", R["t_int"],
        R["S_fwd_int"], R["S_rev_int"], R["S_diff_int"],
        R["zrob_fwd_int"], R["zrob_rev_int"], R["zrob_diff_int"],
        R["zfromp_fwd_int"], R["zfromp_rev_int"], R["zfromp_diff_int"],
        R["cles_fwd_int"], R["cles_rev_int"], R["cles_diff_int"])

    # p-value traces
    plot_pvals_traces(outdir / "pvals_C_raw.png", "C — per-time p (raw)",
        C["t_raw"], C["p_fwd_raw"], C["p_rev_raw"], C["p_diff_raw"], args.alpha)
    plot_pvals_traces(outdir / "pvals_C_int.png", "C — per-time p (integrated)",
        C["t_int"], C["p_fwd_int"], C["p_rev_int"], C["p_diff_int"], args.alpha)
    plot_pvals_traces(outdir / "pvals_R_raw.png", "R — per-time p (raw)",
        R["t_raw"], R["p_fwd_raw"], R["p_rev_raw"], R["p_diff_raw"], args.alpha)
    plot_pvals_traces(outdir / "pvals_R_int.png", "R — per-time p (integrated)",
        R["t_int"], R["p_fwd_int"], R["p_rev_int"], R["p_diff_int"], args.alpha)

    # CSV summary with added metrics
    keys = [
        "feature","band_start","band_end","I_fwd","I_rev","I_diff","pID_diff",
        "mean_S_fwd_band","mean_S_rev_band","mean_S_diff_band",
        "mean_zrob_fwd_band","mean_zrob_rev_band","mean_zrob_diff_band",
        "mean_zfromp_fwd_band","mean_zfromp_rev_band","mean_zfromp_diff_band",
        "mean_cles_fwd_band","mean_cles_rev_band","mean_cles_diff_band",
        "frac_sig_diff_band",
        "mean_S_fwd_all","mean_S_rev_all","mean_S_diff_all",
        "mean_zrob_fwd_all","mean_zrob_rev_all","mean_zrob_diff_all",
        "mean_zfromp_fwd_all","mean_zfromp_rev_all","mean_zfromp_diff_all",
        "mean_cles_fwd_all","mean_cles_rev_all","mean_cles_diff_all",
        "frac_sig_diff_all"
    ]
    with open(outdir / "pairdiff_metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerow(dict(feature="C", **C["summary"]))
        w.writerow(dict(feature="R", **R["summary"]))

    print(f"[ok] wrote plots + NPZ + CSV to {outdir}")


if __name__ == "__main__":
    main()
