#!/usr/bin/env python3
"""
Verify outputs, aggregate across sessions, and make over-session 3-panel plots,
STRICTLY per monkey (M vs S; never combined).

Monkeys:
- sid startswith "2020" -> M, regions: MFEF, MLIP, MSC
- sid startswith "2023" -> S, regions: SFEF, SLIP, SSC

For each (align, feature, monkey, pair A->B) produce one figure with 3 vertical panels:
(a) mean flow A->B and B->A over sessions (+ mean null for each direction)
(b) "evidence strength" S = -log2 p for A->B and B->A (mean ± SEM)
(c) mean (A->B - B->A) ± SEM, with black dots at time bins significant over sessions
    using an exact two-sided sign test across sessions on standardized Δ, with BH-FDR.

Also writes a tidy CSV: out/analysis/summary_flows_over_sessions.csv
"""

import argparse, json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- config: monkey & regions ----------------
REGIONS_M = ["MFEF", "MLIP", "MSC"]
REGIONS_S = ["SFEF", "SLIP", "SSC"]

def monkey_from_sid(sid: str) -> str:
    if str(sid).startswith("2020"): return "M"
    if str(sid).startswith("2023"): return "S"
    return "unknown"

def regions_for_monkey(monkey: str) -> List[str]:
    return REGIONS_M if monkey == "M" else (REGIONS_S if monkey == "S" else [])

# ---------------- filesystem helpers ----------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def list_sessions(out_root: Path, align: str) -> List[str]:
    base = out_root / align
    if not base.exists(): return []
    return sorted([p.name for p in base.iterdir()
                   if p.is_dir() and (p / "caches").is_dir()])

def dir_pair(out_root: Path, align: str, sid: str, tag: str, feat: str) -> Path:
    return out_root / align / sid / "flow" / tag / feat

def list_pair_files(out_root: Path, align: str, sid: str, tag: str, feat: str) -> List[Path]:
    d = dir_pair(out_root, align, sid, tag, feat)
    if not d.exists(): return []
    return sorted(d.glob(f"flow_{feat}_*to*.npz"))

def parse_pair_from_stem(stem: str) -> Tuple[str, str]:
    # flow_<feat>_<A>to<B>
    s = stem.split("_")[-1]
    A, B = s.split("to")
    return A, B

# ---------------- NaN-safe means, SEM, and FDR ----------------
def nanmean_no_warn(a: np.ndarray, axis=0):
    a = np.asarray(a, float)
    valid = np.isfinite(a)
    n = valid.sum(axis=axis)
    s = np.nansum(np.where(valid, a, 0.0), axis=axis)
    return np.divide(s, np.maximum(n, 1), out=np.full(s.shape, np.nan), where=n>0)

def nanmean_sem(a: np.ndarray, axis=0):
    """Mean and SEM ignoring NaNs; SEM=NaN where n<2."""
    a = np.asarray(a, float)
    valid = np.isfinite(a)
    n = valid.sum(axis=axis)

    s = np.nansum(np.where(valid, a, 0.0), axis=axis)
    mu = np.divide(s, np.maximum(n, 1), out=np.full(s.shape, np.nan), where=n>0)

    ss = np.nansum(np.where(valid, a*a, 0.0), axis=axis)
    var_num = ss - (s**2) / np.maximum(n, 1)
    var = np.divide(var_num, np.maximum(n-1, 1), out=np.full(s.shape, np.nan), where=n>1)
    sd = np.sqrt(var)
    sem = np.divide(sd, np.sqrt(np.maximum(n, 1)), out=np.full(s.shape, np.nan), where=n>1)
    return mu, sem

def bh_significant(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini–Hochberg across time.
    Returns a boolean mask over the ORIGINAL length of p.
    """
    p = np.asarray(p, float)
    sig = np.zeros_like(p, dtype=bool)

    finite_idx = np.where(np.isfinite(p))[0]
    m = finite_idx.size
    if m == 0:
        return sig

    p_fin = p[finite_idx]
    order = np.argsort(p_fin)
    p_sorted = p_fin[order]                 # length m
    thresh = alpha * (np.arange(1, m+1) / m)

    passed = np.where(p_sorted <= thresh)[0]
    if passed.size:
        k = passed.max()
        cutoff = p_sorted[k]
        sig_fin = (p_fin <= cutoff)
        sig[finite_idx] = sig_fin
    return sig

# ---------------- per-file loader ----------------
def load_pair_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Load one file flow_<feat>_<A>to<B>.npz, which (in your flow.py) contains both directions:
    bits_AtoB, bits_BtoA, null_mean_* and p_* for each direction.
    """
    z = np.load(npz_path, allow_pickle=True)
    out = dict(
        time = np.asarray(z["time"], float),
        fwd  = np.asarray(z["bits_AtoB"], float),  # A->B
        rev  = np.asarray(z["bits_BtoA"], float),  # B->A
        mu_fwd = np.asarray(z.get("null_mean_AtoB", np.full_like(z["bits_AtoB"], np.nan)), float),
        mu_rev = np.asarray(z.get("null_mean_BtoA", np.full_like(z["bits_BtoA"], np.nan)), float),
        sd_fwd = np.asarray(z.get("null_std_AtoB", np.full_like(z["bits_AtoB"], np.nan)), float),
        sd_rev = np.asarray(z.get("null_std_BtoA", np.full_like(z["bits_BtoA"], np.nan)), float),
        p_fwd  = np.asarray(z.get("p_AtoB", np.full_like(z["bits_AtoB"], np.nan)), float),
        p_rev  = np.asarray(z.get("p_BtoA", np.full_like(z["bits_BtoA"], np.nan)), float),
    )
    return out

def pick_common_time(times: List[np.ndarray]) -> Optional[np.ndarray]:
    """Pick the most frequent time grid length; return a representative vector."""
    if not times: return None
    lens = [len(t) for t in times]
    if not lens: return None
    unique, counts = np.unique(lens, return_counts=True)
    L = int(unique[np.argmax(counts)])
    for t in times:
        if len(t) == L:
            return t
    return None

# ---------------- plots ----------------
def plot_three_panel(out_png: Path,
                     time_ms: np.ndarray,
                     fwd_mean: np.ndarray, fwd_sem: np.ndarray,
                     rev_mean: np.ndarray, rev_sem: np.ndarray,
                     mu_fwd_mean: np.ndarray, mu_rev_mean: np.ndarray,
                     S_fwd_mean: np.ndarray, S_fwd_sem: np.ndarray,
                     S_rev_mean: np.ndarray, S_rev_sem: np.ndarray,
                     diff_mean: np.ndarray, diff_sem: np.ndarray,
                     sig_mask: np.ndarray,
                     title: str,
                     A_name: str,
                     B_name: str):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # (a) flows
    ax = axes[0]
    ax.plot(time_ms, fwd_mean, lw=2, label=f"{A_name}→{B_name}")
    ax.fill_between(time_ms, fwd_mean - fwd_sem, fwd_mean + fwd_sem, alpha=0.2, edgecolor="none")
    ax.plot(time_ms, rev_mean, lw=2, ls="--", color="#d98400", label=f"{B_name}→{A_name}")
    ax.fill_between(time_ms, rev_mean - rev_sem, rev_mean + rev_sem, alpha=0.2, edgecolor="none", color="#d98400")
    ax.plot(time_ms, mu_fwd_mean, lw=1, color="#555", alpha=0.85, label=f"null μ ({A_name}→{B_name})")
    ax.plot(time_ms, mu_rev_mean, lw=1, color="#999", alpha=0.85, ls="--", label=f"null μ ({B_name}→{A_name})")
    ax.axhline(0, lw=1, ls=":", color="#666")
    ax.set_ylabel("Flow (bits)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2, linestyle=":")

    # (b) S bits
    ax = axes[1]
    ax.plot(time_ms, S_fwd_mean, lw=2, label=f"S({A_name}→{B_name})")
    ax.fill_between(time_ms, S_fwd_mean - S_fwd_sem, S_fwd_mean + S_fwd_sem, alpha=0.2, edgecolor="none")
    ax.plot(time_ms, S_rev_mean, lw=2, ls="--", color="#d98400", label=f"S({B_name}→{A_name})")
    ax.fill_between(time_ms, S_rev_mean - S_rev_sem, S_rev_mean + S_rev_sem, alpha=0.2, edgecolor="none", color="#d98400")
    ax.axhline(0, lw=1, ls=":", color="#666")
    ax.set_ylabel("Evidence S = -log2 p")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2, linestyle=":")

    # (c) Δ and significance
    ax = axes[2]
    ax.plot(time_ms, diff_mean, lw=2, color="k", label=f"({A_name}→{B_name}) − ({B_name}→{A_name})")
    ax.fill_between(time_ms, diff_mean - diff_sem, diff_mean + diff_sem, alpha=0.2, edgecolor="none", color="k")
    ax.axhline(0, lw=1, ls=":", color="#666")
    # sig dots at bottom
    ymin = np.nanmin(diff_mean - diff_sem) if np.any(np.isfinite(diff_mean)) else 0.0
    ydot = ymin - 0.05 * (np.nanmax(np.abs(diff_mean) + np.nan_to_num(diff_sem, nan=0.0)) + 1e-9)
    xs = time_ms[sig_mask]
    if xs.size:
        ax.scatter(xs, np.full_like(xs, ydot), s=10, color="k", zorder=3, clip_on=False)
    ax.set_ylabel("ΔFlow (bits)")
    ax.set_xlabel("Time (ms)")
    ax.grid(alpha=0.2, linestyle=":")

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--tag", default="crnull-vertical")
    ap.add_argument("--align", default="stim", choices=["stim","sacc"])
    ap.add_argument("--features", nargs="+", default=["C","R"],
                    help="For stim: use C R; for sacc: use S")
    ap.add_argument("--alpha", type=float, default=0.05, help="BH-FDR across time for panel (c)")
    ap.add_argument("--csv_out", default="summary_flows_over_sessions.csv")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    plots_root = out_root / "analysis" / f"over_session_plots_{args.tag}"
    ensure_dir(plots_root)

    sids = list_sessions(out_root, args.align)
    sids_M = [sid for sid in sids if monkey_from_sid(sid) == "M"]
    sids_S = [sid for sid in sids if monkey_from_sid(sid) == "S"]
    print(f"[info] sessions: M={len(sids_M)}, S={len(sids_S)}")

    rows = []

    for feat in args.features:
        for monkey, sids_this in (("M", sids_M), ("S", sids_S)):
            if not sids_this: continue
            regs = regions_for_monkey(monkey)
            if not regs: continue

            # ordered pairs in this monkey's region set
            pairs = [(A,B) for A in regs for B in regs if A != B]

            for (A,B) in pairs:
                # collect per-session series (one file has both directions)
                times_list = []
                fwd_list, rev_list = [], []
                mu_fwd_list, mu_rev_list = [], []
                sd_fwd_list, sd_rev_list = [], []
                S_fwd_list, S_rev_list = [], []
                ok_sids = []

                for sid in sids_this:
                    fdir = dir_pair(out_root, args.align, sid, args.tag, feat)
                    npz = fdir / f"flow_{feat}_{A}to{B}.npz"
                    if not npz.exists():  # require this pair file
                        continue
                    try:
                        D = load_pair_npz(npz)
                    except Exception as e:
                        warnings.warn(f"skip {npz}: {e}")
                        continue
                    if D["time"].size == 0:
                        continue
                    times_list.append(D["time"])
                    fwd_list.append(D["fwd"])
                    rev_list.append(D["rev"])
                    mu_fwd_list.append(D["mu_fwd"])
                    mu_rev_list.append(D["mu_rev"])
                    sd_fwd_list.append(D["sd_fwd"])
                    sd_rev_list.append(D["sd_rev"])
                    S_fwd_list.append(-np.log2(np.maximum(D["p_fwd"], 1e-300)))
                    S_rev_list.append(-np.log2(np.maximum(D["p_rev"], 1e-300)))
                    ok_sids.append(sid)

                if len(ok_sids) < 2:
                    continue

                # choose common time grid and keep matching sessions
                t_ref = pick_common_time(times_list)
                if t_ref is None: continue
                keep = [i for i,t in enumerate(times_list) if (len(t)==len(t_ref) and np.allclose(t,t_ref,atol=1e-12))]
                if len(keep) < 2:
                    continue

                fwd_mat = np.vstack([fwd_list[i] for i in keep])
                rev_mat = np.vstack([rev_list[i] for i in keep])
                mu_fwd_mat = np.vstack([mu_fwd_list[i] for i in keep])
                mu_rev_mat = np.vstack([mu_rev_list[i] for i in keep])
                sd_fwd_mat = np.vstack([sd_fwd_list[i] for i in keep])
                sd_rev_mat = np.vstack([sd_rev_list[i] for i in keep])
                S_fwd_mat = np.vstack([S_fwd_list[i] for i in keep])
                S_rev_mat = np.vstack([S_rev_list[i] for i in keep])
                used_sids = [ok_sids[i] for i in keep]

                # (a) means/SEMs
                fwd_mean, fwd_sem = nanmean_sem(fwd_mat, axis=0)
                rev_mean, rev_sem = nanmean_sem(rev_mat, axis=0)
                mu_fwd_mean = nanmean_no_warn(mu_fwd_mat, axis=0)
                mu_rev_mean = nanmean_no_warn(mu_rev_mat, axis=0)

                # (b) S bits
                S_fwd_mean, S_fwd_sem = nanmean_sem(S_fwd_mat, axis=0)
                S_rev_mean, S_rev_sem = nanmean_sem(S_rev_mat, axis=0)

                # (c) Δ and significance (sign test on standardized Δ)
                diff_mat = fwd_mat - rev_mat
                diff_mean, diff_sem = nanmean_sem(diff_mat, axis=0)

                mu_diff_mat = mu_fwd_mat - mu_rev_mat
                sd_diff_mat = np.sqrt(np.maximum(sd_fwd_mat,0.0)**2 + np.maximum(sd_rev_mat,0.0)**2)
                sd_diff_mat = np.where(sd_diff_mat <= 1e-12, np.nan, sd_diff_mat)
                Z_mat = (diff_mat - mu_diff_mat) / sd_diff_mat  # standardized Δ

                # exact two-sided sign test per time
                valid = np.isfinite(Z_mat)
                n_t = valid.sum(axis=0)
                k_t = (Z_mat > 0).sum(axis=0)

                from math import comb
                def binom_cdf(k, n):
                    return sum(comb(n, i) for i in range(0, k+1)) * (0.5 ** n)
                def binom_two_sided_p(k, n):
                    if n < 2: return np.nan
                    p_lo = binom_cdf(k, n)
                    p_hi = binom_cdf(n-k, n)
                    return min(1.0, 2.0 * min(p_lo, p_hi))

                p_over = np.array([binom_two_sided_p(int(k_t[j]), int(n_t[j])) if n_t[j] >= 2 else np.nan
                                   for j in range(Z_mat.shape[1])])
                sig_mask = bh_significant(p_over, alpha=args.alpha)

                # write rows for summary CSV
                for i, sid in enumerate(used_sids):
                    rows.append(dict(
                        sid=sid, monkey=monkey, align=args.align, feature=feat, A=A, B=B,
                        mean_flow_fwd=float(np.nanmean(fwd_mat[i])),
                        mean_flow_rev=float(np.nanmean(rev_mat[i])),
                        mean_S_fwd=float(np.nanmean(S_fwd_mat[i])),
                        mean_S_rev=float(np.nanmean(S_rev_mat[i])),
                        mean_diff=float(np.nanmean(diff_mat[i])),
                        n_time=int(len(t_ref))
                    ))

                # skip empty figures
                if (not np.any(np.isfinite(fwd_mean))) and (not np.any(np.isfinite(rev_mean))):
                    continue

                time_ms = t_ref * 1000.0
                title = f"{args.align.upper()} · {feat} · {A}→{B} · monkey {monkey} (n={len(used_sids)})"
                out_png = plots_root / args.align / feat / monkey / f"{A}to{B}.png"
                plot_three_panel(out_png,
                                 time_ms,
                                 fwd_mean, fwd_sem,
                                 rev_mean, rev_sem,
                                 mu_fwd_mean, mu_rev_mean,
                                 S_fwd_mean, S_fwd_sem,
                                 S_rev_mean, S_rev_sem,
                                 diff_mean, diff_sem,
                                 sig_mask,
                                 title, A_name=A, B_name=B)
                print(f"[ok] {out_png}")

    df = pd.DataFrame(rows).sort_values(["align","feature","monkey","A","B","sid"])
    out_csv = out_root / "analysis" / args.csv_out
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv} ({len(df)} rows)")
    print("[done] verification + aggregation + per-monkey plots complete.")

if __name__ == "__main__":
    main()
