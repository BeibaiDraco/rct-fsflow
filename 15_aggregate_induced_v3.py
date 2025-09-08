#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_aggregate_induced_v3.py

Group analysis for INDUCED (evoked-removed) results via per-session permutation p-values + meta-analysis.
Also produces bidirectional overlays (A→B and B→A) per feature (C, R).

Usage:
  python 15_aggregate_induced_v3.py --tag induced_k2_win016_p500 --mode both \
      --combine stouffer --annotate_p --p_stride 6
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

BASE = Path(__file__).resolve().parent
SESS_ROOT = BASE / "results" / "session"
OUT_ROOT  = BASE / "results" / "group_induced_pmeta"

VALID_AREAS = {"M": ["MFEF","MLIP","MSC"], "S": ["SFEF","SLIP","SSC"]}

# ---------- utils ----------
def sid_monkey(sid: int) -> str|None:
    s = str(sid)
    if s.startswith("2020"): return "M"
    if s.startswith("2023"): return "S"
    return None

def load_npz_induced(path: Path):
    z = np.load(path, allow_pickle=True)
    rec = {
        "tC": z["tC"] if "tC" in z.files else None,
        "tR": z["tR"] if "tR" in z.files else None,
        # RAW induced forward + null
        "C_raw": z.get("C_fwd", None),        "R_raw": z.get("R_fwd", None),
        "C_null_raw": z.get("C_fnull", None), "R_null_raw": z.get("R_fnull", None),
        # INT (sliding-integrated)
        "C_int": z.get("C_fwd_sl", None),         "R_int": z.get("R_fwd_sl", None),
        "C_null_int": z.get("C_null_sl", None),   "R_null_int": z.get("R_null_sl", None),
        # Scalars
        "IC": float(z["IC_fwd"]) if "IC_fwd" in z.files else np.nan,
        "pIC": float(z["pC_fwd"]) if "pC_fwd" in z.files else np.nan,
        "IR": float(z["IR_fwd"]) if "IR_fwd" in z.files else np.nan,
        "pIR": float(z["pR_fwd"]) if "pR_fwd" in z.files else np.nan,
    }
    return rec

def interp_to(x_src, y_src, x_ref):
    if x_src is None or y_src is None or len(x_src) < 2:
        return np.full_like(x_ref, np.nan, float)
    y = np.asarray(y_src, float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.full_like(x_ref, np.nan, float)
    return np.interp(x_ref, x_src[m], y[m], left=np.nan, right=np.nan)

def mean_ci(y_mat: np.ndarray):
    mu = np.nanmean(y_mat, axis=0)
    n  = np.sum(np.isfinite(y_mat), axis=0).astype(float)
    sd = np.nanstd(y_mat, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(sd), where=(n>0))
    return mu, mu - 1.96*sem, mu + 1.96*sem, n

def p_from_null(obs: np.ndarray, null_mat: np.ndarray) -> np.ndarray:
    T = obs.shape[0]
    p = np.full(T, np.nan, float)
    if null_mat is None or null_mat.size == 0:
        return p
    for t in range(T):
        col = null_mat[:, t]
        m = np.isfinite(col)
        P = int(np.sum(m))
        if P == 0 or not np.isfinite(obs[t]): 
            p[t] = np.nan
        else:
            ge = int(np.sum(col[m] >= obs[t]))
            p[t] = (1 + ge) / (1 + P)
    return p

def combine_p_matrix(P: np.ndarray, method: str="stouffer") -> np.ndarray:
    S, T = P.shape
    pg = np.full(T, np.nan, float)
    eps = 1e-12
    from scipy.stats import norm, chi2
    for t in range(T):
        ps = P[:, t]
        m = np.isfinite(ps)
        k = int(np.sum(m))
        if k == 0:
            continue
        ps_use = np.clip(ps[m], eps, 1.0)
        if method == "fisher":
            X2 = -2.0 * np.sum(np.log(ps_use))
            pg[t] = 1.0 - chi2.cdf(X2, df=2*k)
        else:
            z = norm.ppf(1.0 - ps_use)
            Z = np.sum(z) / math.sqrt(k)
            pg[t] = 1.0 - norm.cdf(Z)
    return pg

def bh_fdr(p: np.ndarray, q=0.05) -> np.ndarray:
    p = np.array(p, float)
    m = np.sum(np.isfinite(p))
    if m == 0: 
        return np.zeros_like(p, bool)
    idx = np.argsort(np.where(np.isfinite(p), p, np.inf))
    ranks = np.arange(1, len(p)+1)
    thr = np.full_like(p, np.nan, float)
    thr[idx] = q * ranks / max(1, m)
    sig = np.isfinite(p) & (p <= thr)
    max_rank = np.max(np.where(sig, ranks, 0))
    return np.isfinite(p) & (p <= (q * max_rank / max(1, m)))

# ---------- plotting ----------
def plot_single_pair(x, mu, lo, hi, p_grp, sig, mk, A, B, label, mode,
                     out_png: Path, title_prefix: str, annotate_p: bool, p_stride: int):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.fill_between(x, lo, hi, color="grey", alpha=0.20, label="95% CI of mean")
    ax.plot(x, mu, lw=2, label=f"Mean forward ({label})")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    if annotate_p:
        for i,(xx,yy,pp) in enumerate(zip(x, mu, p_grp)):
            if (i % max(1, p_stride) == 0) and np.isfinite(pp) and np.isfinite(yy):
                ax.text(xx, yy, f"{pp:.2f}", fontsize=7, ha="center", va="bottom")
    if np.any(sig):
        ax.scatter(x[sig], mu[sig], s=12, color="black", zorder=3, label="BH-FDR<0.05")
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel(("Integrated " if mode=="int" else "") + f"GC ({label})")
    ax.legend(frameon=False)
    ax.set_title(f"{title_prefix} ({mode}) — Monkey {mk}: {A}→{B}")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_overlay_pair(x, res_ab, res_ba, mk, A, B, label, mode, out_png: Path,
                      annotate_p: bool, p_stride: int):
    fig, ax = plt.subplots(figsize=(10,4))
    # A->B
    ax.plot(x, res_ab["mu"], lw=2, label=f"{A}→{B} ({label})")
    ax.fill_between(x, res_ab["lo"], res_ab["hi"], alpha=0.15)
    if np.any(res_ab["sig"]):
        ax.scatter(x[res_ab["sig"]], res_ab["mu"][res_ab["sig"]], s=12, color="black", zorder=3)
    # B->A
    ax.plot(x, res_ba["mu"], lw=2, ls="--", label=f"{B}→{A} ({label})")
    ax.fill_between(x, res_ba["lo"], res_ba["hi"], alpha=0.15)
    if np.any(res_ba["sig"]):
        ax.scatter(x[res_ba["sig"]], res_ba["mu"][res_ba["sig"]], s=12, color="black", zorder=3)
    # p annotations
    if annotate_p:
        yspan = np.nanmax([res_ab["mu"].max(), res_ba["mu"].max()]) - np.nanmin([res_ab["mu"].min(), res_ba["mu"].min()])
        dy = 0.05 * (yspan if np.isfinite(yspan) and yspan > 0 else 1.0)
        for i,xx in enumerate(x):
            if i % max(1, p_stride) != 0: continue
            if np.isfinite(res_ab["p"][i]) and np.isfinite(res_ab["mu"][i]):
                ax.text(xx, res_ab["mu"][i] + dy, f"{res_ab['p'][i]:.2f}", fontsize=7, ha="center", va="bottom")
            if np.isfinite(res_ba["p"][i]) and np.isfinite(res_ba["mu"][i]):
                ax.text(xx, res_ba["mu"][i] - dy, f"{res_ba['p'][i]:.2f}", fontsize=7, ha="center", va="top")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("time (s) from cat_stim_on")
    ax.set_ylabel(("Integrated " if mode=="int" else "") + f"GC ({label})")
    ax.legend(frameon=False)
    ax.set_title(f"Bidirectional overlay ({mode}) — Monkey {mk}: {A}↔{B} ({label})")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------- main aggregation ----------
def aggregate(tag: str, mode: str, out_dir: Path, combine: str,
              annotate_p: bool, p_stride: int):

    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {"M": {}, "S": {}}
    manifests = {"M": [], "S": []}
    scalars   = {"M": [], "S": []}

    for sdir in sorted(SESS_ROOT.glob("*")):
        if not sdir.is_dir() or not sdir.name.isdigit(): 
            continue
        sid = int(sdir.name)
        mk  = sid_monkey(sid)
        if mk is None:
            continue
        sub = sdir / tag
        if not sub.exists():
            continue

        for npz in sorted(sub.glob("induced_flow_*to*.npz")):
            name = npz.stem[len("induced_flow_"):]
            if "to" not in name: 
                continue
            A, B = name.split("to")
            if mk == "M" and (not A.startswith("M") or not B.startswith("M")): 
                continue
            if mk == "S" and (not A.startswith("S") or not B.startswith("S")): 
                continue
            rec = load_npz_induced(npz)
            if mode == "raw":
                tC, Cser, Cnull = rec["tC"], rec["C_raw"], rec["C_null_raw"]
                tR, Rser, Rnull = rec["tR"], rec["R_raw"], rec["R_null_raw"]
            else:
                tC, Cser, Cnull = rec["tC"], rec["C_int"], rec["C_null_int"]
                tR, Rser, Rnull = rec["tR"], rec["R_int"], rec["R_null_int"]

            buckets[mk].setdefault((A,B), []).append((sid, tC, Cser, Cnull, tR, Rser, Rnull))
            manifests[mk].append({"sid": sid, "pair": f"{A}->{B}", "file": npz.name})
            scalars[mk].append({"sid": sid, "pair": f"{A}->{B}",
                                "IC": rec["IC"], "pIC": rec["pIC"], "IR": rec["IR"], "pIR": rec["pIR"]})

    # Save manifests & scalars
    for mk in ("M","S"):
        mdir = out_dir / f"monkey_{mk}"
        mdir.mkdir(parents=True, exist_ok=True)
        if manifests[mk]:
            pd.DataFrame(manifests[mk]).sort_values(["pair","sid"]).to_csv(mdir/"manifest.csv", index=False)
        if scalars[mk]:
            pd.DataFrame(scalars[mk]).sort_values(["pair","sid"]).to_csv(mdir/"band_integrals.csv", index=False)

        store = {"C": {}, "R": {}}

        areas = VALID_AREAS[mk]
        for A in areas:
            for B in areas:
                if A == B: 
                    continue
                recs = buckets[mk].get((A,B), [])
                if not recs:
                    continue

                # reference axis
                x_ref = None
                for _, tC, Cser, *_ in recs:
                    if tC is not None and Cser is not None and len(tC) > 1:
                        x_ref = tC; break
                if x_ref is None:
                    for _, *rest in recs:
                        tR, Rser = rest[3], rest[4]
                        if tR is not None and Rser is not None and len(tR) > 1:
                            x_ref = tR; break
                if x_ref is None: 
                    continue

                # CATEGORY
                C_obs, C_ps = [], []
                for sid, tC, Cser, Cnull, *_ in recs:
                    if Cser is not None and tC is not None and len(tC) > 1:
                        obs_i = interp_to(tC, Cser, x_ref)
                        C_obs.append(obs_i)
                        if Cnull is not None and Cnull.size:
                            rows = np.vstack([interp_to(tC, row, x_ref) for row in Cnull])
                            C_ps.append(p_from_null(obs_i, rows))
                        else:
                            C_ps.append(np.full_like(obs_i, np.nan))
                if C_obs:
                    C_obs = np.vstack(C_obs); C_ps = np.vstack(C_ps)
                    mu, lo, hi, n = mean_ci(C_obs)
                    p_grp = combine_p_matrix(C_ps, method=combine)
                    sig   = bh_fdr(p_grp, q=0.05)
                    out_png = mdir / f"agg_{A}to{B}_C_{mode}.png"
                    plot_single_pair(x_ref, mu, lo, hi, p_grp, sig, mk, A, B, "C", mode,
                                     out_png, "Induced aggregate", annotate_p, p_stride)
                    pd.DataFrame({"t": x_ref, "mu": mu, "lo": lo, "hi": hi,
                                  f"p_{combine}": p_grp, "sig_fdr": sig.astype(int), "n_sessions": n
                                  }).to_csv(mdir / f"agg_{A}to{B}_C_{mode}.csv", index=False)
                    store["C"][f"{A}->{B}"] = {"x": x_ref, "mu": mu, "lo": lo, "hi": hi, "p": p_grp, "sig": sig}

                # DIRECTION
                R_obs, R_ps = [], []
                for sid, *rest in recs:
                    tR, Rser, Rnull = rest[3], rest[4], rest[5]
                    if Rser is not None and tR is not None and len(tR) > 1:
                        obs_i = interp_to(tR, Rser, x_ref)
                        R_obs.append(obs_i)
                        if Rnull is not None and Rnull.size:
                            rows = np.vstack([interp_to(tR, row, x_ref) for row in Rnull])
                            R_ps.append(p_from_null(obs_i, rows))
                        else:
                            R_ps.append(np.full_like(obs_i, np.nan))
                if R_obs:
                    R_obs = np.vstack(R_obs); R_ps = np.vstack(R_ps)
                    mu, lo, hi, n = mean_ci(R_obs)
                    p_grp = combine_p_matrix(R_ps, method=combine)
                    sig   = bh_fdr(p_grp, q=0.05)
                    out_png = mdir / f"agg_{A}to{B}_R_{mode}.png"
                    plot_single_pair(x_ref, mu, lo, hi, p_grp, sig, mk, A, B, "R", mode,
                                     out_png, "Induced aggregate", annotate_p, p_stride)
                    pd.DataFrame({"t": x_ref, "mu": mu, "lo": lo, "hi": hi,
                                  f"p_{combine}": p_grp, "sig_fdr": sig.astype(int), "n_sessions": n
                                  }).to_csv(mdir / f"agg_{A}to{B}_R_{mode}.csv", index=False)
                    store["R"][f"{A}->{B}"] = {"x": x_ref, "mu": mu, "lo": lo, "hi": hi, "p": p_grp, "sig": sig}

        # overlays
        for A in areas:
            for B in areas:
                if A >= B:
                    continue
                for label in ("C","R"):
                    key_ab = f"{A}->{B}"
                    key_ba = f"{B}->{A}"
                    if key_ab in store[label] and key_ba in store[label]:
                        res_ab = store[label][key_ab]
                        res_ba = store[label][key_ba]
                        x = res_ab["x"]
                        if not np.array_equal(res_ba["x"], x):
                            def _interp(res):
                                def safe_interp(y): 
                                    return np.interp(x, res_ba["x"], y, left=np.nan, right=np.nan)
                                return {"mu": safe_interp(res["mu"]),
                                        "lo": safe_interp(res["lo"]),
                                        "hi": safe_interp(res["hi"]),
                                        "p":  safe_interp(res["p"]),
                                        "sig": np.interp(x, res_ba["x"], res["sig"].astype(float),
                                                         left=np.nan, right=np.nan) > 0.5}
                            res_ba = {**res_ba, **_interp(res_ba)}; res_ba["x"] = x
                        out_png = mdir / f"overlay_{label}_{A}_{B}_{mode}.png"
                        plot_overlay_pair(x, res_ab, res_ba, mk, A, B, label, mode, out_png,
                                          annotate_p, p_stride)
                        df = pd.DataFrame({
                            "t": x,
                            "mu_AB": res_ab["mu"], "lo_AB": res_ab["lo"], "hi_AB": res_ab["hi"], f"p_{combine}_AB": res_ab["p"], "sig_AB": res_ab["sig"].astype(int),
                            "mu_BA": res_ba["mu"], "lo_BA": res_ba["lo"], "hi_BA": res_ba["hi"], f"p_{combine}_BA": res_ba["p"], "sig_BA": res_ba["sig"].astype(int),
                        })
                        df.to_csv(mdir / f"overlay_{label}_{A}_{B}_{mode}.csv", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="subfolder beginning with 'induced' (e.g., induced_k2_win016_p500)")
    ap.add_argument("--mode", choices=["raw","int","both"], default="both")
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--combine", choices=["stouffer","fisher"], default="stouffer")
    ap.add_argument("--annotate_p", action="store_true", default=False)
    ap.add_argument("--p_stride", type=int, default=6)
    args = ap.parse_args()

    OUT = Path(args.out_root) / args.tag
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode in ("raw","both"):
        aggregate(args.tag, mode="raw", out_dir=OUT, combine=args.combine,
                  annotate_p=args.annotate_p, p_stride=args.p_stride)
    if args.mode in ("int","both"):
        aggregate(args.tag, mode="int", out_dir=OUT, combine=args.combine,
                  annotate_p=args.annotate_p, p_stride=args.p_stride)

    print(f"[done] Induced p-meta aggregation + overlays ({args.mode}, {args.combine}) → {OUT}")

if __name__ == "__main__":
    main()
