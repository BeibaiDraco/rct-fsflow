#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate per-session pair-difference outputs across sessions to group-level results.

Adds:
  • Group metrics plots with A→B and B→A overlays (mean ± 95% CI) for S, z_robust, z_from_p, CLES
  • Significance dots for DIFF appear only when significant (FDR<α)

Outputs:
  results/group_pairdiff/<TAG>/monkey_<M|S>/...
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, sys, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- utils --------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p
def as1d(a): return np.asarray(a, dtype=float).ravel()

def discover_sids(session_root: Path) -> list[str]:
    sids=[]
    if session_root.exists():
        for d in session_root.iterdir():
            if d.is_dir() and d.name.isdigit() and len(d.name)==8:
                sids.append(d.name)
    return sorted(sids)

def read_sid_list_file(p: Path) -> list[str]:
    out=[]
    with open(p,"r") as f:
        for ln in f:
            s=ln.strip()
            if s and re.fullmatch(r"\d{8}", s): out.append(s)
    return out

def monkey_of_pair(pair: str) -> str: return pair[0]  # 'M' or 'S'
def areas_from_pair(pair: str) -> tuple[str,str]:
    m = re.match(r"([A-Za-z0-9]+)to([A-Za-z0-9]+)$", pair)
    if not m: raise ValueError(f"Bad pair name: {pair}")
    return m.group(1), m.group(2)

def _norm_ppf(p):
    p = np.asarray(p, dtype=float); eps=1e-16
    pp = np.clip(p, eps, 1-eps)
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    plow, phigh=0.02425,0.97575
    q=np.zeros_like(pp); lo=pp<plow; md=(pp>=plow)&(pp<=phigh); hi=pp>phigh
    if np.any(lo):
        xl=np.sqrt(-2*np.log(pp[lo]))
        q[lo]=(((((c[0]*xl+c[1])*xl+c[2])*xl+c[3])*xl+c[4])*xl+c[5])/(((((d[0]*xl+d[1])*xl+d[2])*xl+d[3])*xl)+1)
        q[lo]*=-1
    if np.any(md):
        xl=pp[md]-0.5; r=xl*xl
        q[md]=(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*xl/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    if np.any(hi):
        xl=np.sqrt(-2*np.log(1-pp[hi]))
        q[hi]=(((((c[0]*xl+c[1])*xl+c[2])*xl+c[3])*xl+c[4])*xl+c[5])/(((((d[0]*xl+d[1])*xl+d[2])*xl+d[3])*xl)+1)
    return q

def _norm_sf(z):
    try:
        erfc = np.erfc
    except AttributeError:
        import math; erfc = np.vectorize(math.erfc)
    return 0.5 * erfc(np.asarray(z,dtype=float)/np.sqrt(2.0))

def stouffer_meta_p(pvals_2d: np.ndarray) -> np.ndarray:
    z_i = _norm_ppf(1.0 - np.clip(pvals_2d, 1e-300, 1-1e-16))
    mask = np.isfinite(z_i)
    Z = np.full(z_i.shape[1], np.nan, float)
    for t in range(z_i.shape[1]):
        col=z_i[:,t]; m=mask[:,t]; k=int(np.sum(m))
        if k==0: continue
        Z[t]=np.nansum(col[m])/np.sqrt(k)
    return _norm_sf(Z)

def bh_fdr(p: np.ndarray, alpha: float) -> np.ndarray:
    p=as1d(p); m=np.sum(np.isfinite(p)); mask=np.zeros_like(p,dtype=bool)
    if m==0: return mask
    idx=np.argsort(np.where(np.isfinite(p), p, np.inf))
    p_sorted=p[idx]; thresh=alpha*(np.arange(1,len(p_sorted)+1))/m
    ok=np.where(np.isfinite(p_sorted)&(p_sorted<=thresh))[0]
    if ok.size==0: return mask
    cutoff=p_sorted[ok.max()]
    return np.isfinite(p) & (p<=cutoff)

# ---------------- plotting ----------------
def _ci95(arr2d: np.ndarray):
    m=np.nanmean(arr2d,axis=0)
    n=np.sum(np.isfinite(arr2d),axis=0).astype(float)
    sd=np.nanstd(arr2d,axis=0,ddof=1)
    sem=np.divide(sd,np.sqrt(np.maximum(n,1.0)),out=np.zeros_like(sd),where=n>0)
    lo=m-1.96*sem; hi=m+1.96*sem
    return m,lo,hi,n

def _sigbar_y(series):
    vals=as1d(series); vals=vals[np.isfinite(vals)]
    if vals.size==0: return 0.0
    rng=np.nanmax(vals)-np.nanmin(vals)
    if not np.isfinite(rng) or rng<=0: rng=1e-6
    return float(np.nanmin(vals)-0.05*rng)

def plot_group_overlay_diff(out_png: Path, title: str, t, mean_diff, lo, hi,
                            mean_null, sig_mask, alpha=0.05):
    t=as1d(t); mean_diff=as1d(mean_diff); lo=as1d(lo); hi=as1d(hi); mean_null=as1d(mean_null)
    plt.figure(figsize=(8.6,4.6),dpi=160)
    plt.fill_between(t,lo,hi,color="tab:blue",alpha=0.20,linewidth=0,label="mean±95% CI")
    plt.plot(t,mean_diff,lw=2.4,color="tab:blue",label="mean DIFF")
    plt.plot(t,mean_null,ls="--",lw=1.5,color="tab:gray",alpha=0.9,label="mean null μ (DIFF)")
    plt.axhline(0,color="k",ls=":",lw=1)
    if sig_mask is not None and sig_mask.size==t.size:
        sig = sig_mask.astype(bool)
        if np.any(sig):
            ybar = _sigbar_y(mean_diff)
            plt.plot(t[sig], np.full(sig.sum(), ybar), ".", ms=6, color="k", label="FDR<α")
    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("Flow (DIFF)")
    plt.legend(frameon=False,ncol=3); plt.tight_layout(); plt.savefig(out_png); plt.close()

def _overlay_ci(ax, t, arr2d, color, label):
    m,lo,hi,_=_ci95(arr2d)
    ax.fill_between(t, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.plot(t, m, lw=2.0, color=color, label=label)

def plot_group_metrics4_overlay_bidir(out_png: Path, title: str, t,
                                      S_fwd, S_rev, zrob_fwd, zrob_rev, zfp_fwd, zfp_rev, cles_fwd, cles_rev,
                                      label_fwd: str, label_rev: str):
    t=as1d(t)
    fig,axes=plt.subplots(4,1,figsize=(8.6,10.0),dpi=160,sharex=True)
    # S bits
    ax=axes[0]; _overlay_ci(ax,t,S_fwd,"tab:blue",f"S {label_fwd}")
    _overlay_ci(ax,t,S_rev,"tab:orange",f"S {label_rev}")
    ax.set_ylabel("S bits"); ax.grid(alpha=0.25); ax.legend(frameon=False,ncol=2)
    # robust z
    ax=axes[1]; _overlay_ci(ax,t,zrob_fwd,"tab:blue",f"z_rob {label_fwd}")
    _overlay_ci(ax,t,zrob_rev,"tab:orange",f"z_rob {label_rev}")
    ax.set_ylabel("z_rob"); ax.grid(alpha=0.25); ax.legend(frameon=False,ncol=2)
    # z_from_p
    ax=axes[2]; _overlay_ci(ax,t,zfp_fwd,"tab:blue",f"z_from_p {label_fwd}")
    _overlay_ci(ax,t,zfp_rev,"tab:orange",f"z_from_p {label_rev}")
    ax.set_ylabel("z_from_p"); ax.grid(alpha=0.25); ax.legend(frameon=False,ncol=2)
    # CLES
    ax=axes[3]; _overlay_ci(ax,t,cles_fwd,"tab:blue",f"CLES {label_fwd}")
    _overlay_ci(ax,t,cles_rev,"tab:orange",f"CLES {label_rev}")
    ax.set_ylabel("CLES (1−p)"); ax.set_ylim(0,1); ax.grid(alpha=0.25); ax.legend(frameon=False,ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title,y=0.995); fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# ---------------- aggregation core ----------------
def collect_pairs_for_tag(session_root: Path, sid: str, tag: str) -> list[str]:
    base = session_root / sid / tag / "pairdiff"
    if not base.exists(): return []
    man = base / "pairdiff_manifest.json"
    pairs=[]
    if man.exists():
        j = json.loads(Path(man).read_text())
        for rec in j.get("pairs", []):
            if rec.get("status") == "ok": pairs.append(rec["pair"])
        return sorted(set(pairs))
    for d in base.iterdir():
        if d.is_dir() and re.fullmatch(r"[A-Za-z0-9]+to[A-Za-z0-9]+", d.name):
            pairs.append(d.name)
    return sorted(set(pairs))

def load_pair_npz(session_root: Path, sid: str, tag: str, pair: str):
    npz = session_root / sid / tag / "pairdiff" / pair / "pairdiff_timeseries_metrics.npz"
    if not npz.exists(): return None
    return np.load(npz, allow_pickle=True)

def aggregate_one_pair(session_root: Path, sids: list[str], tag: str, pair: str,
                       feature: str, series: str, alpha: float):
    """
    feature in {'C','R'}, series in {'raw','int'}
    Returns dict with aggregated arrays and stats, or None if <2 sessions available.
    """
    diffs=[]; null_mu=[]; pvals=[]; used_sids=[]

    S_fwd=[]; S_rev=[]; zrob_fwd=[]; zrob_rev=[]; zfp_fwd=[]; zfp_rev=[]; cles_fwd=[]; cles_rev=[]
    p_fwd=[]; p_rev=[]

    t_ref=None
    for sid in sids:
        z = load_pair_npz(session_root, sid, tag, pair)
        if z is None: continue
        D = z[feature].item()
        t = D[f"t_{series}"]
        if t_ref is None:
            t_ref = t
        else:
            if not (t.size==t_ref.size and np.allclose(t,t_ref,atol=1e-9)):
                print(f"[warn] t mismatch for {sid} {tag} {pair} {feature}/{series}; skipping this session.")
                continue

        diffs.append(as1d(D[f"diff_{series}"]))
        null_mu.append(as1d(D[f"dnull_mu_{series}"]))
        pvals.append(as1d(D[f"p_diff_{series}"]))
        used_sids.append(sid)

        # forward/reverse metrics (for overlay)
        S_fwd.append(as1d(D[f"S_fwd_{series}"]))
        S_rev.append(as1d(D[f"S_rev_{series}"]))
        zrob_fwd.append(as1d(D[f"zrob_fwd_{series}"]))
        zrob_rev.append(as1d(D[f"zrob_rev_{series}"]))
        zfp_fwd.append(as1d(D[f"zfromp_fwd_{series}"]))
        zfp_rev.append(as1d(D[f"zfromp_rev_{series}"]))
        cles_fwd.append(as1d(D[f"cles_fwd_{series}"]))
        cles_rev.append(as1d(D[f"cles_rev_{series}"]))
        p_fwd.append(as1d(D[f"p_fwd_{series}"]))
        p_rev.append(as1d(D[f"p_rev_{series}"]))

    if len(diffs) < 2:
        return None

    diffs=np.vstack(diffs); null_mu=np.vstack(null_mu); pvals=np.vstack(pvals)
    S_fwd=np.vstack(S_fwd); S_rev=np.vstack(S_rev)
    zrob_fwd=np.vstack(zrob_fwd); zrob_rev=np.vstack(zrob_rev)
    zfp_fwd=np.vstack(zfp_fwd); zfp_rev=np.vstack(zfp_rev)
    cles_fwd=np.vstack(cles_fwd); cles_rev=np.vstack(cles_rev)
    p_fwd=np.vstack(p_fwd); p_rev=np.vstack(p_rev)

    mean_diff,lo,hi,n=_ci95(diffs)
    mean_null=np.nanmean(null_mu,axis=0)

    # meta p for DIFF; FDR across time
    p_meta = stouffer_meta_p(pvals)
    fdr_mask = bh_fdr(p_meta, alpha=alpha)
    frac_sig = np.nanmean(pvals < alpha, axis=0)

    return dict(
        t=t_ref, mean_diff=mean_diff, lo=lo, hi=hi,
        mean_null=mean_null, p_meta=p_meta, fdr_mask=fdr_mask,
        n_sessions_used=int(np.nanmax(n)), frac_sig=frac_sig, used_sids=used_sids,
        # forward/reverse 2D arrays for overlay plots (mean±CI)
        S_fwd=S_fwd, S_rev=S_rev,
        zrob_fwd=zrob_fwd, zrob_rev=zrob_rev,
        zfp_fwd=zfp_fwd, zfp_rev=zfp_rev,
        cles_fwd=cles_fwd, cles_rev=cles_rev,
        p_fwd=p_fwd, p_rev=p_rev
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="Tag folders to aggregate")
    ap.add_argument("--session-root", default="results/session")
    ap.add_argument("--out-root", default="results/group_pairdiff")
    ap.add_argument("--sid-list-file", default="")
    ap.add_argument("--sids", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    session_root=Path(args.session_root); out_root=Path(args.out_root)
    if args.sids: sids=args.sids
    elif args.sid_list_file: sids=read_sid_list_file(Path(args.sid_list_file))
    else: sids=discover_sids(session_root)
    if not sids: print("[FATAL] No sessions to aggregate.", file=sys.stderr); sys.exit(2)

    print(f"[info] Aggregating {len(sids)} sessions")
    print(f"[info] Tags: {' '.join(args.tags)}")

    for tag in args.tags:
        # collect all pairs seen in this tag
        pair_set=set()
        for sid in sids:
            for pair in collect_pairs_for_tag(session_root, sid, tag):
                pair_set.add(pair)

        pairs_by_monkey={"M":[],"S":[]}
        for pair in sorted(pair_set):
            mk=monkey_of_pair(pair)
            if mk in pairs_by_monkey: pairs_by_monkey[mk].append(pair)

        for mk,pairs in pairs_by_monkey.items():
            if not pairs: continue
            out_dir=ensure_dir(out_root / tag / f"monkey_{mk}")
            print(f"[info] Tag {tag}  Monkey {mk}: {len(pairs)} pairs")

            for pair in pairs:
                A,B=areas_from_pair(pair); base=f"{A}_{B}"
                lab_fwd=f"{A}→{B}"; lab_rev=f"{B}→{A}"

                for feat in ("C","R"):
                    for series in ("raw","int"):
                        agg = aggregate_one_pair(session_root, sids, tag, pair, feat, series, args.alpha)
                        if agg is None:
                            print(f"[warn] insufficient sessions for {pair} {feat}/{series}; skipping.")
                            continue
                        t=agg["t"]

                        # DIFF overlay with FDR significance dots (only when significant)
                        plot_group_overlay_diff(
                            out_dir / f"overlay_DIFF_{feat}_{base}_{series}.png",
                            f"{feat} {series} — DIFF {A}−{B} (N={agg['n_sessions_used']})",
                            t, agg["mean_diff"], agg["lo"], agg["hi"], agg["mean_null"],
                            agg["fdr_mask"], alpha=args.alpha
                        )

                        # NEW: A→B vs B→A overlay for metrics (mean±CI)
                        plot_group_metrics4_overlay_bidir(
                            out_dir / f"metrics_BIDIR_{feat}_{base}_{series}.png",
                            f"{feat} {series} — how-far metrics (mean±95% CI) — {A}↔{B}",
                            t,
                            agg["S_fwd"], agg["S_rev"],
                            agg["zrob_fwd"], agg["zrob_rev"],
                            agg["zfp_fwd"], agg["zfp_rev"],
                            agg["cles_fwd"], agg["cles_rev"],
                            lab_fwd, lab_rev
                        )

                        # Meta-p trace for DIFF
                        # (you can optionally add separate meta-p for fwd/rev later if needed)
                        plt_path = out_dir / f"pvals_DIFF_{feat}_{base}_{series}.png"
                        _plot_pvals_trace(plt_path, f"{feat} {series} — meta p (Stouffer) — DIFF {A}−{B}", t, agg["p_meta"], args.alpha)

                        # CSV: group DIFF summary per time
                        csv_path = out_dir / f"group_DIFF_{feat}_{base}_{series}.csv"
                        with open(csv_path,"w",newline="") as f:
                            w=csv.writer(f)
                            w.writerow(["t","mean_diff","ci_lo","ci_hi","null_mu_mean","meta_p","fdr_sig","frac_sig_sessions"])
                            for i in range(t.size):
                                w.writerow([
                                    float(t[i]), float(agg["mean_diff"][i]), float(agg["lo"][i]), float(agg["hi"][i]),
                                    float(agg["mean_null"][i]),
                                    float(agg["p_meta"][i]) if np.isfinite(agg["p_meta"][i]) else "",
                                    int(agg["fdr_mask"][i]) if i<agg["fdr_mask"].size else 0,
                                    float(agg["frac_sig"][i])
                                ])

            with open(out_dir / "pairdiff_group_manifest.json","w") as f:
                json.dump(dict(tag=tag,monkey=mk,pairs=pairs),f,indent=2)

    print("[ok] aggregation complete.")

def _plot_pvals_trace(out_png: Path, title: str, t, p_meta, alpha: float):
    t=as1d(t)
    plt.figure(figsize=(8.6,4.0),dpi=160)
    tolog=lambda p: -np.log10(np.maximum(as1d(p),1e-300))
    plt.plot(t, tolog(p_meta), lw=2.0, label="-log10 meta p (DIFF)")
    plt.axhline(-np.log10(alpha), color="k", ls=":", lw=1, label=f"α={alpha}")
    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("-log10 p")
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(out_png); plt.close()

if __name__=="__main__":
    main()
