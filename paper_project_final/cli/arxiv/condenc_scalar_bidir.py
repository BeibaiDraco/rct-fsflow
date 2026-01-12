#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def parse_window(s: str) -> Tuple[float,float]:
    a,b = s.split(":"); return float(a), float(b)

def load_npz(p: Path) -> Dict:
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta" in out and not isinstance(out["meta"], dict):
        try: out["meta"] = json.loads(out["meta"].item())
        except Exception: pass
    return out

def cache_path(out_root: Path, align: str, sid: str, area: str) -> Path:
    return out_root/align/sid/"caches"/f"area_{area}.npz"

def axis_path(out_root: Path, align: str, sid: str, axes_tag: str, area: str) -> Path:
    return out_root/align/sid/"axes"/axes_tag/f"axes_{area}.npz"

def list_areas(cache_dir: Path):
    return sorted([p.name[5:-4] for p in cache_dir.glob("area_*.npz")])

def pick_area(areas, key):
    hits = [a for a in areas if key.upper() in a.upper()]
    if not hits: raise SystemExit(f"No {key} area in {areas}")
    return hits[0]

def trial_mask(cache: Dict, orientation: str, pt_min_ms: float):
    N = cache["Z"].shape[0]
    keep = np.ones(N, dtype=bool)
    keep &= cache.get("lab_is_correct", np.ones(N, bool)).astype(bool)
    if orientation != "pooled" and "lab_orientation" in cache:
        keep &= (cache["lab_orientation"].astype(str) == orientation)
    if pt_min_ms is not None and "lab_PT_ms" in cache:
        PT = cache["lab_PT_ms"].astype(float)
        keep &= np.isfinite(PT) & (PT >= float(pt_min_ms))
    keep &= np.isfinite(cache.get("lab_C", np.full(N, np.nan)).astype(float))
    keep &= np.isfinite(cache.get("lab_R", np.full(N, np.nan)).astype(float))
    return keep

def project_1d(cache: Dict, s: np.ndarray, keep: np.ndarray):
    Z = cache["Z"][keep].astype(float)  # (N,B,U)
    s = np.asarray(s, float).reshape(-1)
    return np.tensordot(Z, s, axes=([2],[0]))  # (N,B)

def encode_CR(cache: Dict, keep: np.ndarray):
    C = np.round(cache["lab_C"][keep].astype(float)).astype(int)
    R = np.round(cache["lab_R"][keep].astype(float)).astype(int)
    return (C*1000 + R).astype(int)

def split_within_strata(d: np.ndarray, strata: np.ndarray, min_per: int):
    low=[]; high=[]
    for v in np.unique(strata):
        idx = np.where(strata==v)[0]
        if idx.size < min_per: continue
        order = np.argsort(d[idx])
        h = idx.size//2
        if h==0: continue
        low.append(idx[order[:h]])
        high.append(idx[order[-h:]])
    if not low or not high: return np.array([],int), np.array([],int)
    return np.concatenate(low), np.concatenate(high)

def auc(scores, Cpm1):
    y=(Cpm1>0).astype(int)
    if np.unique(y).size<2: return np.nan
    return float(roc_auc_score(y, scores))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical")
    ap.add_argument("--pt_min_ms", type=float, default=200.0)
    ap.add_argument("--axes_tag", default="axes_sweep-stim-pooled",
                    help="Use pooled C axes by default")
    ap.add_argument("--src", default="FEF", help="Source area key: FEF or LIP or SC")
    ap.add_argument("--tgt", default="LIP", help="Target area key: FEF or LIP or SC")
    ap.add_argument("--win_src", default="0.12:0.20")
    ap.add_argument("--win_tgt", default="0.22:0.35")
    ap.add_argument("--min_per_stratum", type=int, default=6)
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="condenc_scalar_v1")
    args=ap.parse_args()

    out_root=Path(args.out_root); sid=args.sid
    align="stim"
    win_src=parse_window(args.win_src)
    win_tgt=parse_window(args.win_tgt)

    areas=list_areas(out_root/align/sid/"caches")
    A=pick_area(areas, args.src)
    B=pick_area(areas, args.tgt)

    cacheA=load_npz(cache_path(out_root,align,sid,A))
    cacheB=load_npz(cache_path(out_root,align,sid,B))
    keep = trial_mask(cacheA,args.orientation,args.pt_min_ms) & trial_mask(cacheB,args.orientation,args.pt_min_ms)
    if keep.sum() < 80: raise SystemExit(f"Too few trials after filtering: N={keep.sum()}")

    axesA=load_npz(axis_path(out_root,align,sid,args.axes_tag,A))
    axesB=load_npz(axis_path(out_root,align,sid,args.axes_tag,B))
    sA=axesA.get("sC",np.array([])).ravel()
    sB=axesB.get("sC",np.array([])).ravel()
    if sA.size==0 or sB.size==0: raise SystemExit("Missing sC axis")

    time=cacheA["time"].astype(float)
    YA=project_1d(cacheA,sA,keep)  # (N,B)
    YB=project_1d(cacheB,sB,keep)  # (N,B)
    C=cacheA["lab_C"][keep].astype(float)

    mA = (time>=win_src[0]) & (time<=win_src[1])
    mB = (time>=win_tgt[0]) & (time<=win_tgt[1])
    if not mA.any() or not mB.any(): raise SystemExit("Window has no bins")

    dA = np.nanmean(YA[:,mA], axis=1)
    yB = np.nanmean(YB[:,mB], axis=1)  # target late scalar

    strata = encode_CR(cacheA, keep)
    low, high = split_within_strata(dA, strata, args.min_per_stratum)
    if low.size<30 or high.size<30: raise SystemExit(f"Too few after split low={low.size} high={high.size}")

    # signal-level metrics on target
    EB = C * yB  # signed evidence
    delta_E = float(np.nanmean(EB[high]) - np.nanmean(EB[low]))
    delta_AUC = float(auc(yB[high], C[high]) - auc(yB[low], C[low]))

    # permutation null
    rng=np.random.default_rng(args.seed)
    deltas_E=np.full(args.perms, np.nan)
    deltas_A=np.full(args.perms, np.nan)
    stratum_to_idx={v:np.where(strata==v)[0] for v in np.unique(strata)}

    for p in range(args.perms):
        low_p=[]; high_p=[]
        for v, idx in stratum_to_idx.items():
            if idx.size < args.min_per_stratum: continue
            perm=rng.permutation(idx)
            h=idx.size//2
            if h==0: continue
            low_p.append(perm[:h]); high_p.append(perm[-h:])
        if not low_p or not high_p: continue
        low_p=np.concatenate(low_p); high_p=np.concatenate(high_p)
        deltas_E[p] = np.nanmean(EB[high_p]) - np.nanmean(EB[low_p])
        deltas_A[p] = auc(yB[high_p], C[high_p]) - auc(yB[low_p], C[low_p])

    muE, sdE = np.nanmean(deltas_E), np.nanstd(deltas_E, ddof=1)
    muA, sdA = np.nanmean(deltas_A), np.nanstd(deltas_A, ddof=1)
    zE = (delta_E - muE)/sdE if sdE>0 else np.nan
    zA = (delta_AUC - muA)/sdA if sdA>0 else np.nan
    pE = (1 + np.sum(deltas_E >= delta_E)) / (1 + np.sum(np.isfinite(deltas_E)))
    pA = (1 + np.sum(deltas_A >= delta_AUC)) / (1 + np.sum(np.isfinite(deltas_A)))

    out_dir = out_root/align/sid/"condenc"/args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir/f"{A}_to_{B}_scalar_{args.orientation}.npz"
    np.savez_compressed(
        out_npz,
        delta_E=delta_E, delta_AUC=delta_AUC,
        null_E=deltas_E, null_AUC=deltas_A,
        z_E=zE, z_AUC=zA, p_E=pE, p_AUC=pA,
        meta=dict(
            sid=sid, align=align, orientation=args.orientation,
            src_area=A, tgt_area=B,
            axes_tag=args.axes_tag,
            win_src=win_src, win_tgt=win_tgt,
            n_trials=int(keep.sum()), n_low=int(low.size), n_high=int(high.size),
            min_per_stratum=int(args.min_per_stratum), perms=int(args.perms),
        )
    )
    print(f"[ok] wrote {out_npz}")
    print(f"[result] ΔE={delta_E:.4f} (z={zE:.2f}, p={pE:.4g}) | ΔAUC={delta_AUC:.4f} (z={zA:.2f}, p={pA:.4g})")

    # quick plot: null histograms + observed
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.hist(deltas_E[np.isfinite(deltas_E)], bins=40)
    ax1.axvline(delta_E, lw=2)
    ax1.set_title("ΔE null")
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(deltas_A[np.isfinite(deltas_A)], bins=40)
    ax2.axvline(delta_AUC, lw=2)
    ax2.set_title("ΔAUC null")
    fig.suptitle(f"{sid} {A}->{B} ori={args.orientation} win_src={win_src} win_tgt={win_tgt}")
    fig.tight_layout()
    fig.savefig(out_npz.with_suffix(".png"), dpi=300)
    plt.close(fig)

if __name__=="__main__":
    main()
