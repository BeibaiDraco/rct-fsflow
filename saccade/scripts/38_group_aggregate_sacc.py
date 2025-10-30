#!/usr/bin/env python
import argparse, os, json, numpy as np, matplotlib.pyplot as plt
import math
from glob import glob
from saccflow.io import list_sessions

def _pairs_in_dir(flow_dir):
    files = sorted(glob(os.path.join(flow_dir, "induced_flow_S_*to*.npz")))
    pairs = []
    for f in files:
        base = os.path.basename(f).replace("induced_flow_S_","").replace(".npz","")
        A,B = base.split("to")
        pairs.append((A,B))
    return sorted(set(pairs))

def _fisher_combine(pvals):
    # Fisher's method; normal approx for chi-square tail
    p = [min(max(float(x), 1e-300), 1.0) for x in pvals]   # clip for safety
    stat = -2.0 * sum(math.log(x) for x in p)
    k = len(p)
    z = (stat - 2*k) / math.sqrt(4*k + 1e-9)
    # two-sided normal tail: 2*(1 - Phi(|z|)) == erfc(|z|/sqrt(2))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))

def _stack_time_series(files, key):
    arrs = []
    time = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if time is None: time = d["time"].astype(float)
        if key not in d: continue
        arrs.append(d[key].astype(float))
    if not arrs:
        return time, np.zeros_like(time), np.zeros_like(time), []
    X = np.vstack(arrs)  # (S,T)
    mean = np.nanmean(X, axis=0)
    sem  = np.nanstd(X, axis=0, ddof=1) / np.sqrt(max(1, X.shape[0]))
    return time, mean, sem, arrs

def _combine_p(files, pkey):
    if not files: return None, None
    d0 = np.load(files[0], allow_pickle=True)
    time = d0["time"].astype(float); T = len(time)
    out = np.ones(T, dtype=float)
    for t in range(T):
        pt = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            pv = d[pkey].astype(float)
            if np.isnan(pv[t]): continue
            pt.append(pv[t])
        out[t] = _fisher_combine(pt) if pt else 1.0
    return time, out

def main():
    ap = argparse.ArgumentParser(description="Aggregate S-flow across sessions (group overlays and pair-diff).")
    ap.add_argument("--tag", required=True, help="saccflow run tag, e.g., sacc_v1")
    ap.add_argument("--orientation", choices=["vertical","horizontal"], default="vertical")
    ap.add_argument("--out_root", default="results_sacc")
    ap.add_argument("--root_sessions", default="RCT_02")
    ap.add_argument("--by_monkey", choices=["separate","pooled"], default="separate")
    args = ap.parse_args()

    sids = list_sessions(args.root_sessions)

    # Discover all pairs present somewhere
    all_pairs = set()
    for sid in sids:
        fdir = os.path.join(args.out_root, sid, "saccflow", args.tag)
        if os.path.isdir(fdir):
            for p in _pairs_in_dir(fdir): all_pairs.add(p)
    all_pairs = sorted(all_pairs)

    # Group directories
    gdir = os.path.join(args.out_root, "group_sacc", args.tag, args.orientation)
    os.makedirs(gdir, exist_ok=True)

    for (A,B) in all_pairs:
        # Monkey split (by prefix M* vs S*)
        cohort = ["M","S"] if args.by_monkey == "separate" else ["MS"]

        for mk in cohort:
            if mk=="M":
                ok_pair = (A.startswith("M") and B.startswith("M"))
            elif mk=="S":
                ok_pair = (A.startswith("S") and B.startswith("S"))
            else:
                ok_pair = True
            if not ok_pair: continue

            flow_files = []
            diff_files = []
            for sid in sids:
                fdir = os.path.join(args.out_root, sid, "saccflow", args.tag)
                fflow = os.path.join(fdir, f"induced_flow_S_{A}to{B}.npz")
                fdiff = os.path.join(fdir, "pairdiff", f"pairdiff_S_{A}to{B}.npz")
                if os.path.exists(fflow):
                    meta = json.loads(np.load(fflow, allow_pickle=True)["meta"].item())
                    if meta.get("orientation","vertical") != args.orientation: continue
                    flow_files.append(fflow)
                if os.path.exists(fdiff):
                    meta2 = json.loads(np.load(fdiff, allow_pickle=True)["meta"].item())
                    if meta2.get("orientation","vertical") != args.orientation: continue
                    diff_files.append(fdiff)

            if not flow_files or not diff_files: continue

            # Overlays aggregation
            t, mean_AB, sem_AB, _ = _stack_time_series(flow_files, "bits_AtoB")
            _, mean_BA, sem_BA, _ = _stack_time_series(flow_files, "bits_BtoA")
            _, pAB = _combine_p(flow_files, "p_AtoB")
            _, pBA = _combine_p(flow_files, "p_BtoA")

            # Pair-diff aggregation
            td, mean_DIFF, sem_DIFF, _ = _stack_time_series(diff_files, "diff_bits")
            _, pDIFF = _combine_p(diff_files, "p_diff")

            # Save NPZ
            out_pair_dir = os.path.join(gdir, f"{mk}_{A}to{B}")
            os.makedirs(out_pair_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(out_pair_dir, "group_overlay.npz"),
                time=t, mean_AtoB=mean_AB, sem_AtoB=sem_AB, mean_BtoA=mean_BA, sem_BtoA=sem_BA,
                p_AtoB=pAB, p_BtoA=pBA, n_sessions=len(flow_files)
            )
            np.savez_compressed(
                os.path.join(out_pair_dir, "group_pairdiff.npz"),
                time=td, mean_DIFF=mean_DIFF, sem_DIFF=sem_DIFF, p_DIFF=pDIFF, n_sessions=len(diff_files)
            )

            # Quick PDFs
            # Overlay
            plt.figure(figsize=(6.8,3.2))
            plt.axhline(0, ls="--", c="k", lw=0.8); plt.axvline(0, ls="--", c="k", lw=0.8)
            plt.plot(t*1000, mean_AB, label="A→B", lw=2)
            plt.fill_between(t*1000, mean_AB-sem_AB, mean_AB+sem_AB, alpha=0.15)
            plt.plot(t*1000, mean_BA, label="B→A", lw=2)
            plt.fill_between(t*1000, mean_BA-sem_BA, mean_BA+sem_BA, alpha=0.15)
            plt.title(f"{mk} {A}→{B} overlay ({args.orientation})"); plt.xlabel("ms"); plt.ylabel("ΔLL bits")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_pair_dir, "group_overlay.pdf")); plt.close()

            # Pair-diff
            plt.figure(figsize=(6.8,3.2))
            plt.axhline(0, ls="--", c="k", lw=0.8); plt.axvline(0, ls="--", c="k", lw=0.8)
            plt.plot(td*1000, mean_DIFF, label="(A→B − B→A)", lw=2, color="C3")
            plt.fill_between(td*1000, mean_DIFF-sem_DIFF, mean_DIFF+sem_DIFF, alpha=0.15, color="C3")
            plt.title(f"{mk} {A}→{B} pair-diff ({args.orientation})"); plt.xlabel("ms"); plt.ylabel("ΔLL bits")
            plt.tight_layout()
            plt.savefig(os.path.join(out_pair_dir, "group_pairdiff.pdf")); plt.close()

            print(f"[group] {mk} {A}->{B}: {len(flow_files)} sessions")

if __name__ == "__main__":
    main()
