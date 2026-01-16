#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For one session (sid) and one or more tags, run pairwise DIFF+metrics and make
both:
  • diff-only overlays (requested style): show only (A→B − B→A) with null mean
  • keep full overlays produced by 18_pairdiff_and_metrics.py separately

This script:
  1) auto-discovers all available pairs under each tag folder
  2) calls 18_pairdiff_and_metrics.py for each pair NPZ
  3) builds additional DIFF-ONLY overlays from the saved metrics NPZ
  4) writes a per-tag manifest JSON listing outputs

Examples:
  python 18_run_pairdiff_session.py --sid 20200402 \
    --tags induced_k2_win016_p500 induced_k5_win016_p500

  # include a non-induced tag too
  python 18_run_pairdiff_session.py --sid 20201001 \
    --tags induced_k2_win016_p500 win160_k2_perm500_integrated
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, subprocess, sys, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------- utilities --------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def as1d(a): return np.asarray(a, dtype=float).ravel()

def _sigbar_y(series_list):
    vals = np.concatenate([as1d(s) for s in series_list])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0
    rng = np.nanmax(vals) - np.nanmin(vals)
    if not np.isfinite(rng) or rng <= 0: rng = 1e-6
    return float(np.nanmin(vals) - 0.05 * rng)

def plot_diff_only_mean_null(out_png: Path, title: str, x, diff, dnull_mu, p_diff, alpha=0.05):
    x = as1d(x); diff = as1d(diff); dnull_mu = as1d(dnull_mu) if dnull_mu is not None else None
    plt.figure(figsize=(8.6, 4.2), dpi=160)
    if dnull_mu is not None:
        plt.plot(x, dnull_mu, ls="--", lw=1.2, alpha=0.9, label="null μ (diff)")
    plt.plot(x, diff, lw=2.4, label="(A→B) − (B→A)")
    plt.axhline(0, color="k", ls=":", lw=1)

    # significance ticks ONLY when significant
    if p_diff is not None:
        p_diff = as1d(p_diff)
        if p_diff.size == x.size:
            sigmask = np.isfinite(p_diff) & (p_diff < alpha)
            if np.any(sigmask):
                vals = diff[np.isfinite(diff)]
                ybar = (np.nanmin(vals) - 0.05*(np.nanmax(vals)-np.nanmin(vals))) if vals.size else 0.0
                plt.plot(x[sigmask], np.full(sigmask.sum(), ybar), ".", ms=6, color="k")

    plt.title(title)
    plt.xlabel("Time (s)"); plt.ylabel("Flow (difference)")
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(out_png); plt.close()


def find_pair_npzs(tag_dir: Path):
    """Return list of (mode, file_path) where mode in {'induced','noninduced'}."""
    induced = sorted(tag_dir.glob("induced_flow_*to*.npz"))
    if induced:
        return [("induced", p) for p in induced]
    # else non-induced, integrated products
    nonind = sorted(tag_dir.glob("flow_timeseriesINT_*to*.npz"))
    return [("noninduced", p) for p in nonind]

def parse_pair_from_filename(p: Path) -> tuple[str,str]:
    """Extract 'A','B' from induced_flow_AtoB.npz or flow_timeseriesINT_AtoB.npz."""
    name = p.name
    m = re.search(r"(?:induced_flow|flow_timeseriesINT)_([A-Za-z0-9]+)to([A-Za-z0-9]+)\.npz$", name)
    if not m:
        raise ValueError(f"Unrecognized pair filename: {name}")
    return m.group(1), m.group(2)

def run_pairdiff_engine(engine_py: Path, pair_npz: Path, outdir: Path, alpha: float, redo: bool):
    """Call 18_pairdiff_and_metrics.py for one pair."""
    ensure_dir(outdir)
    # If already done and not redo, skip
    marker = outdir / "pairdiff_timeseries_metrics.npz"
    if marker.exists() and not redo:
        return True
    cmd = [sys.executable, str(engine_py), "--pair", str(pair_npz), "--outdir", str(outdir), "--alpha", str(alpha)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"[ERR] Engine failed for {pair_npz}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        return False
    print(r.stdout.strip())
    return True

def make_diff_only_overlays(pair_outdir: Path, alpha: float):
    """Load the metrics NPZ and write 4 diff-only overlays (C/R × raw/int)."""
    npz = pair_outdir / "pairdiff_timeseries_metrics.npz"
    if not npz.exists():
        print(f"[WARN] missing metrics npz: {npz}")
        return []
    z = np.load(npz, allow_pickle=True)
    C = z["C"].item(); R = z["R"].item()
    outs = []

    # C raw / int
    outCraw = pair_outdir / "overlay_DIFFONLY_C_raw.png"
    plot_diff_only_mean_null(
        outCraw,
        f"C (diff only)   band[{C['band'][0]:.2f},{C['band'][1]:.2f}]",
        C["t_raw"], C["diff_raw"], C["dnull_mu_raw"], C["p_diff_raw"], alpha
    ); outs.append(outCraw)
    outCint = pair_outdir / "overlay_DIFFONLY_C_int.png"
    plot_diff_only_mean_null(
        outCint,
        f"C integrated (diff only, win={C['int_win']:.3f}s)   band[{C['band'][0]:.2f},{C['band'][1]:.2f}]",
        C["t_int"], C["diff_int"], C["dnull_mu_int"], C["p_diff_int"], alpha
    ); outs.append(outCint)

    # R raw / int
    outRraw = pair_outdir / "overlay_DIFFONLY_R_raw.png"
    plot_diff_only_mean_null(
        outRraw,
        f"R (diff only)   band[{R['band'][0]:.2f},{R['band'][1]:.2f}]",
        R["t_raw"], R["diff_raw"], R["dnull_mu_raw"], R["p_diff_raw"], alpha
    ); outs.append(outRraw)
    outRint = pair_outdir / "overlay_DIFFONLY_R_int.png"
    plot_diff_only_mean_null(
        outRint,
        f"R integrated (diff only, win={R['int_win']:.3f}s)   band[{R['band'][0]:.2f},{R['band'][1]:.2f}]",
        R["t_int"], R["diff_int"], R["dnull_mu_int"], R["p_diff_int"], alpha
    ); outs.append(outRint)

    return outs

# ------------------------------ main --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True, help="Session ID, e.g., 20200402")
    ap.add_argument("--tags", nargs="+", required=True, help="One or more tag folders under results/session/<sid>/")
    ap.add_argument("--session_root", default="results/session", help="Base path to session outputs")
    ap.add_argument("--out_subdir", default="pairdiff", help="Subfolder to write pairdiff outputs")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--redo", action="store_true", help="Recompute even if outputs exist")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    engine_py = repo_root / "18_pairdiff_and_metrics.py"
    if not engine_py.exists():
        print("[FATAL] 18_pairdiff_and_metrics.py not found next to this script.")
        sys.exit(2)

    sid_dir = Path(args.session_root) / str(args.sid)
    if not sid_dir.exists():
        print(f"[FATAL] Session folder not found: {sid_dir}")
        sys.exit(2)

    overall = {"sid": str(args.sid), "alpha": args.alpha, "tags": {}}

    for tag in args.tags:
        tag_dir = sid_dir / tag
        if not tag_dir.exists():
            print(f"[WARN] Tag folder missing: {tag_dir}, skipping.")
            continue

        mode_pairs = find_pair_npzs(tag_dir)
        if not mode_pairs:
            print(f"[WARN] No pair NPZs detected in {tag_dir}.")
            continue

        tag_manifest = []
        for mode, pair_npz in mode_pairs:
            A, B = parse_pair_from_filename(pair_npz)
            pair_name = f"{A}to{B}"
            pair_outdir = ensure_dir(tag_dir / args.out_subdir / pair_name)

            ok = run_pairdiff_engine(engine_py, pair_npz, pair_outdir, args.alpha, args.redo)
            if not ok:
                tag_manifest.append({"pair": pair_name, "status": "engine_failed", "pair_npz": str(pair_npz)})
                continue

            # Create DIFF-ONLY overlays (requested)
            outs = make_diff_only_overlays(pair_outdir, args.alpha)

            tag_manifest.append({
                "pair": pair_name,
                "mode": mode,
                "pair_npz": str(pair_npz),
                "outdir": str(pair_outdir),
                "diff_only_plots": [str(p) for p in outs],
                "full_plots": [
                    # from engine script naming
                    str(pair_outdir / "overlay_diff_C_raw.png"),
                    str(pair_outdir / "overlay_diff_C_int.png"),
                    str(pair_outdir / "overlay_diff_R_raw.png"),
                    str(pair_outdir / "overlay_diff_R_int.png"),
                    str(pair_outdir / "metrics_C_raw.png"),
                    str(pair_outdir / "metrics_C_int.png"),
                    str(pair_outdir / "metrics_R_raw.png"),
                    str(pair_outdir / "metrics_R_int.png"),
                    str(pair_outdir / "pvals_C_raw.png"),
                    str(pair_outdir / "pvals_C_int.png"),
                    str(pair_outdir / "pvals_R_raw.png"),
                    str(pair_outdir / "pvals_R_int.png"),
                    str(pair_outdir / "pairdiff_timeseries_metrics.npz"),
                    str(pair_outdir / "pairdiff_metrics_summary.csv"),
                ],
                "status": "ok"
            })

        # write a manifest per tag
        manifest_path = tag_dir / args.out_subdir / "pairdiff_manifest.json"
        ensure_dir(manifest_path.parent)
        with open(manifest_path, "w") as f:
            json.dump({"sid": str(args.sid), "tag": tag, "pairs": tag_manifest}, f, indent=2)
        print(f"[ok] wrote {manifest_path}")
        overall["tags"][tag] = str(manifest_path)

    # overall pointer under session root
    top_manifest = sid_dir / "pairdiff_alltags.json"
    with open(top_manifest, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"[ok] wrote {top_manifest}")


if __name__ == "__main__":
    main()
