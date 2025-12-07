#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, warnings
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- helpers ----------

def _areas_present(out_root: str, align: str, sid: str) -> List[str]:
    fdir = os.path.join(out_root, align, sid, "flow")
    if not os.path.isdir(fdir):  # flow root may not exist yet
        # Fall back to caches to detect recorded areas
        cdir = os.path.join(out_root, align, sid, "caches")
        if not os.path.isdir(cdir): return []
        return sorted([os.path.basename(p)[5:-4] for p in glob(os.path.join(cdir, "area_*.npz"))])
    # Prefer areas that actually have flow outputs
    areas = set()
    for tagdir in glob(os.path.join(fdir, "*")):
        for featdir in glob(os.path.join(tagdir, "*")):
            for fn in glob(os.path.join(featdir, "flow_*_*.npz")):
                base = os.path.basename(fn)
                # flow_C_MFEFtoMLIP.npz
                try:
                    pair = base.split("_", 2)[-1].replace(".npz", "")
                    A,B = pair.split("to")
                    areas.add(A); areas.add(B)
                except Exception:
                    pass
    if not areas:
        # fallback to caches
        return _areas_present(out_root, align, sid)
    return sorted(areas)

def _canonical_pairs(areas: List[str]) -> List[Tuple[str,str]]:
    if not areas: return []
    # Infer monkey from prefix
    pref = areas[0][0].upper()
    if pref == "M":
        want = ["MFEF","MLIP","MSC"]
        triples = [("MFEF","MLIP"), ("MFEF","MSC"), ("MLIP","MSC")]
    else:
        want = ["SFEF","SLIP","SSC"]
        triples = [("SFEF","SLIP"), ("SFEF","SSC"), ("SLIP","SSC")]
    present = set(areas)
    pairs = [(a,b) for (a,b) in triples if a in present and b in present]
    # If only 2 areas present, include that pair in canonical order
    if len(pairs) == 0 and len(areas) == 2:
        # sort by preference list if we can
        idx = {name:i for i,name in enumerate(want)}
        a,b = sorted(areas, key=lambda x: idx.get(x, 99))
        pairs = [(a,b)]
    return pairs

def _load_pair_npz(out_root: str, align: str, sid: str, tag: str, feature: str, A: str, B: str):
    base = os.path.join(out_root, align, sid, "flow", tag, feature)
    p_fwd = os.path.join(base, f"flow_{feature}_{A}to{B}.npz")
    p_rev = os.path.join(base, f"flow_{feature}_{B}to{A}.npz")
    if not (os.path.exists(p_fwd) and os.path.exists(p_rev)):
        raise FileNotFoundError(f"Missing pair files for {A}<->{B}: {p_fwd} or {p_rev}")
    Zf = np.load(p_fwd, allow_pickle=True)
    Zr = np.load(p_rev, allow_pickle=True)
    return Zf, Zr

def _safe(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float)

def _mask_finite(*arrs):
    # returns indices where all provided arrays are finite
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def _evidence_strength(obs: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.where(sd > 0, sd, np.nan)
    return (obs - mu) / sd

def _signif_dots(pA: np.ndarray, pB: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean mask where exactly one direction is significant (conservative single-session dots).
    """
    mA = np.isfinite(pA) & (pA < alpha)
    mB = np.isfinite(pB) & (pB < alpha)
    return (mA & (~mB)) | (mB & (~mA))

# ---------- plotting per pair ----------

def make_pair_figure(Zf, Zr, pair: Tuple[str,str], feature: str, out_pdf: str, alpha: float = 0.05):
    A,B = pair
    t = _safe(Zf["time"]) * 1000.0  # ms

    bits_AB = _safe(Zf["bits_AtoB"])
    bits_BA = _safe(Zr["bits_AtoB"])   # note: reverse file contains B->A as its A->B
    mu_AB   = _safe(Zf["null_mean_AtoB"])
    sd_AB   = _safe(Zf["null_std_AtoB"])
    mu_BA   = _safe(Zr["null_mean_AtoB"])
    sd_BA   = _safe(Zr["null_std_AtoB"])
    p_AB    = _safe(Zf["p_AtoB"])
    p_BA    = _safe(Zr["p_AtoB"])

    # Panel C: evidence strength
    z_AB = _evidence_strength(bits_AB, mu_AB, sd_AB)
    z_BA = _evidence_strength(bits_BA, mu_BA, sd_BA)

    # Panel B: pair-diff
    diff = bits_AB - bits_BA
    sigdot = _signif_dots(p_AB, p_BA, alpha=alpha)

    fig = plt.figure(figsize=(7.8, 7.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0], hspace=0.25)

    # Panel A — overlay with null bands
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvline(0, ls="--", c="k", lw=0.8)
    # bands (mask to finite)
    mA = _mask_finite(t, mu_AB, sd_AB)
    mB = _mask_finite(t, mu_BA, sd_BA)
    if np.any(mA):
        ax1.fill_between(t[mA], mu_AB[mA]-sd_AB[mA], mu_AB[mA]+sd_AB[mA], color="C0", alpha=0.15, linewidth=0, label=f"{A}→{B} null μ±σ")
        ax1.plot(t[mA], mu_AB[mA], color="C0", lw=1.0, ls=":")
    if np.any(mB):
        ax1.fill_between(t[mB], mu_BA[mB]-sd_BA[mB], mu_BA[mB]+sd_BA[mB], color="C1", alpha=0.15, linewidth=0, label=f"{B}→{A} null μ±σ")
        ax1.plot(t[mB], mu_BA[mB], color="C1", lw=1.0, ls=":")

    ax1.plot(t, bits_AB, color="C0", lw=2.2, label=f"{A}→{B}")
    ax1.plot(t, bits_BA, color="C1", lw=2.2, label=f"{B}→{A}")
    ax1.set_ylabel("ΔLL (bits)")
    ax1.set_title(f"({A} vs {B}) — {feature}-flow — Panel A: overlay + null bands")
    ax1.legend(ncol=2, frameon=False, loc="upper left")

    # Panel B — pair-diff with significance dots
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axvline(0, ls="--", c="k", lw=0.8)
    ax2.axhline(0, ls=":", c="k", lw=0.8)
    ax2.plot(t, diff, color="C3", lw=2.2, label="(A→B − B→A)")
    # black dots where only one direction is significant
    if np.any(sigdot):
        ax2.plot(t[sigdot], diff[sigdot], "k.", ms=5, label=f"p<{alpha:g} (one-sided)")
    ax2.set_ylabel("ΔΔLL (bits)")
    ax2.set_title("Panel B: pair-diff with significance dots")
    ax2.legend(frameon=False, loc="upper left")

    # Panel C — evidence strength (z-scores)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axvline(0, ls="--", c="k", lw=0.8)
    ax3.plot(t, z_AB, color="C0", lw=2.0, label=f"z({A}→{B})")
    ax3.plot(t, z_BA, color="C1", lw=2.0, label=f"z({B}→{A})")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Z (σ from null)")
    ax3.set_title("Panel C: evidence strength vs null")
    ax3.legend(frameon=False, loc="upper left")

    # save
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    # Suppress tight_layout warning (occurs with gridspec layouts but layout still works)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=1.0)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf",".png"), dpi=300)
    fig.savefig(out_pdf.replace(".pdf",".svg"))
    plt.close(fig)

    # also dump a compact JSON with derived arrays
    out_json = out_pdf.replace(".pdf", ".json")
    payload = dict(
        pair=[A,B], feature=feature,
        time_ms=t.tolist(),
        bits_AtoB=bits_AB.tolist(), bits_BtoA=bits_BA.tolist(),
        null_mu_AtoB=mu_AB.tolist(), null_sd_AtoB=sd_AB.tolist(),
        null_mu_BtoA=mu_BA.tolist(), null_sd_BtoA=sd_BA.tolist(),
        p_AtoB=p_AB.tolist(), p_BtoA=p_BA.tolist(),
        diff=(bits_AB - bits_BA).tolist(),
        sigmask=sigdot.astype(int).tolist(),
        z_AtoB=z_AB.tolist(), z_BtoA=z_BA.tolist(),
    )
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    return out_pdf

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Single-session flow overlays (per canonical pair) with panels A/B/C.")
    ap.add_argument("--out_root", default=os.path.join(os.environ.get("PAPER_HOME","."),"out"))
    ap.add_argument("--align", choices=["stim","sacc"], required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--feature", choices=["C","R","S"], required=True)
    ap.add_argument("--orientation", choices=["vertical","horizontal","pooled"], default="vertical",
                    help="Used only to locate the right flow tag directory; not applied here since flow has been computed.")
    ap.add_argument("--tag", default="flow_v1")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    areas = _areas_present(args.out_root, args.align, args.sid)
    if not areas:
        raise SystemExit(f"No areas found under {args.out_root}/{args.align}/{args.sid}")
    pairs = _canonical_pairs(areas)
    if not pairs:
        raise SystemExit(f"No canonical pairs available for areas: {areas}")

    # draw figures per canonical pair
    base_fig_dir = os.path.join(args.out_root, args.align, args.sid, "flow", args.tag, args.feature, "figs")
    os.makedirs(base_fig_dir, exist_ok=True)

    done = 0
    for (A,B) in pairs:
        try:
            Zf, Zr = _load_pair_npz(args.out_root, args.align, args.sid, args.tag, args.feature, A, B)
        except FileNotFoundError as e:
            print(f"[skip] {A}<->{B}: {e}")
            continue
        out_pdf = os.path.join(base_fig_dir, f"{A}_vs_{B}_flow_{args.feature}.pdf")
        make_pair_figure(Zf, Zr, (A,B), args.feature, out_pdf, alpha=args.alpha)
        print(f"[{args.sid}] wrote {out_pdf}")
        done += 1

    if done == 0:
        print("[warn] no pairs had both directions available; nothing drawn.")

if __name__ == "__main__":
    main()
