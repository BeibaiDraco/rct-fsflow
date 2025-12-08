#!/usr/bin/env python3
"""
Summarize flow results across sessions, per tag/config, per align, per feature,
and per canonical pair, separately for each monkey (M vs S via area prefixes).

For each (align, tag, feature, pair A-B, monkey_label) it:

  - finds all sessions with both directions present (A->B and B->A)
  - loads bits, null means/SDs, p-values
  - computes per-session:
        bits_AtoB(t), bits_BtoA(t),
        z_AtoB(t), z_BtoA(t),
        diff_bits(t) = bits_AtoB(t) - bits_BtoA(t),
        sig masks (p<alpha),
        window-averaged excess bits & z in a task window
  - aggregates across sessions:
        mean ± SE for bits, z, diff over time,
        fraction of sessions significant at each time,
        window-level mean ± SE and fraction sig

Outputs per tag/align/feature/pair:

  out/<align>/summary/<tag>/<feature>/summary_<A>_vs_<B>.npz
  out/<align>/summary/<tag>/<feature>/figs/<A>_vs_<B>.pdf/.png

The .npz contains:
  - time (sec)
  - mean_bits_AtoB, se_bits_AtoB, mean_bits_BtoA, se_bits_BtoA
  - mean_z_AtoB,   se_z_AtoB,   mean_z_BtoA,   se_z_BtoA
  - frac_sig_AtoB, frac_sig_BtoA
  - mean_diff_bits, se_diff_bits
  - window-level stats (excess bits, z, diff) + fraction sig
  - session_ids (string array)
  - meta_json (JSON-encoded dict: tag, align, feature, pair, monkey_label, alpha, window, n_sessions)

Usage examples (from paper_project_final/):

  # summarize everything we have (stim + sacc, all tags, default windows)
  python cli/summarize_flow_across_sessions.py \
      --out_root out \
      --align both \
      --alpha 0.05

  # summarize stim-align only, specific tags
  python cli/summarize_flow_across_sessions.py \
      --out_root out \
      --align stim \
      --tags crsweep-stim-vertical-none-trial crsweep-stim-vertical-zreg-trial
"""

from __future__ import annotations
import argparse, os, json, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- helpers --------------------

def find_sessions(out_root: Path, align: str) -> List[str]:
    base = out_root / align
    if not base.exists():
        return []
    sids = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "caches").is_dir():
            sids.append(p.name)
    return sids


def discover_tags(out_root: Path, align: str, sids: List[str]) -> List[str]:
    tags = set()
    for sid in sids:
        flow_root = out_root / align / sid / "flow"
        if not flow_root.is_dir():
            continue
        for tag_dir in flow_root.iterdir():
            if tag_dir.is_dir():
                tags.add(tag_dir.name)
    return sorted(tags)


def discover_features(out_root: Path, align: str, tag: str, sids: List[str]) -> List[str]:
    """
    Discover available features for a given tag by checking what feature
    directories exist across sessions. Returns sorted list of feature names.
    """
    features = set()
    for sid in sids:
        tag_root = out_root / align / sid / "flow" / tag
        if not tag_root.is_dir():
            continue
        for feat_dir in tag_root.iterdir():
            if feat_dir.is_dir():
                features.add(feat_dir.name)
    return sorted(features)


def canonical_pairs(monkey_label: str) -> List[Tuple[str, str]]:
    """
    Canonical area pairs per monkey type.
    M: MFEF, MLIP, MSC
    S: SFEF, SLIP, SSC
    """
    if monkey_label.upper() == "M":
        return [("MFEF", "MLIP"), ("MFEF", "MSC"), ("MLIP", "MSC")]
    else:
        return [("SFEF", "SLIP"), ("SFEF", "SSC"), ("SLIP", "SSC")]


def parse_window(s: str) -> Tuple[float, float]:
    a, b = s.split(":")
    return float(a), float(b)


def safe_z(bits: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.full_like(bits, np.nan, dtype=float)
    mask = np.isfinite(bits) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    z[mask] = (bits[mask] - mu[mask]) / sd[mask]
    return z


def nanmean_se(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (mean, se, n_valid) along axis, ignoring NaNs.
    Suppresses warnings for empty slices or insufficient degrees of freedom.
    """
    with warnings.catch_warnings():
        # Suppress numpy warnings about empty slices and insufficient degrees of freedom
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(arr, axis=axis)
            n = np.sum(np.isfinite(arr), axis=axis).astype(float)
            std = np.nanstd(arr, axis=axis, ddof=1)
            se = std / np.sqrt(n)
    return mean, se, n


def plot_summary_figure(
    out_path_pdf: Path,
    time: np.ndarray,
    mean_bits_AB: np.ndarray,
    se_bits_AB: np.ndarray,
    mean_bits_BA: np.ndarray,
    se_bits_BA: np.ndarray,
    mean_z_AB: np.ndarray,
    se_z_AB: np.ndarray,
    mean_z_BA: np.ndarray,
    se_z_BA: np.ndarray,
    mean_diff: np.ndarray,
    se_diff: np.ndarray,
    frac_sig_AB: np.ndarray,
    win: Tuple[float, float],
    title: str,
) -> None:
    """
    Make a summary figure with:
      - Panel 1: bits ± SE (A->B, B->A)
      - Panel 2: z ± SE
      - Panel 3: diff bits ± SE + frac_sig_AB(t)
    """
    t_ms = time * 1000.0
    w_start_ms = win[0] * 1000.0
    w_end_ms = win[1] * 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
    ax1, ax2, ax3 = axes

    # Panel 1: bits
    ax1.axvline(0, ls="--", c="k", lw=0.8)
    ax1.axvspan(w_start_ms, w_end_ms, color="0.9", alpha=0.5, label="window")
    ax1.plot(t_ms, mean_bits_AB, color="C0", lw=2.0, label="A→B bits")
    ax1.fill_between(
        t_ms,
        mean_bits_AB - se_bits_AB,
        mean_bits_AB + se_bits_AB,
        color="C0",
        alpha=0.25,
        linewidth=0,
    )
    ax1.plot(t_ms, mean_bits_BA, color="C1", lw=2.0, label="B→A bits")
    ax1.fill_between(
        t_ms,
        mean_bits_BA - se_bits_BA,
        mean_bits_BA + se_bits_BA,
        color="C1",
        alpha=0.25,
        linewidth=0,
    )
    ax1.set_ylabel("ΔLL (bits)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", frameon=False)

    # Panel 2: z-scores
    ax2.axvline(0, ls="--", c="k", lw=0.8)
    ax2.axhline(0, ls=":", c="k", lw=0.8)
    ax2.axvspan(w_start_ms, w_end_ms, color="0.9", alpha=0.5)
    ax2.plot(t_ms, mean_z_AB, color="C0", lw=2.0, label="A→B z")
    ax2.fill_between(
        t_ms,
        mean_z_AB - se_z_AB,
        mean_z_AB + se_z_AB,
        color="C0",
        alpha=0.25,
        linewidth=0,
    )
    ax2.plot(t_ms, mean_z_BA, color="C1", lw=2.0, label="B→A z")
    ax2.fill_between(
        t_ms,
        mean_z_BA - se_z_BA,
        mean_z_BA + se_z_BA,
        color="C1",
        alpha=0.25,
        linewidth=0,
    )
    ax2.set_ylabel("Z (σ from null)")
    ax2.legend(loc="upper left", frameon=False)

    # Panel 3: diff + frac sig
    ax3.axvline(0, ls="--", c="k", lw=0.8)
    ax3.axhline(0, ls=":", c="k", lw=0.8)
    ax3.axvspan(w_start_ms, w_end_ms, color="0.9", alpha=0.5)
    ax3.plot(t_ms, mean_diff, color="C3", lw=2.0, label="A→B − B→A")
    ax3.fill_between(
        t_ms,
        mean_diff - se_diff,
        mean_diff + se_diff,
        color="C3",
        alpha=0.25,
        linewidth=0,
    )
    ax3.set_ylabel("ΔΔLL (bits)")

    ax3b = ax3.twinx()
    ax3b.plot(t_ms, frac_sig_AB, color="k", lw=1.5, ls="--", label="Frac sig A→B")
    ax3b.set_ylabel("Fraction p<α")
    ax3b.set_ylim(0, 1.0)

    # Combined legend for panel 3
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3b.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax3.set_xlabel("Time (ms)")

    fig.tight_layout()
    out_dir = out_path_pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


# -------------------- main summarizer --------------------

def summarize_for_tag_align_feature(
    out_root: Path,
    align: str,
    tag: str,
    feature: str,
    sids: List[str],
    alpha: float,
    win: Tuple[float, float],
) -> None:
    """
    Summarize flows across sessions for one (align, tag, feature).
    Writes one .npz + .pdf/.png per canonical pair (A,B).
    """
    for monkey_label in ("M", "S"):
        pairs = canonical_pairs(monkey_label)
        for (A, B) in pairs:
            all_bits_AB = []
            all_bits_BA = []
            all_z_AB = []
            all_z_BA = []
            all_diff = []
            all_sig_AB = []
            all_sig_BA = []
            win_excess_AB = []
            win_excess_BA = []
            win_z_AB = []
            win_z_BA = []
            win_diff = []
            win_sig_AB = []

            session_ids = []
            time = None

            for sid in sids:
                base = out_root / align / sid / "flow" / tag / feature
                p_fwd = base / f"flow_{feature}_{A}to{B}.npz"
                p_rev = base / f"flow_{feature}_{B}to{A}.npz"
                if not (p_fwd.is_file() and p_rev.is_file()):
                    continue

                Zf = np.load(p_fwd, allow_pickle=True)
                Zr = np.load(p_rev, allow_pickle=True)

                t = np.asarray(Zf["time"], dtype=float)  # seconds
                if time is None:
                    time = t
                else:
                    if time.shape != t.shape or not np.allclose(time, t):
                        raise ValueError(
                            f"Inconsistent time grid for {align}, tag={tag}, feature={feature}, "
                            f"pair {A}->{B}, sid={sid}"
                        )

                bits_AB = np.asarray(Zf["bits_AtoB"], dtype=float)
                bits_BA = np.asarray(Zr["bits_AtoB"], dtype=float)
                mu_AB = np.asarray(Zf["null_mean_AtoB"], dtype=float)
                sd_AB = np.asarray(Zf["null_std_AtoB"], dtype=float)
                mu_BA = np.asarray(Zr["null_mean_AtoB"], dtype=float)
                sd_BA = np.asarray(Zr["null_std_BtoA"], dtype=float) if "null_std_BtoA" in Zr else np.asarray(Zr["null_std_AtoB"], dtype=float)
                p_AB = np.asarray(Zf["p_AtoB"], dtype=float)
                p_BA = np.asarray(Zr["p_BtoA"], dtype=float) if "p_BtoA" in Zr else np.asarray(Zr["p_AtoB"], dtype=float)

                z_AB = safe_z(bits_AB, mu_AB, sd_AB)
                z_BA = safe_z(bits_BA, mu_BA, sd_BA)
                diff_bits = bits_AB - bits_BA

                sig_AB = (p_AB < alpha) & np.isfinite(p_AB)
                sig_BA = (p_BA < alpha) & np.isfinite(p_BA)

                # window mask in seconds
                ws, we = win
                wmask = (time >= ws) & (time <= we)
                if not np.any(wmask):
                    w_excess_AB = np.nan
                    w_excess_BA = np.nan
                    w_z_AB = np.nan
                    w_z_BA = np.nan
                    w_diff = np.nan
                    w_sig = np.nan
                else:
                    excess_AB = bits_AB - mu_AB
                    excess_BA = bits_BA - mu_BA
                    w_excess_AB = float(np.nanmean(excess_AB[wmask]))
                    w_excess_BA = float(np.nanmean(excess_BA[wmask]))
                    w_z_AB = float(np.nanmean(z_AB[wmask]))
                    w_z_BA = float(np.nanmean(z_BA[wmask]))
                    w_diff = float(np.nanmean(diff_bits[wmask]))
                    w_sig = float(np.any(sig_AB[wmask]))

                all_bits_AB.append(bits_AB)
                all_bits_BA.append(bits_BA)
                all_z_AB.append(z_AB)
                all_z_BA.append(z_BA)
                all_diff.append(diff_bits)
                all_sig_AB.append(sig_AB.astype(float))
                all_sig_BA.append(sig_BA.astype(float))
                win_excess_AB.append(w_excess_AB)
                win_excess_BA.append(w_excess_BA)
                win_z_AB.append(w_z_AB)
                win_z_BA.append(w_z_BA)
                win_diff.append(w_diff)
                win_sig_AB.append(w_sig)
                session_ids.append(sid)

            if not all_bits_AB:
                # no sessions had this pair for this tag/feature/monkey
                continue

            # stack into (N_sessions, T)
            bits_AB_arr = np.vstack(all_bits_AB)
            bits_BA_arr = np.vstack(all_bits_BA)
            z_AB_arr = np.vstack(all_z_AB)
            z_BA_arr = np.vstack(all_z_BA)
            diff_arr = np.vstack(all_diff)
            sig_AB_arr = np.vstack(all_sig_AB)
            sig_BA_arr = np.vstack(all_sig_BA)

            # per-time summaries
            mean_bits_AB, se_bits_AB, n_AB = nanmean_se(bits_AB_arr, axis=0)
            mean_bits_BA, se_bits_BA, n_BA = nanmean_se(bits_BA_arr, axis=0)
            mean_z_AB,   se_z_AB,   _     = nanmean_se(z_AB_arr,   axis=0)
            mean_z_BA,   se_z_BA,   _     = nanmean_se(z_BA_arr,   axis=0)
            mean_diff,   se_diff,   _     = nanmean_se(diff_arr,   axis=0)

            # fraction of sessions sig at each time
            with np.errstate(invalid="ignore", divide="ignore"):
                frac_sig_AB = np.nanmean(sig_AB_arr, axis=0)
                frac_sig_BA = np.nanmean(sig_BA_arr, axis=0)

            # window-level summaries
            win_excess_AB_arr = np.array(win_excess_AB, dtype=float)
            win_excess_BA_arr = np.array(win_excess_BA, dtype=float)
            win_z_AB_arr = np.array(win_z_AB, dtype=float)
            win_z_BA_arr = np.array(win_z_BA, dtype=float)
            win_diff_arr = np.array(win_diff, dtype=float)
            win_sig_AB_arr = np.array(win_sig_AB, dtype=float)

            w_mean_excess_AB, w_se_excess_AB, w_n = nanmean_se(win_excess_AB_arr, axis=0)
            w_mean_excess_BA, w_se_excess_BA, _   = nanmean_se(win_excess_BA_arr, axis=0)
            w_mean_z_AB,      w_se_z_AB,      _   = nanmean_se(win_z_AB_arr,      axis=0)
            w_mean_z_BA,      w_se_z_BA,      _   = nanmean_se(win_z_BA_arr,      axis=0)
            w_mean_diff,      w_se_diff,      _   = nanmean_se(win_diff_arr,      axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                w_frac_sig_AB = float(np.nanmean(win_sig_AB_arr))

            n_sessions = int(len(session_ids))
            print(f"[summary] align={align}, tag={tag}, feature={feature}, "
                  f"pair={A}-{B}, monkey={monkey_label}, N={n_sessions}")

            # output dir
            summary_dir = out_root / align / "summary" / tag / feature
            summary_dir.mkdir(parents=True, exist_ok=True)
            figs_dir = summary_dir / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{A}_vs_{B}"

            # Save npz
            meta = dict(
                tag=tag,
                align=align,
                feature=feature,
                pair=f"{A}-{B}",
                monkey_label=monkey_label,
                alpha=float(alpha),
                win_start_s=float(win[0]),
                win_end_s=float(win[1]),
                n_sessions=n_sessions,
            )
            meta_json = json.dumps(meta)

            out_path_npz = summary_dir / f"summary_{pair_name}.npz"
            np.savez_compressed(
                out_path_npz,
                time=time,
                mean_bits_AtoB=mean_bits_AB,
                se_bits_AtoB=se_bits_AB,
                mean_bits_BtoA=mean_bits_BA,
                se_bits_BtoA=se_bits_BA,
                mean_z_AtoB=mean_z_AB,
                se_z_AtoB=se_z_AB,
                mean_z_BtoA=mean_z_BA,
                se_z_BtoA=se_z_BA,
                frac_sig_AtoB=frac_sig_AB,
                frac_sig_BtoA=frac_sig_BA,
                mean_diff_bits=mean_diff,
                se_diff_bits=se_diff,
                win_mean_excess_bits_AtoB=w_mean_excess_AB,
                win_se_excess_bits_AtoB=w_se_excess_AB,
                win_mean_excess_bits_BtoA=w_mean_excess_BA,
                win_se_excess_bits_BtoA=w_se_excess_BA,
                win_mean_z_AtoB=w_mean_z_AB,
                win_se_z_AtoB=w_se_z_AB,
                win_mean_z_BtoA=w_mean_z_BA,
                win_se_z_BtoA=w_se_z_BA,
                win_mean_diff_bits=w_mean_diff,
                win_se_diff_bits=w_se_diff,
                win_frac_sig_AtoB=w_frac_sig_AB,
                session_ids=np.array(session_ids, dtype="U"),
                meta_json=np.array(meta_json),
            )

            # Save figure
            title = (f"{align.upper()} | {tag} | {feature} | {A} vs {B} "
                     f"| monkey={monkey_label} | N={n_sessions}")
            fig_path_pdf = figs_dir / f"{pair_name}.pdf"
            plot_summary_figure(
                out_path_pdf=fig_path_pdf,
                time=time,
                mean_bits_AB=mean_bits_AB,
                se_bits_AB=se_bits_AB,
                mean_bits_BA=mean_bits_BA,
                se_bits_BA=se_bits_BA,
                mean_z_AB=mean_z_AB,
                se_z_AB=se_z_AB,
                mean_z_BA=mean_z_BA,
                se_z_BA=se_z_BA,
                mean_diff=mean_diff,
                se_diff=se_diff,
                frac_sig_AB=frac_sig_AB,
                win=win,
                title=title,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out",
                    help="Root under which stim/sacc live (default: out)")
    ap.add_argument("--align", choices=["stim", "sacc", "both"], default="both",
                    help="Which alignments to summarize (default: both)")
    ap.add_argument("--tags", nargs="*",
                    help="Flow tags to summarize (e.g. crsweep-stim-vertical-none-trial). "
                         "If omitted, auto-detect per align.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for p-values (default: 0.05)")
    ap.add_argument("--win_stim", default="0.10:0.30",
                    help="Window [start:end] in seconds for stim-align summary (default: 0.10:0.30)")
    ap.add_argument("--win_sacc", default="-0.20:0.10",
                    help="Window [start:end] in seconds for sacc-align summary (default: -0.20:0.10)")
    ap.add_argument("--features", nargs="*",
                    help="Features to include. Default: stim→['C','R'], sacc→['S']")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    aligns = []
    if args.align in ("stim", "both"):
        aligns.append("stim")
    if args.align in ("sacc", "both"):
        aligns.append("sacc")

    win_stim = parse_window(args.win_stim)
    win_sacc = parse_window(args.win_sacc)

    for align in aligns:
        sids = find_sessions(out_root, align)
        if not sids:
            print(f"[warn] No sessions found for align={align} under {out_root}")
            continue

        # discover tags if not provided
        if args.tags:
            tags = args.tags
        else:
            tags = discover_tags(out_root, align, sids)
        if not tags:
            print(f"[warn] No flow tags found for align={align}")
            continue

        print(f"[info] align={align}, sessions={len(sids)}, tags={tags}")

        win = win_stim if align == "stim" else win_sacc

        for tag in tags:
            # discover features for this tag if not explicitly provided
            if args.features:
                feats = args.features
            else:
                feats = discover_features(out_root, align, tag, sids)
                if not feats:
                    print(f"[warn] No features found for tag={tag}, align={align}, skipping")
                    continue
            
            print(f"[tag={tag}] features: {feats}")
            
            for feat in feats:
                print(f"\n[tag={tag}] align={align}, feature={feat}")
                summarize_for_tag_align_feature(
                    out_root=out_root,
                    align=align,
                    tag=tag,
                    feature=feat,
                    sids=sids,
                    alpha=args.alpha,
                    win=win,
                )

    print("\n[done] summary + figures completed.")


if __name__ == "__main__":
    main()
