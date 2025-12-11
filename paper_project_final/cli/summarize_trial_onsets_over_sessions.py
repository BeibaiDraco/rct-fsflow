#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_npz(p: Path) -> Dict:
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    if "meta" in out and not isinstance(out["meta"], dict):
        try:
            import json
            out["meta"] = json.loads(out["meta"].item())
        except Exception:
            pass
    return out


def find_onset_files(out_root: Path, align: str, tag: str) -> List[Tuple[str, Path]]:
    """Find all onset NPZ files, return list of (sid, path) tuples."""
    files = []
    base_dir = out_root / align
    if not base_dir.exists():
        return files
    
    for sid_dir in base_dir.iterdir():
        if not sid_dir.is_dir():
            continue
        sid = sid_dir.name
        npz_dir = sid_dir / "trialtiming" / tag
        if not npz_dir.exists():
            continue
        
        # Find NPZ files matching the pattern
        for npz_file in npz_dir.glob("*_onset_*.npz"):
            files.append((sid, npz_file))
    
    return files


def get_monkey(sid: str) -> str:
    """Return 'M' for sessions starting with 2020, 'S' for 2023."""
    if sid.startswith("2020"):
        return "M"
    elif sid.startswith("2023"):
        return "S"
    else:
        return "Unknown"


def aggregate_trials_by_monkey(files: List[Tuple[str, Path]]) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Aggregate trials by monkey. Returns dict: monkey -> (t_fef_ms, t_lip_ms, sids)"""
    monkey_data = {"M": ([], [], []), "S": ([], [], [])}
    
    for sid, npz_path in files:
        monkey = get_monkey(sid)
        if monkey not in monkey_data:
            continue
        
        try:
            data = load_npz(npz_path)
            t_fef = data.get("t_fef", np.array([]))
            t_lip = data.get("t_lip", np.array([]))
            good_mask = data.get("good_mask", np.ones(len(t_fef), dtype=bool))
            
            if isinstance(good_mask, np.ndarray):
                good = good_mask.astype(bool)
            else:
                good = np.ones(len(t_fef), dtype=bool)
            
            # Convert to ms and filter good trials
            t_fef_ms = t_fef[good] * 1000.0
            t_lip_ms = t_lip[good] * 1000.0
            
            # Only keep finite values
            valid = np.isfinite(t_fef_ms) & np.isfinite(t_lip_ms)
            t_fef_ms = t_fef_ms[valid]
            t_lip_ms = t_lip_ms[valid]
            
            if len(t_fef_ms) > 0:
                monkey_data[monkey][0].append(t_fef_ms)
                monkey_data[monkey][1].append(t_lip_ms)
                monkey_data[monkey][2].append(sid)
        except Exception as e:
            print(f"[warning] Failed to load {npz_path}: {e}")
            continue
    
    # Concatenate all trials for each monkey
    result = {}
    for monkey, (fef_list, lip_list, sids) in monkey_data.items():
        if len(fef_list) > 0:
            t_fef_all = np.concatenate(fef_list)
            t_lip_all = np.concatenate(lip_list)
            result[monkey] = (t_fef_all, t_lip_all, sids)
    
    return result


def main():
    ap = argparse.ArgumentParser(description="Summarize trial onset times across sessions by monkey.")
    ap.add_argument("--out_root", default="out")
    ap.add_argument("--align", choices=["stim"], default="stim")
    ap.add_argument("--tag", default="trialonset_v1",
                    help="Input subfolder name (default: trialonset_v1)")
    ap.add_argument("--out_tag", default="trialonset_summary",
                    help="Output subfolder name (default: trialonset_summary)")
    args = ap.parse_args()
    
    out_root = Path(args.out_root)
    align = args.align
    
    # Find all onset files
    files = find_onset_files(out_root, align, args.tag)
    if len(files) == 0:
        raise SystemExit(f"No onset files found in {out_root / align}/*/trialtiming/{args.tag}/")
    
    print(f"[info] Found {len(files)} onset files")
    
    # Group by monkey
    monkey_data = aggregate_trials_by_monkey(files)
    
    if len(monkey_data) == 0:
        raise SystemExit("No valid data found for any monkey")
    
    # Create output directory
    out_dir = out_root / align / "trialtiming" / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot for each monkey
    for monkey, (t_fef_ms, t_lip_ms, sids) in monkey_data.items():
        if len(t_fef_ms) == 0:
            continue
        
        # Calculate statistics
        dt_ms = t_lip_ms - t_fef_ms
        median_dt = np.nanmedian(dt_ms)
        mean_dt = np.nanmean(dt_ms)
        n_trials = len(t_fef_ms)
        n_sessions = len(set(sids))
        
        print(f"[{monkey}] {n_trials} trials from {n_sessions} sessions | median(LIP-FEF) = {median_dt:.1f} ms | mean = {mean_dt:.1f} ms")
        
        # Create scatter plot
        fig = plt.figure(figsize=(7.0, 6.5))
        ax = fig.add_subplot(1, 1, 1)
        
        # Scatter plot
        ax.plot(t_fef_ms, t_lip_ms, "k.", ms=2, alpha=0.4, label=f"N={n_trials} trials")
        
        # Mark mean and median (without legend labels to avoid overlap)
        mean_fef = np.nanmean(t_fef_ms)
        mean_lip = np.nanmean(t_lip_ms)
        median_fef = np.nanmedian(t_fef_ms)
        median_lip = np.nanmedian(t_lip_ms)
        
        ax.plot(mean_fef, mean_lip, "ro", ms=10, markerfacecolor="red", 
                markeredgecolor="darkred", markeredgewidth=2, zorder=5)
        ax.plot(median_fef, median_lip, "bs", ms=10, markerfacecolor="blue", 
                markeredgecolor="darkblue", markeredgewidth=2, zorder=5)
        
        # Diagonal line
        lo = np.nanmin(np.r_[t_fef_ms, t_lip_ms])
        hi = np.nanmax(np.r_[t_fef_ms, t_lip_ms])
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x", alpha=0.7)
        
        # Labels and title
        monkey_name = "Monkey M" if monkey == "M" else "Monkey S"
        area_prefix = "M" if monkey == "M" else "S"
        ax.set_xlabel(f"{area_prefix}FEF onset time (ms)", fontsize=12)
        ax.set_ylabel(f"{area_prefix}LIP onset time (ms)", fontsize=12)
        
        title = f"{monkey_name} ({area_prefix}FEF vs {area_prefix}LIP)\n"
        title += f"{n_sessions} sessions, {n_trials} trials\n"
        title += f"mean: ({mean_fef:.1f}, {mean_lip:.1f}) ms | median: ({median_fef:.1f}, {median_lip:.1f}) ms"
        ax.set_title(title, fontsize=12)
        
        # Add statistics text
        stats_text = f"median(LIP-FEF) = {median_dt:.1f} ms\n"
        stats_text += f"mean(LIP-FEF) = {mean_dt:.1f} ms"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(frameon=False, loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()
        
        # Save
        out_png = out_dir / f"monkey_{monkey}_onset_summary.png"
        out_pdf = out_dir / f"monkey_{monkey}_onset_summary.pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[ok] wrote {out_png} and {out_pdf}")
        
        # Save data
        out_npz = out_dir / f"monkey_{monkey}_onset_summary.npz"
        np.savez_compressed(
            out_npz,
            t_fef_ms=t_fef_ms,
            t_lip_ms=t_lip_ms,
            dt_ms=dt_ms,
            sids=np.array(sids),
            meta=dict(
                monkey=monkey,
                n_trials=n_trials,
                n_sessions=n_sessions,
                median_dt_ms=float(median_dt),
                mean_dt_ms=float(mean_dt),
            )
        )
        print(f"[ok] wrote {out_npz}")


if __name__ == "__main__":
    main()

