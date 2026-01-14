#!/usr/bin/env python3
"""
Plot QC figures for paper: category (stim) and saccade (sacc) AUC curves.

Summary figures overlay ALL sessions for each monkey, with each session
as a separate curve with varying alpha values.

Monkey M: sessions starting with "2020" (11 sessions)
Monkey S: sessions starting with "2023" (12 sessions)

Output (4 summary figures, one per monkey × condition):
- out/paper_figures/qc/qc_stim_category_M_summary.pdf/.png/.svg
- out/paper_figures/qc/qc_sacc_M_summary.pdf/.png/.svg
- out/paper_figures/qc/qc_stim_category_S_summary.pdf/.png/.svg
- out/paper_figures/qc/qc_sacc_S_summary.pdf/.png/.svg

Usage:
    # Default (10ms workflow)
    python cli/plot_qc_paper.py
    
    # 20ms workflow (use different QC subdirs and output suffix)
    python cli/plot_qc_paper.py \
        --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20msbin \
        --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20msbin \
        --suffix _20msbin \
        --smooth_ms 40
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==================== Figure dimensions (matching panel C) ====================
PLOT_WIDTH_IN = 10.0
PLOT_HEIGHT_IN = 5.0
MARGIN_LEFT_IN = 1.0
MARGIN_RIGHT_IN = 0.5
MARGIN_BOTTOM_IN = 0.8
MARGIN_TOP_IN = 0.5

FIG_WIDTH = PLOT_WIDTH_IN + MARGIN_LEFT_IN + MARGIN_RIGHT_IN
FIG_HEIGHT = PLOT_HEIGHT_IN + MARGIN_BOTTOM_IN + MARGIN_TOP_IN

# ==================== Area configuration ====================
AREA_COLORS = {
    "MFEF": "#0e87cc",
    "MLIP": "#f10c45",
    "SFEF": "#0e87cc",
    "SLIP": "#f10c45",
}

# Areas to plot per monkey (order determines legend order)
AREAS_M = ["MFEF", "MLIP"]
AREAS_S = ["SFEF", "SLIP"]

# Display names (strip monkey prefix)
AREA_LABELS = {
    "MFEF": "FEF",
    "MLIP": "LIP",
    "SFEF": "FEF",
    "SLIP": "LIP",
}

# Smoothing configuration
SMOOTH_MS = 30.0  # smoothing window in milliseconds


def smooth_curve(values: np.ndarray, time_ms: np.ndarray, smooth_ms: float) -> np.ndarray:
    """
    Apply uniform (boxcar) smoothing to a curve.
    
    Parameters
    ----------
    values : np.ndarray
        Values to smooth.
    time_ms : np.ndarray
        Time axis in milliseconds.
    smooth_ms : float
        Smoothing window width in milliseconds.
    
    Returns
    -------
    np.ndarray
        Smoothed values.
    """
    if smooth_ms <= 0 or len(values) < 2:
        return values
    
    # Estimate bin size from time axis
    dt_ms = np.median(np.diff(time_ms))
    if dt_ms <= 0:
        return values
    
    # Number of bins in smoothing window
    n_bins = max(1, int(np.round(smooth_ms / dt_ms)))
    
    if n_bins <= 1:
        return values
    
    # Apply uniform filter (convolution with boxcar)
    kernel = np.ones(n_bins) / n_bins
    # Use 'same' mode and handle edges with 'reflect'
    smoothed = np.convolve(values, kernel, mode='same')
    
    # Fix edge effects by using partial windows at edges
    for i in range(n_bins // 2):
        left_kernel = np.ones(i + 1 + n_bins // 2) / (i + 1 + n_bins // 2)
        smoothed[i] = np.convolve(values[:i + 1 + n_bins // 2], left_kernel, mode='valid')[0]
        
        right_idx = len(values) - 1 - i
        right_kernel = np.ones(i + 1 + n_bins // 2) / (i + 1 + n_bins // 2)
        smoothed[right_idx] = np.convolve(values[right_idx - n_bins // 2:], right_kernel, mode='valid')[-1]
    
    return smoothed


def discover_sessions(out_root: Path, align: str, year_prefix: str) -> List[str]:
    """
    Discover all session IDs for a given alignment and year prefix.
    
    Parameters
    ----------
    out_root : Path
        Root output directory.
    align : str
        Alignment type (e.g., "stim", "sacc").
    year_prefix : str
        Year prefix to filter sessions (e.g., "2020" for Monkey M, "2023" for Monkey S).
    
    Returns
    -------
    List[str]
        Sorted list of session IDs matching the year prefix.
    """
    base_dir = out_root / align
    if not base_dir.exists():
        return []
    
    sessions = []
    for p in sorted(base_dir.iterdir()):
        if p.is_dir() and p.name.startswith(year_prefix) and (p / "caches").is_dir():
            sessions.append(p.name)
    return sessions


def load_qc_json(qc_dir: Path, area: str) -> Optional[dict]:
    """Load QC JSON for an area. Returns None if file doesn't exist."""
    qc_path = qc_dir / f"qc_axes_{area}.json"
    if not qc_path.exists():
        return None
    with open(qc_path, "r") as f:
        return json.load(f)


def load_all_sessions_qc(
    out_root: Path,
    align: str,
    sessions: List[str],
    qc_subdir: str,
    areas: List[str],
) -> Dict[str, Dict[str, dict]]:
    """
    Load QC data for all sessions.
    
    Returns
    -------
    dict
        Nested dict: {session_id: {area: qc_data}}
    """
    all_data = {}
    for sid in sessions:
        qc_dir = out_root / align / sid / "qc" / qc_subdir
        if not qc_dir.exists():
            continue
        
        session_data = {}
        for area in areas:
            qc_data = load_qc_json(qc_dir, area)
            if qc_data is not None:
                session_data[area] = qc_data
        
        if session_data:
            all_data[sid] = session_data
    
    return all_data


def plot_summary_qc_figure(
    out_path: Path,
    all_sessions_data: Dict[str, Dict[str, dict]],
    areas: List[str],
    metric: str,
    ylabel: str,
    xlabel: str,
    t_min_ms: float,
    t_max_ms: float,
    y_min: float = 0.35,
    y_max: float = 1.0,
    chance_level: float = 0.5,
    monkey: str = "",
    smooth_ms: float = 0.0,
) -> None:
    """
    Plot summary QC figure with all sessions overlaid.
    
    Each session gets its own curve with different alpha values.
    Sessions are ranked by peak QC value - highest peak = highest alpha (most visible).
    Areas are distinguished by color.
    
    Parameters
    ----------
    out_path : Path
        Output path for the figure (PDF). Also saves PNG and SVG.
    all_sessions_data : dict
        Nested dict: {session_id: {area: qc_data}}.
    areas : list
        List of area names to plot.
    metric : str
        Metric key in QC data (e.g., "auc_C", "auc_S_inv").
    ylabel : str
        Y-axis label.
    t_min_ms, t_max_ms : float
        Time range in milliseconds.
    y_min, y_max : float
        Y-axis limits.
    chance_level : float
        Horizontal reference line for chance level.
    monkey : str
        Monkey identifier for title.
    smooth_ms : float
        Smoothing window in milliseconds (0 = no smoothing).
    """
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Position axes explicitly (matching panel C style)
    ax = fig.add_axes([
        MARGIN_LEFT_IN / FIG_WIDTH,
        MARGIN_BOTTOM_IN / FIG_HEIGHT,
        PLOT_WIDTH_IN / FIG_WIDTH,
        PLOT_HEIGHT_IN / FIG_HEIGHT
    ])
    
    # Reference lines
    ax.axvline(0, ls="--", c="k", lw=0.8)
    ax.axhline(chance_level, ls=":", c="k", lw=0.8)
    
    sessions = sorted(all_sessions_data.keys())
    n_sessions = len(sessions)
    
    if n_sessions == 0:
        print(f"  [warn] No sessions with data, skipping figure")
        plt.close(fig)
        return
    
    # Compute peak QC value for each session (max across all areas)
    session_peaks = {}
    for sid in sessions:
        session_data = all_sessions_data[sid]
        max_peak = -np.inf
        for area in areas:
            if area not in session_data:
                continue
            values = session_data[area].get(metric)
            if values is not None:
                peak = np.nanmax(np.array(values))
                if peak > max_peak:
                    max_peak = peak
        session_peaks[sid] = max_peak if max_peak > -np.inf else 0.0
    
    # Rank sessions by peak value (ascending order for alpha assignment)
    # Lower rank = lower alpha, higher rank = higher alpha
    sorted_by_peak = sorted(session_peaks.items(), key=lambda x: x[1])
    rank_map = {sid: rank for rank, (sid, _) in enumerate(sorted_by_peak)}
    
    # Compute alpha values based on rank (0.3 to 1.0)
    if n_sessions == 1:
        alpha_map = {sessions[0]: 1.0}
    else:
        alpha_map = {
            sid: 0.3 + 0.7 * rank_map[sid] / (n_sessions - 1)
            for sid in sessions
        }
    
    # Print ranking info
    print(f"  Session ranking by peak {metric}:")
    for sid, peak in sorted(sorted_by_peak, key=lambda x: -x[1]):  # descending for display
        print(f"    {sid}: peak={peak:.3f}, alpha={alpha_map[sid]:.2f}")
    
    # Plot sessions in order of alpha (lowest first, so highest alpha is on top)
    sessions_by_alpha = sorted(sessions, key=lambda s: alpha_map[s])
    
    for sid in sessions_by_alpha:
        session_data = all_sessions_data[sid]
        alpha = alpha_map[sid]
        
        for area in areas:
            if area not in session_data:
                continue
            
            qc_data = session_data[area]
            time_s = np.array(qc_data["time"])
            t_ms = time_s * 1000.0
            
            values = qc_data.get(metric)
            if values is None:
                continue
            values = np.array(values)
            
            # Apply smoothing if requested
            if smooth_ms > 0:
                values = smooth_curve(values, t_ms, smooth_ms)
            
            ax.plot(
                t_ms, values,
                color=AREA_COLORS[area],
                lw=2.0,
                alpha=alpha,
            )
    
    # Create custom legend with full-opacity colors
    from matplotlib.lines import Line2D
    legend_handles = []
    for area in areas:
        legend_handles.append(
            Line2D([0], [0], color=AREA_COLORS[area], lw=2.0, alpha=1.0,
                   label=AREA_LABELS.get(area, area))
        )
    
    # Axis settings
    ax.set_xlim(t_min_ms, t_max_ms)
    ax.set_ylim(y_min, y_max)
    
    # Labels with matched font sizes (from panel C)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=20)
    
    # Add session count annotation
    ax.text(
        0.98, 0.02,
        f"n={n_sessions} sessions",
        transform=ax.transAxes,
        fontsize=18,
        ha="right",
        va="bottom",
        color="gray",
    )
    
    # Save figures
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    
    print(f"Saved: {out_path}")
    print(f"       {out_path.with_suffix('.png')}")
    print(f"       {out_path.with_suffix('.svg')}")
    print(f"       ({n_sessions} sessions plotted)")


def main():
    # ==================== Argument parsing ====================
    ap = argparse.ArgumentParser(
        description="Plot QC figures for paper: category (stim) and saccade (sacc) AUC curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (10ms workflow)
  python cli/plot_qc_paper.py

  # 20ms workflow
  python cli/plot_qc_paper.py \\
      --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20msbin \\
      --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20msbin \\
      --suffix _20msbin \\
      --smooth_ms 40
        """
    )
    ap.add_argument("--out_root", default="out",
                    help="Output root directory (default: out)")
    ap.add_argument("--stim_qc_subdir", default="axes_peakbin_stimCR-stim-vertical",
                    help="QC subdir for stim (default: axes_peakbin_stimCR-stim-vertical)")
    ap.add_argument("--sacc_qc_subdir", default="axes_peakbin_saccS-sacc-horizontal-10msbin",
                    help="QC subdir for sacc (default: axes_peakbin_saccS-sacc-horizontal-10msbin)")
    ap.add_argument("--suffix", default="",
                    help="Suffix for output filenames, e.g., '_20msbin' (default: none)")
    ap.add_argument("--smooth_ms", type=float, default=SMOOTH_MS,
                    help=f"Smoothing window in ms (default: {SMOOTH_MS})")
    ap.add_argument("--t_min_stim", type=float, default=-100.0,
                    help="Stim time range min (ms) (default: -100)")
    ap.add_argument("--t_max_stim", type=float, default=400.0,
                    help="Stim time range max (ms) (default: 400)")
    ap.add_argument("--t_min_sacc", type=float, default=-300.0,
                    help="Sacc time range min (ms) (default: -300)")
    ap.add_argument("--t_max_sacc", type=float, default=100.0,
                    help="Sacc time range max (ms) (default: 100)")
    args = ap.parse_args()
    
    out_root = Path(args.out_root)
    suffix = args.suffix
    smooth_ms = args.smooth_ms
    
    # Output directory for paper figures
    paper_fig_dir = out_root / "paper_figures" / "qc"
    
    # Print configuration
    print(f"[config] out_root: {out_root}")
    print(f"[config] stim_qc_subdir: {args.stim_qc_subdir}")
    print(f"[config] sacc_qc_subdir: {args.sacc_qc_subdir}")
    print(f"[config] suffix: '{suffix}'")
    print(f"[config] smooth_ms: {smooth_ms}")
    
    # Monkey-specific configuration
    monkey_config = {
        "M": {
            "year_prefix": "2020",
            "areas": AREAS_M,
            "stim_qc_subdir": args.stim_qc_subdir,
            "sacc_qc_subdir": args.sacc_qc_subdir,
        },
        "S": {
            "year_prefix": "2023",
            "areas": AREAS_S,
            "stim_qc_subdir": args.stim_qc_subdir,
            "sacc_qc_subdir": args.sacc_qc_subdir,
        },
    }
    
    for monkey, cfg in monkey_config.items():
        year_prefix = cfg["year_prefix"]
        areas = cfg["areas"]
        
        # ==================== Stimulus Category AUC ====================
        print(f"\n{'='*60}")
        print(f"Monkey {monkey}: Stim Category AUC (Summary)")
        print(f"{'='*60}")
        
        # Discover sessions
        stim_sessions = discover_sessions(out_root, "stim", year_prefix)
        print(f"  Found {len(stim_sessions)} sessions for monkey {monkey}")
        
        # Load all QC data
        all_stim_data = load_all_sessions_qc(
            out_root, "stim", stim_sessions,
            cfg["stim_qc_subdir"], areas
        )
        print(f"  Loaded QC data from {len(all_stim_data)} sessions")
        
        for sid in sorted(all_stim_data.keys()):
            areas_loaded = list(all_stim_data[sid].keys())
            print(f"    {sid}: {areas_loaded}")
        
        # Output filename with optional suffix
        stim_outname = f"qc_stim_category_{monkey}{suffix}_summary.pdf"
        plot_summary_qc_figure(
            out_path=paper_fig_dir / stim_outname,
            all_sessions_data=all_stim_data,
            areas=areas,
            metric="auc_C",
            ylabel="AUC (Category)",
            xlabel="Time from Stimulus Onset (ms)",
            t_min_ms=args.t_min_stim,
            t_max_ms=args.t_max_stim,
            y_min=0.35,
            y_max=1.0,
            chance_level=0.5,
            monkey=monkey,
            smooth_ms=smooth_ms,
        )
        
        # ==================== Saccade AUC ====================
        print(f"\n{'='*60}")
        print(f"Monkey {monkey}: Saccade AUC (Summary)")
        print(f"{'='*60}")
        
        # Discover sessions
        sacc_sessions = discover_sessions(out_root, "sacc", year_prefix)
        print(f"  Found {len(sacc_sessions)} sessions for monkey {monkey}")
        
        # Load all QC data
        all_sacc_data = load_all_sessions_qc(
            out_root, "sacc", sacc_sessions,
            cfg["sacc_qc_subdir"], areas
        )
        print(f"  Loaded QC data from {len(all_sacc_data)} sessions")
        
        for sid in sorted(all_sacc_data.keys()):
            areas_loaded = list(all_sacc_data[sid].keys())
            print(f"    {sid}: {areas_loaded}")
        
        # Determine metric (prefer auc_S_inv, fallback to auc_S_raw)
        metric_sacc = "auc_S_inv"
        has_inv = any(
            all_sacc_data.get(sid, {}).get(a, {}).get("auc_S_inv") is not None
            for sid in all_sacc_data
            for a in areas
        )
        if not has_inv:
            metric_sacc = "auc_S_raw"
            print(f"  Using auc_S_raw (auc_S_inv not available)")
        else:
            print(f"  Using auc_S_inv (preferred for flow analysis)")
        
        # Output filename with optional suffix
        sacc_outname = f"qc_sacc_{monkey}{suffix}_summary.pdf"
        plot_summary_qc_figure(
            out_path=paper_fig_dir / sacc_outname,
            all_sessions_data=all_sacc_data,
            areas=areas,
            metric=metric_sacc,
            ylabel="AUC (Saccade)",
            xlabel="Time from Saccade Onset (ms)",
            t_min_ms=args.t_min_sacc,
            t_max_ms=args.t_max_sacc,
            y_min=0.35,
            y_max=1.0,
            chance_level=0.5,
            monkey=monkey,
            smooth_ms=smooth_ms,
        )
    
    print(f"\n{'='*60}")
    print("[done] QC paper figures (summary) completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
