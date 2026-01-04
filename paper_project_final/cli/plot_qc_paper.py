#!/usr/bin/env python3
"""
Plot QC figures for paper: category (stim) and saccade (sacc) AUC curves.

Monkey M (session 20201211):
- Stim category: out/stim/20201211/qc/axes_peakbin_stimCR-stim-vertical
- Saccade: out/sacc/20201211/qc/axes_peakbin_saccS-sacc-horizontal-10msbin

Monkey S (session 20231123):
- Stim category: out/stim/20231123/qc/axes_peakbin_stimCR-stim-vertical
- Saccade: out/sacc/20231123/qc/axes_peakbin_saccS-sacc-horizontal-10msbin

Output:
- out/paper_figures/qc/qc_stim_category_M.pdf/.png/.svg
- out/paper_figures/qc/qc_sacc_M.pdf/.png/.svg
- out/paper_figures/qc/qc_stim_category_S.pdf/.png/.svg
- out/paper_figures/qc/qc_sacc_S.pdf/.png/.svg

Usage:
    python cli/plot_qc_paper.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==================== Figure dimensions (matching panel C) ====================
# These match plot_panel_c_paper in summarize_flow_across_sessions.py
PLOT_WIDTH_IN = 10.0
PLOT_HEIGHT_IN = 5.0
MARGIN_LEFT_IN = 1.0
MARGIN_RIGHT_IN = 0.5
MARGIN_BOTTOM_IN = 0.8
MARGIN_TOP_IN = 0.5

FIG_WIDTH = PLOT_WIDTH_IN + MARGIN_LEFT_IN + MARGIN_RIGHT_IN
FIG_HEIGHT = PLOT_HEIGHT_IN + MARGIN_BOTTOM_IN + MARGIN_TOP_IN

# ==================== Area configuration ====================
# Colors as specified
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


def load_qc_json(qc_dir: Path, area: str) -> dict:
    """Load QC JSON for an area."""
    qc_path = qc_dir / f"qc_axes_{area}.json"
    with open(qc_path, "r") as f:
        return json.load(f)


def plot_qc_figure(
    out_path: Path,
    qc_data_by_area: dict,
    areas: list,
    metric: str,
    ylabel: str,
    t_min_ms: float,
    t_max_ms: float,
    y_min: float = 0.35,
    y_max: float = 1.0,
    chance_level: float = 0.5,
) -> None:
    """
    Plot QC figure with multiple areas, matching panel C style.
    
    Parameters
    ----------
    out_path : Path
        Output path for the figure (PDF). Also saves PNG and SVG.
    qc_data_by_area : dict
        Dictionary mapping area name to QC data dict.
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
    
    # Plot each area
    for area in areas:
        if area not in qc_data_by_area:
            print(f"  [warn] {area}: not in data, skipping")
            continue
        
        qc_data = qc_data_by_area[area]
        time_s = np.array(qc_data["time"])
        t_ms = time_s * 1000.0
        
        values = qc_data.get(metric)
        if values is None:
            print(f"  [warn] {area}: metric '{metric}' is None, skipping")
            continue
        values = np.array(values)
        
        label = AREA_LABELS.get(area, area)
        ax.plot(t_ms, values, color=AREA_COLORS[area], lw=3, label=label)
    
    # Axis settings
    ax.set_xlim(t_min_ms, t_max_ms)
    ax.set_ylim(y_min, y_max)
    
    # Labels with matched font sizes (from panel C)
    ax.set_xlabel("Time (ms)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc="upper left", frameon=False, fontsize=20)
    
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


def main():
    out_root = Path("out")
    
    # Output directory for paper figures
    paper_fig_dir = out_root / "paper_figures" / "qc"
    
    # Monkey-specific configuration
    monkey_config = {
        "M": {
            "sid": "20201211",
            "areas": AREAS_M,
            "stim_qc_subdir": "axes_peakbin_stimCR-stim-vertical",
            "sacc_qc_subdir": "axes_peakbin_saccS-sacc-horizontal-10msbin",
        },
        "S": {
            "sid": "20231123",
            "areas": AREAS_S,
            "stim_qc_subdir": "axes_peakbin_stimCR-stim-vertical",
            "sacc_qc_subdir": "axes_peakbin_saccS-sacc-horizontal-10msbin",
        },
    }
    
    for monkey, cfg in monkey_config.items():
        sid = cfg["sid"]
        areas = cfg["areas"]
        
        # ==================== Stimulus Category AUC ====================
        qc_dir_stim = out_root / "stim" / sid / "qc" / cfg["stim_qc_subdir"]
        
        print(f"\n{'='*60}")
        print(f"Monkey {monkey}: Stim Category AUC (from {qc_dir_stim})")
        print(f"{'='*60}")
        
        qc_data_stim = {}
        for area in areas:
            try:
                qc_data_stim[area] = load_qc_json(qc_dir_stim, area)
                print(f"  Loaded {area}")
            except FileNotFoundError as e:
                print(f"  [warn] {area}: {e}")
        
        plot_qc_figure(
            out_path=paper_fig_dir / f"qc_stim_category_{monkey}.pdf",
            qc_data_by_area=qc_data_stim,
            areas=areas,
            metric="auc_C",
            ylabel="AUC (Category)",
            t_min_ms=-100.0,
            t_max_ms=500.0,
            y_min=0.35,
            y_max=1.0,
            chance_level=0.5,
        )
        
        # ==================== Saccade AUC ====================
        qc_dir_sacc = out_root / "sacc" / sid / "qc" / cfg["sacc_qc_subdir"]
        
        print(f"\n{'='*60}")
        print(f"Monkey {monkey}: Saccade AUC (from {qc_dir_sacc})")
        print(f"{'='*60}")
        
        qc_data_sacc = {}
        for area in areas:
            try:
                qc_data_sacc[area] = load_qc_json(qc_dir_sacc, area)
                print(f"  Loaded {area}")
            except FileNotFoundError as e:
                print(f"  [warn] {area}: {e}")
        
        # Use auc_S_inv (preferred for flow analysis, per summarize_flow_across_sessions.py)
        # Falls back to auc_S_raw if inv is not available
        metric_sacc = "auc_S_inv"
        has_inv = any(
            qc_data_sacc.get(a, {}).get("auc_S_inv") is not None 
            for a in areas if a in qc_data_sacc
        )
        if not has_inv:
            metric_sacc = "auc_S_raw"
            print(f"  Using auc_S_raw (auc_S_inv not available)")
        else:
            print(f"  Using auc_S_inv (preferred for flow analysis)")
        
        plot_qc_figure(
            out_path=paper_fig_dir / f"qc_sacc_{monkey}.pdf",
            qc_data_by_area=qc_data_sacc,
            areas=areas,
            metric=metric_sacc,
            ylabel="AUC (Saccade)",
            t_min_ms=-290.0,
            t_max_ms=200.0,
            y_min=0.35,
            y_max=1.0,
            chance_level=0.5,
        )
    
    print(f"\n{'='*60}")
    print("[done] QC paper figures completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

