# Summarize Flow Across Sessions

This script aggregates information flow results across all sessions for each area pair and monkey, producing summary statistics and publication-ready figures.

## Quick Start

### Stimulus-aligned (Category encoding, lag 50ms and 80ms)

```bash
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags evoked_peakbin_stimC_vertical_lag50ms-none-trial evoked_peakbin_stimC_vertical_lag80ms-none-trial
```

### Saccade-aligned (Saccade encoding, lag 30ms and 50ms)

```bash
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag30ms-none-trial evoked_peakbin_saccS_horizontal_lag50ms-none-trial
```

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--qc_threshold` | `0.6` | QC AUC threshold for area filtering (symmetric rejection) |
| `--group_diff_p` | `True` | Compute group-level p-values for net flow |
| `--smooth_ms` | `30.0` | Smoothing window (ms) for group p-value computation |
| `--sacc_bin_combine` | `1` | Combine adjacent bins for sacc (1=no combining, 2=5ms→10ms) |
| `--alpha` | `0.05` | Significance threshold |
| `--win_stim` | `0.10:0.30` | Summary window for stim alignment (sec) |
| `--win_sacc` | `-0.20:0.10` | Summary window for sacc alignment (sec) |

## Output Structure

```
out/<align>/summary/<tag>/<feature>/
├── summary_MFEF_vs_MLIP.npz      # Summary statistics (monkey M)
├── summary_MFEF_vs_MSC.npz
├── summary_MLIP_vs_MSC.npz
├── summary_SFEF_vs_SLIP.npz      # Summary statistics (monkey S)
├── summary_SFEF_vs_SSC.npz
├── summary_SLIP_vs_SSC.npz
└── figs/
    ├── MFEF_vs_MLIP.pdf          # Full summary plot (4 panels)
    ├── MFEF_vs_MLIP.png
    ├── MFEF_vs_MLIP_panel_a.pdf  # Paper-quality: bits ± SE
    ├── MFEF_vs_MLIP_panel_a.png
    ├── MFEF_vs_MLIP_panel_a.svg
    ├── MFEF_vs_MLIP_panel_c.pdf  # Paper-quality: net flow ± SE
    ├── MFEF_vs_MLIP_panel_c.png
    ├── MFEF_vs_MLIP_panel_c.svg
    └── ... (same for all pairs, both directions)
```

## Figure Descriptions

### Full Summary Plot (`*_vs_*.pdf`)
4-panel figure showing:
1. **Panel 1**: Bits (ΔLL) ± SE for both directions
2. **Panel 2**: Z-scores ± SE for both directions
3. **Panel 3**: Net flow (diff bits) ± SE with significance dots
4. **Panel 4**: Rebinned z-scores (if `--rebin_win` specified)

### Paper-Quality Figures

**Panel A** (`*_panel_a.pdf`):
- Bits (ΔLL) ± SE for both flow directions
- Time range: -100 to 500 ms (stim) or -300 to 200 ms (sacc)
- Aspect ratio 2:1, matched height to scatter plots

**Panel C** (`*_panel_c.pdf`):
- Net flow ± SE with significance dots (black)
- Y-axis: -10 to 20 (monkey M), -8 to 13 (monkey S)
- Significance dots show p < α from group-level permutation test

## Y-Axis Limits by Monkey

| Monkey | Feature | Y-min | Y-max |
|--------|---------|-------|-------|
| M | C, R, S | -10 | 20 |
| S | C, R, S | -8 | 13 |

## Bin Combining for Saccade Data

Saccade-aligned data may use 5ms or 10ms bins depending on workflow. By default, `--sacc_bin_combine 1` (no combining).

If using the **5ms workflow** and want 10ms output, set `--sacc_bin_combine 2`:
- Original: 100 bins × 5ms = 500ms
- Combined: 50 bins × 10ms = 500ms

If using the **10ms workflow** (with `--rebin_factor 2`), keep the default `--sacc_bin_combine 1`.

## QC Filtering

**Symmetric rejection**: For each area pair (A, B), if EITHER area fails QC for the feature, the entire session is excluded for that pair. This ensures both directions (A→B and B→A) have the same N.

To disable QC filtering: `--qc_threshold 0`

## Group-Level P-Values

The significance dots in Panel C are computed using a group-level permutation test:
1. For each session, null samples are loaded from flow_*.npz
2. Null difference is computed: null_AtoB - null_BtoA
3. Group null distribution: randomly sample one null from each session, average
4. P-value: fraction of group null ≥ observed mean difference
5. Smoothing applied to both observed and null before comparison

To disable: `--no_group_diff_p`

## Advanced Options

```bash
# Process all tags starting with "evoked" (default)
python cli/summarize_flow_across_sessions.py --out_root out --align stim

# Process specific QC tag instead of auto-detection
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags evoked_peakbin_stimC_vertical_lag50ms-none-trial \
    --qc_tag axes_peakbin_stimCR-stim-vertical

# Custom significance threshold and smoothing
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag50ms-none-trial \
    --alpha 0.01 \
    --smooth_ms 50

# Add rebinned z-score panel (panel 4)
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags evoked_peakbin_stimC_vertical_lag50ms-none-trial \
    --rebin_win 0.05 \
    --rebin_step 0.02
```

## 10ms Bin Workflow for Saccade

If you want to use 10ms bins (instead of 5ms) from the very beginning of the pipeline for saccade-aligned data, use the `--rebin_factor 2` option throughout:

### Run the 10ms pipeline (SLURM)

```bash
# Lag 30ms
sbatch jobs/peakbin_axes_qc_flow_array_10ms.sbatch

# Lag 50ms
sbatch jobs/peakbin_axes_qc_flow_array_10ms_big.sbatch
```

### Run locally (single session)

```bash
# Step 1: Train axes with rebinning
python cli/train_axes.py \
    --out_root out --align sacc --sid 20231130 \
    --orientation horizontal --features S \
    --tag axes_peakbin_saccS-sacc-horizontal-10msbin \
    --searchS --search_range_S="-0.20:0.05" \
    --search_score_mode peak_bin_auc \
    --rebin_factor 2

# Step 2: QC with rebinning
python cli/qc_axes.py \
    --out_root out --align sacc --sid 20231130 \
    --orientation horizontal \
    --tag axes_peakbin_saccS-sacc-horizontal-10msbin \
    --rebin_factor 2

# Step 3: Flow with rebinning
python cli/flow_session.py \
    --out_root out --align sacc --sid 20231130 \
    --feature S --orientation horizontal \
    --lags_ms 30 \
    --axes_tag axes_peakbin_saccS-sacc-horizontal-10msbin \
    --tag evoked_peakbin_saccS_horizontal_lag30ms_10msbin \
    --flow_tag_base evoked_peakbin_saccS_horizontal_lag30ms_10msbin \
    --save_null_samples \
    --rebin_factor 2
```

### Summarize 10ms results

```bash
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag30ms_10msbin-none-trial evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial
# No --sacc_bin_combine needed - default is 1 (no combining)
```

### Key differences from 5ms workflow

| Aspect | 5ms Workflow | 10ms Workflow |
|--------|--------------|---------------|
| Bin size | 5ms (original) | 10ms (rebinned) |
| `--rebin_factor` | 1 (default) | 2 |
| Tag suffix | (none) | `-10msbin` |
| Time points | ~100 bins | ~50 bins |
| Summary `--sacc_bin_combine` | 2 (to get 10ms) | 1 (default) |

## NPZ File Contents

Each `summary_*.npz` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `time` | (T,) | Time points in seconds |
| `mean_bits_AtoB` | (T,) | Mean bits A→B across sessions |
| `se_bits_AtoB` | (T,) | SE of bits A→B |
| `mean_bits_BtoA` | (T,) | Mean bits B→A across sessions |
| `se_bits_BtoA` | (T,) | SE of bits B→A |
| `mean_diff_bits` | (T,) | Mean net flow (A→B minus B→A) |
| `se_diff_bits` | (T,) | SE of net flow |
| `p_group_diff` | (T,) | Group-level p-values for net flow |
| `sig_group_diff` | (T,) | Binary significance mask (p < α) |
| `session_ids` | (N,) | List of session IDs included |
| `meta_json` | str | JSON with config (tag, alpha, N, etc.) |

