# Trial Onset Latency Analysis

This folder contains per-trial onset latency scatter plots for **category** and **direction** encoding, aligned to stimulus onset.

## Reproduce Command

```bash
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align stim \
    --axes_tag_stim axes_peakbin_stimCR-stim-vertical \
    --orientation_stim vertical \
    --tag trialonset_axes_peakbin_stimCR
```

## Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--align` | `stim` | Stimulus-aligned analysis |
| `--axes_tag_stim` | `axes_peakbin_stimCR-stim-vertical` | Peakbin-optimized encoding axes |
| `--orientation_stim` | `vertical` | Use vertical (up/down) trials |
| `--k_sigma` | `4` (default) | Threshold = baseline_mean + 4σ |
| `--runlen` | `5` (default) | 5 consecutive bins above threshold |
| `--smooth_ms` | `20.0` (default) | Gaussian smoothing sigma |
| `--qc_threshold` | `0.6` (default) | QC AUC threshold |
| `--baseline_stim` | `-0.20:0.00` (default) | Baseline window (sec) |
| `--search_stim` | `0.00:0.50` (default) | Search window (sec) |

## Output Files

For each monkey (M, S) and area pair (SC-LIP, SC-FEF, FEF-LIP):

- `monkey_{M,S}_{area1}_vs_{area2}_category_summary.{png,pdf,svg,npz}` - Category onset
- `monkey_{M,S}_{area1}_vs_{area2}_direction_summary.{png,pdf,svg,npz}` - Direction onset
- `config.json` - Full configuration parameters

## Interpretation

- **X-axis**: Onset latency in area 1 (ms)
- **Y-axis**: Onset latency in area 2 (ms)
- Points **above** the diagonal (y=x) indicate area 2 encodes later than area 1
- **Red circle**: Mean latency
- **Blue square**: Median latency

