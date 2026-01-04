# Trial Onset Latency Analysis

This folder contains per-trial onset latency scatter plots for **saccade** encoding, aligned to saccade onset.

## Reproduce Command

```bash
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align sacc \
    --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal \
    --orientation_sacc horizontal \
    --tag trialonset_axes_peakbin_saccS
```

## Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--align` | `sacc` | Saccade-aligned analysis |
| `--axes_tag_sacc` | `axes_peakbin_saccS-sacc-horizontal` | Peakbin-optimized encoding axes |
| `--orientation_sacc` | `horizontal` | Use horizontal (left/right) trials |
| `--k_sigma` | `4` (default) | Threshold = baseline_mean + 4σ |
| `--runlen` | `5` (default) | 5 consecutive bins above threshold |
| `--smooth_ms` | `20.0` (default) | Gaussian smoothing sigma |
| `--qc_threshold` | `0.6` (default) | QC AUC threshold |
| `--baseline_sacc` | `-0.35:-0.20` (default) | Baseline window (sec) |
| `--search_sacc` | `-0.30:0.20` (default) | Search window (sec) |

## Output Files

For each monkey (M, S) and area pair (SC-LIP, SC-FEF, FEF-LIP):

- `monkey_{M,S}_{area1}_vs_{area2}_saccade_summary.{png,pdf,svg,npz}` - Saccade onset
- `config.json` - Full configuration parameters

## Interpretation

- **X-axis**: Onset latency in area 1 (ms)
- **Y-axis**: Onset latency in area 2 (ms)
- Points **above** the diagonal (y=x) indicate area 2 encodes later than area 1
- **Red circle**: Mean latency
- **Blue square**: Median latency

## Note on Orientation

For saccade analysis, we use **horizontal** trials (left/right targets) because the saccade direction label (S) distinguishes left vs right saccades, which is most relevant for horizontal target configurations.

