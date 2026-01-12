# Paper Figures Documentation

This README documents the six panel figures used in the paper and how to reproduce them.

---

## Overview of Figures

The paper contains **6 panels**, each with **2 plots** (Monkey M on top, Monkey S on bottom):

| Panel | Description | Monkey M Figure | Monkey S Figure |
|-------|-------------|-----------------|-----------------|
| A | QC: Category AUC over time (stim-aligned) | `out/paper_figures/qc/qc_stim_category_M_summary.png` | `out/paper_figures/qc/qc_stim_category_S_summary.png` |
| B | Trial onset latency scatter: Category (FEF vs LIP) | `out/stim/trialtiming/trialonset_axes_peakbin_stimCR/monkey_M_FEF_vs_LIP_category_summary.png` | `out/stim/trialtiming/trialonset_axes_peakbin_stimCR/monkey_S_FEF_vs_LIP_category_summary.png` |
| C | Directed predictability gain: Category (stim-aligned) | `out/stim/summary/evoked_peakbin_stimC_vertical_lag80ms-none-trial/C/figs/MFEF_vs_MLIP_panel_d.png` | `out/stim/summary/evoked_peakbin_stimC_vertical_lag80ms-none-trial/C/figs/SFEF_vs_SLIP_panel_d.png` |
| D | QC: Saccade AUC over time (sacc-aligned) | `out/paper_figures/qc/qc_sacc_M_summary.png` | `out/paper_figures/qc/qc_sacc_S_summary.png` |
| E | Trial onset latency scatter: Saccade (FEF vs LIP) | `out/sacc/trialtiming/trialonset_axes_peakbin_saccS_10msbin/monkey_M_FEF_vs_LIP_saccade_summary.png` | `out/sacc/trialtiming/trialonset_axes_peakbin_saccS_10msbin/monkey_S_FEF_vs_LIP_saccade_summary.png` |
| F | Directed predictability gain: Saccade (sacc-aligned) | `out/sacc/summary/evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial/S/figs/MFEF_vs_MLIP_panel_d.png` | `out/sacc/summary/evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial/S/figs/SFEF_vs_SLIP_panel_d.png` |

---

## Panel A & D: QC AUC Figures

### Description
These figures show the Area Under the Curve (AUC) for decoding accuracy over time:
- **Panel A (Category AUC)**: How well category (left vs right) can be decoded from neural activity (stim-aligned)
- **Panel D (Saccade AUC)**: How well saccade direction can be decoded from neural activity (sacc-aligned)
- Higher AUC indicates stronger encoding of the feature
- Chance level is 0.5 (dashed horizontal line)
- All sessions for each monkey are overlaid with varying alpha (higher alpha = higher peak AUC)

### Source Script
`cli/plot_qc_paper.py`

### Input Data Location
- Stim category QC: `out/stim/<session_id>/qc/axes_peakbin_stimCR-stim-vertical/qc_axes_<area>.json`
- Sacc saccade QC: `out/sacc/<session_id>/qc/axes_peakbin_saccS-sacc-horizontal-10msbin/qc_axes_<area>.json`

### Output Location
- Panel A (Category): `out/paper_figures/qc/qc_stim_category_{M,S}_summary.{pdf,png,svg}`
- Panel D (Saccade): `out/paper_figures/qc/qc_sacc_{M,S}_summary.{pdf,png,svg}`

### How to Reproduce

```bash
python cli/plot_qc_paper.py
```

### Key Parameters

| Parameter | Panel A (Category/Stim) | Panel D (Saccade/Sacc) |
|-----------|-------------------------|------------------------|
| alignment | stim | sacc |
| qc_subdir | `axes_peakbin_stimCR-stim-vertical` | `axes_peakbin_saccS-sacc-horizontal-10msbin` |
| metric | `auc_C` | `auc_S_inv` |
| time range | -100 to 500 ms | -290 to 200 ms |
| y-axis limits | 0.35 to 1.0 | 0.35 to 1.0 |
| chance level | 0.5 | 0.5 |
| smoothing | 30 ms | 30 ms |

### Area Colors
| Area | Color | Hex Code |
|------|-------|----------|
| FEF | Blue | #0e87cc |
| LIP | Red | #f10c45 |

### Figure Specifications
- **Plot area**: 10 × 5 inches
- **Figure size**: 11.5 × 6.3 inches (with margins)
- **Font sizes**: xlabel 18pt, ylabel 20pt, tick labels 18pt, legend 20pt
- **Line width**: 2pt

---

## Panel B & E: Trial Onset Latency Scatter Plots

### Description
These scatter plots show per-trial latency of encoding onset:
- **X-axis**: Latency in FEF
- **Y-axis**: Latency in LIP
- Points above the diagonal indicate LIP encodes later than FEF
- Red circle: mean latency; Blue square: median latency

### Source Script
`cli/trial_onset_comprehensive.py`

### Input Data Location
- Axes: `out/<align>/<session_id>/axes/<axes_tag>/axes_<area>.npz`
- Neural caches: `out/<align>/<session_id>/caches/area_<area>.npz`

### Output Location
- Panel B (Category): `out/stim/trialtiming/trialonset_axes_peakbin_stimCR/monkey_{M,S}_FEF_vs_LIP_category_summary.{png,pdf,svg,npz}`
- Panel E (Saccade): `out/sacc/trialtiming/trialonset_axes_peakbin_saccS_10msbin/monkey_{M,S}_FEF_vs_LIP_saccade_summary.{png,pdf,svg,npz}`

### How to Reproduce

**Panel B (Category, stim-aligned):**
```bash
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align stim \
    --axes_tag_stim axes_peakbin_stimCR-stim-vertical \
    --orientation_stim vertical \
    --tag trialonset_axes_peakbin_stimCR
```

**Panel E (Saccade, sacc-aligned, 10ms bins):**
```bash
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align sacc \
    --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-10msbin \
    --orientation_sacc horizontal \
    --tag trialonset_axes_peakbin_saccS_10msbin
```

### Key Parameters

| Parameter | Panel B (Category/Stim) | Panel E (Saccade/Sacc) |
|-----------|-------------------------|------------------------|
| alignment | stim | sacc |
| orientation | vertical | horizontal |
| axes_tag | `axes_peakbin_stimCR-stim-vertical` | `axes_peakbin_saccS-sacc-horizontal-10msbin` |
| feature | C (category) | S (saccade) |
| baseline window | -0.20 to 0.00 s | -0.35 to -0.20 s |
| search window | 0.00 to 0.50 s | -0.20 to 0.20 s |
| threshold | baseline_mean + 4σ | baseline_mean + 4σ |
| runlen | 5 consecutive bins | 5 consecutive bins |
| smoothing | 20 ms Gaussian | 20 ms Gaussian |
| QC threshold | 0.6 | 0.6 |

### Onset Detection Algorithm
1. Project neural activity onto encoding axis (`sC` for category, `sS_inv` for saccade)
2. Sign activity by trial label (e.g., multiply by ±1 based on category)
3. Smooth with Gaussian kernel (σ = 20 ms)
4. Calculate threshold: `baseline_mean + k_sigma * baseline_std` per trial
5. Find first time point in search window where activity exceeds threshold for `runlen` consecutive bins

### Figure Specifications
- **Plot area**: 5 × 5 inches (square)
- **Figure size**: 6.5 × 6.5 inches (with margins)
- **Axis limits**: Based on search window (stim: 0-500 ms, sacc: -200 to 200 ms)
- **Font sizes**: xlabel/ylabel 18pt, tick labels 16pt, legend 15pt
- **Markers**: scatter dots 7pt, mean (red circle) 12pt, median (blue square) 12pt

### Statistics Reported
- **Median dt**: Median of (LIP_latency - FEF_latency) across trials
- **Mean dt**: Mean of (LIP_latency - FEF_latency) across trials
- **p (two-sided)**: Sign-flip permutation test (20,000 permutations)
- **p (area2 later)**: One-sided test for LIP encoding later

---

## Panel C & F: Directed Predictability Gain Figures

### Description
These figures show the directed predictability gain (bits/trial/dim, df-corrected) between brain areas:
- Shows both directions: FEF→LIP and LIP→FEF
- y=0 means "no more improvement than what's expected from model complexity alone"
- Significance dots (black) mark time points where p < α (group-level permutation test)

### Source Script
`cli/summarize_flow_across_sessions.py`

### Generating Function
`plot_panel_d_paper()` in the summarize script

### Input Data Location
The script reads flow results from:
- Stim-aligned: `out/stim/<session_id>/flow/evoked_peakbin_stimC_vertical_lag80ms-none-trial/C/flow_C_<area1>to<area2>.npz`
- Sacc-aligned: `out/sacc/<session_id>/flow/evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial/S/flow_S_<area1>to<area2>.npz`

### Output Location
- Panel C (Category): `out/stim/summary/evoked_peakbin_stimC_vertical_lag80ms-none-trial/C/figs/{M,S}FEF_vs_{M,S}LIP_panel_d.{pdf,png,svg}`
- Panel F (Saccade): `out/sacc/summary/evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial/S/figs/{M,S}FEF_vs_{M,S}LIP_panel_d.{pdf,png,svg}`

### How to Reproduce

**Panel C (Category, stim-aligned):**
```bash
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags evoked_peakbin_stimC_vertical_lag80ms-none-trial
```

**Panel F (Saccade, sacc-aligned, 10ms bins):**
```bash
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial
```

### Key Parameters

| Parameter | Panel C (Category/Stim) | Panel F (Saccade/Sacc) |
|-----------|-------------------------|------------------------|
| alignment | stim | sacc |
| flow_tag | `evoked_peakbin_stimC_vertical_lag80ms-none-trial` | `evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial` |
| feature | C (category) | S (saccade) |
| lag | 80 ms | 50 ms |
| bin size | 10 ms | 10 ms |
| time range | -100 to 500 ms | -290 to 200 ms |
| QC threshold | 0.6 | 0.6 |
| alpha | 0.05 | 0.05 |
| smoothing | 30 ms | 30 ms |

### Figure Specifications
- **Plot area**: 10 × 5 inches
- **Figure size**: 11.9 × 6.3 inches (with margins)
- **Font sizes**: xlabel/ylabel 18pt, tick labels 16pt, legend 15pt
- **Line width**: 2.5pt
- **Colors**: C0 (blue) for FEF→LIP, C1 (orange) for LIP→FEF
- **Reference lines**: dashed vertical at t=0, dotted horizontal at y=0

---

## Prerequisite: Session Data

### Session List
All sessions are listed in `sid_list.txt`:

**Monkey M (2020)**: 20200327, 20200328, 20200401, 20200402, 20200926, 20200929, 20201001, 20201004, 20201204, 20201211, 20201216

**Monkey S (2023)**: 20230622, 20230627, 20230705, 20230707, 20230710, 20231025, 20231103, 20231109, 20231121, 20231123, 20231130, 20231205

### Required Preprocessing Steps

1. **Build caches** (neural data preprocessing):
   ```bash
   python cli/build_caches_all.py --align stim --sid_list sid_list.txt
   python cli/build_caches_all.py --align sacc --sid_list sid_list.txt
   ```

2. **Train encoding axes**:
   ```bash
   python cli/train_axes.py --align stim --orientation vertical --features C R
   python cli/train_axes.py --align sacc --orientation horizontal --features S
   ```

3. **Run QC** (quality control):
   ```bash
   python cli/qc_axes.py --align stim --axes_tag axes_peakbin_stimCR-stim-vertical
   python cli/qc_axes.py --align sacc --axes_tag axes_peakbin_saccS-sacc-horizontal-10msbin
   ```

4. **Run flow analysis** (per-session):
   Use batch jobs in `jobs/` directory or `cli/run_one_session_evoked.py`

5. **Summarize across sessions**:
   ```bash
   python cli/summarize_flow_across_sessions.py --align stim --tags evoked_peakbin_stimC_vertical_lag80ms-none-trial
   python cli/summarize_flow_across_sessions.py --align sacc --tags evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial
   ```

6. **Generate trial onset figures**:
   ```bash
   python cli/trial_onset_comprehensive.py --align stim --axes_tag_stim axes_peakbin_stimCR-stim-vertical --orientation_stim vertical --tag trialonset_axes_peakbin_stimCR
   python cli/trial_onset_comprehensive.py --align sacc --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-10msbin --orientation_sacc horizontal --tag trialonset_axes_peakbin_saccS_10msbin
   ```

7. **Generate QC paper figures**:
   ```bash
   python cli/plot_qc_paper.py
   ```

---

## Complete Reproduction Command Sequence

```bash
# From paper_project_final/ directory

# Step 1: Build caches (if not already done)
python cli/build_caches_all.py --align stim --sid_list sid_list.txt
python cli/build_caches_all.py --align sacc --sid_list sid_list.txt

# Step 2: Run peakbin workflow (train axes → QC → flow) via SLURM
# For stim (lag=80ms) and sacc (lag=50ms)
sbatch jobs/peakbin_axes_qc_flow_array_big.sbatch

# For sacc with 10ms bins (rebinned)
sbatch jobs/peakbin_axes_qc_flow_array_10ms_big.sbatch

# Step 3: Generate Panel A & D (QC figures)
python cli/plot_qc_paper.py

# Step 4: Generate Panel B (Trial onset latency - Category)
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align stim \
    --axes_tag_stim axes_peakbin_stimCR-stim-vertical \
    --orientation_stim vertical \
    --tag trialonset_axes_peakbin_stimCR

# Step 5: Generate Panel E (Trial onset latency - Saccade, 10ms bins)
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align sacc \
    --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-10msbin \
    --orientation_sacc horizontal \
    --tag trialonset_axes_peakbin_saccS_10msbin

# Step 6: Generate Panel C (Directed predictability gain - Category)
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags evoked_peakbin_stimC_vertical_lag80ms-none-trial

# Step 7: Generate Panel F (Directed predictability gain - Saccade, 10ms bins)
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag50ms_10msbin-none-trial
```

---

## Data Format

### Flow NPZ Files
Located at: `out/<align>/<sid>/flow/<flow_tag>/<feature>/flow_<feature>_<area1>to<area2>.npz`

Contents:
- `time`: Time bins in seconds
- `bits_AtoB`: ΔLL (bits) for A→B direction
- `null_mean_AtoB`, `null_std_AtoB`: Null distribution statistics
- `p_AtoB`: Per-time-bin p-values
- `null_samps_AtoB` (optional): Full null samples for group-level testing

### Summary NPZ Files
Located at: `out/<align>/summary/<flow_tag>/<feature>/summary_<A>_vs_<B>.npz`

Contents:
- `time`: Time bins in seconds
- `mean_bits_AtoB`, `se_bits_AtoB`: Mean ± SE across sessions
- `mean_diff_bits`, `se_diff_bits`: Net flow (A→B - B→A) mean ± SE
- `p_group_diff`, `sig_group_diff`: Group-level p-values and significance masks
- `mean_bits_ptd_corr_AtoB`, `se_bits_ptd_corr_AtoB`: df-corrected bits/trial/dim for Panel D
- `session_ids`: List of included sessions
- `meta_json`: Metadata (tag, align, feature, pair, N sessions, etc.)

### Trial Onset NPZ Files
Located at: `out/<align>/trialtiming/<tag>/monkey_<M|S>_<area1>_vs_<area2>_<feature>_summary.npz`

Example paths:
- Stim: `out/stim/trialtiming/trialonset_axes_peakbin_stimCR/monkey_M_FEF_vs_LIP_category_summary.npz`
- Sacc: `out/sacc/trialtiming/trialonset_axes_peakbin_saccS_10msbin/monkey_M_FEF_vs_LIP_saccade_summary.npz`

Contents:
- `t1_ms`, `t2_ms`: Onset times for each area (in ms)
- `dt_ms`: Difference (t2 - t1) per trial
- `sids`: Session IDs for each trial
- `meta`: Statistics (median_dt, mean_dt, p-values, etc.)

### QC JSON Files
Located at: `out/<align>/<sid>/qc/<axes_tag>/qc_axes_<area>.json`

Example paths:
- Stim: `out/stim/20201211/qc/axes_peakbin_stimCR-stim-vertical/qc_axes_MFEF.json`
- Sacc: `out/sacc/20201211/qc/axes_peakbin_saccS-sacc-horizontal-10msbin/qc_axes_MFEF.json`

Contents:
- `time`: Time bins in seconds
- `auc_C`: Category AUC over time
- `auc_S_inv`: Saccade AUC (invariant) over time
- `auc_S_raw`: Saccade AUC (raw) over time

---

## Notes

1. **Monkey naming convention**: 
   - Monkey M sessions start with "2020" (areas: MFEF, MLIP, MSC)
   - Monkey S sessions start with "2023" (areas: SFEF, SLIP, SSC)

2. **Orientation conventions**:
   - "vertical" = up/down saccade targets (used for stim/category)
   - "horizontal" = left/right saccade targets (used for sacc/saccade)
   - "pooled" = both orientations combined

3. **Bin size conventions**:
   - Stim-aligned: 10 ms bins (original)
   - Sacc-aligned: 10 ms bins (rebinned from 5ms using `--rebin_factor 2`)

4. **QC filtering**: Sessions where either area fails QC (AUC < 0.6) are excluded symmetrically from both directions to ensure fair comparison.

5. **Group-level significance**: Uses permutation-based null distribution by sampling from per-session null distributions (4096 replicates by default).

6. **Flow tags naming convention**:
   - `evoked_peakbin_<feature>_<orientation>_lag<X>ms[-10msbin]-none-trial`
   - The `-10msbin` suffix indicates data was rebinned to 10ms bins

---

## File Modification History

- Created: This documentation was generated to record how to reproduce the six paper figures.
- Updated: Reorganized to reflect 6-panel structure with Monkey M/S subplots per panel.
- Scripts last modified: See individual script headers for version information.
