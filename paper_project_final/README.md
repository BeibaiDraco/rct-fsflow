# Paper Figures Documentation

This README documents the figures used in the paper and how to reproduce them using the **20ms sliding window workflow**.

---

## Overview of Figures

| Figure | File Path | Description |
|--------|-----------|-------------|
| 1 | `out/stim/summary/evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial/C/figs/MFEF_vs_MLIP_panel_c.png` | Net Flow: Category information flow (FEF→LIP vs LIP→FEF), stim-aligned |
| 2 | `out/sacc/summary/evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial/S/figs/MFEF_vs_MLIP_panel_c.png` | Net Flow: Saccade information flow (FEF→LIP vs LIP→FEF), sacc-aligned |
| 3 | `out/stim/trialtiming/trialonset_comprehensive_20mssw/monkey_M_FEF_vs_LIP_category_summary.png` | Trial onset latency scatter: Category encoding FEF vs LIP |
| 4 | `out/sacc/trialtiming/trialonset_comprehensive_20mssw/monkey_M_FEF_vs_LIP_saccade_summary.png` | Trial onset latency scatter: Saccade encoding FEF vs LIP |
| 5 | `out/paper_figures/qc/qc_stim_category_M_20mssw_summary.png` | QC: Category AUC over time (FEF, LIP, SC), stim-aligned, Monkey M |
| 6 | `out/paper_figures/qc/qc_sacc_M_20mssw_summary.png` | QC: Saccade AUC over time (FEF, LIP, SC), sacc-aligned, Monkey M |

---

## Current Workflow: 20ms Sliding Window

The paper figures are generated using the **20ms sliding window** approach:

- **Window**: 20ms integration window
- **Step**: 10ms output bin spacing (50% overlap)
- **Benefit**: Maintains 10ms temporal resolution while smoothing over 20ms

### Key Parameters (20ms Sliding Window)

| Parameter | Stim (Category) | Sacc (Saccade) |
|-----------|-----------------|----------------|
| alignment | stim | sacc |
| orientation | vertical | horizontal |
| window_ms | 20 | 20 |
| step_ms | 10 | 10 |
| **lag_ms** | **80** | **50** |
| evoked_sigma_ms | 10 | 10 |
| **QC threshold** | **0.65** | **0.65** |
| smooth_ms (summary) | 20 | 20 |

### Tag Naming Convention

- Axes: `axes_peakbin_stimCR-stim-vertical-20mssw`, `axes_peakbin_saccS-sacc-horizontal-20mssw`
- Flow: `evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial`, `evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial`

---

## Figure 1 & 2: Net Flow Panel C Figures

### Description
These figures show the **Net Flow** (ΔΔLL in bits) between brain areas:
- **Panel C** displays the difference in information flow: `(A→B) - (B→A)`
- Positive values indicate information flows from area A to area B
- Significance dots (black) mark time points where p < α (group-level permutation test)

### Source Script
`cli/summarize_flow_across_sessions.py`

### Input Data Location
- Stim-aligned: `out/stim/<session_id>/flow/evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial/C/flow_C_<area1>to<area2>.npz`
- Sacc-aligned: `out/sacc/<session_id>/flow/evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial/S/flow_S_<area1>to<area2>.npz`

### Output Location
- Stim category: `out/stim/summary/evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial/C/figs/MFEF_vs_MLIP_panel_c.{pdf,png,svg}`
- Sacc saccade: `out/sacc/summary/evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial/S/figs/MFEF_vs_MLIP_panel_c.{pdf,png,svg}`

### How to Reproduce

**Step 1: Run per-session flow analysis (via SLURM)**

```bash
# Submit array job for all 23 sessions
sbatch jobs/peakbin_axes_qc_flow_array_20ms_sliding.sbatch
```

This runs for each session:
1. Train axes with 20ms sliding window
2. QC validation
3. Flow computation for C, R (stim) and S (sacc) features

**Step 2: Summarize across sessions**

```bash
# Stim (category) - lag 80ms
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial \
    --qc_threshold 0.65 \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw \
    --smooth_ms 20 \
    --group_diff_p

# Sacc (saccade) - lag 50ms
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial \
    --qc_threshold 0.65 \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw \
    --smooth_ms 20 \
    --group_diff_p
```

### Figure Specifications
- **Plot area**: 10 × 5 inches
- **Figure size**: 11.5 × 6.3 inches (with margins)
- **Font sizes**: xlabel/ylabel 18-20pt, tick labels 18pt, legend 20pt
- **Colors**: darkcyan for Net Flow line, black for significance dots
- **Reference lines**: dashed vertical at t=0, dotted horizontal at y=0
- **Time range**: -100 to 500 ms (stim), -300 to 200 ms (sacc)
- **Y-axis limits**: Monkey M: -10 to +20 bits; Monkey S: -8 to +13 bits

---

## Figure 3 & 4: Trial Onset Latency Scatter Plots

### Description
These scatter plots show per-trial latency of encoding onset:
- **X-axis**: Latency in area 1 (e.g., FEF)
- **Y-axis**: Latency in area 2 (e.g., LIP)
- Points above the diagonal indicate area 2 encodes later than area 1
- Red circle: mean latency; Blue square: median latency

### Source Script
`cli/trial_onset_comprehensive.py`

### Input Data Location
- Axes: `out/<align>/<session_id>/axes/axes_peakbin_stimCR-stim-vertical-20mssw/axes_<area>.npz`
- Neural caches: `out/<align>/<session_id>/caches/area_<area>.npz`

### Output Location
- Category: `out/stim/trialtiming/trialonset_comprehensive_20mssw/monkey_M_FEF_vs_LIP_category_summary.{png,pdf,svg,npz}`
- Saccade: `out/sacc/trialtiming/trialonset_comprehensive_20mssw/monkey_M_FEF_vs_LIP_saccade_summary.{png,pdf,svg,npz}`

### How to Reproduce

```bash
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align stim sacc \
    --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw \
    --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw \
    --orientation_stim vertical \
    --orientation_sacc horizontal \
    --sliding_window_ms_stim 20 \
    --sliding_step_ms_stim 10 \
    --sliding_window_ms_sacc 20 \
    --sliding_step_ms_sacc 10 \
    --qc_threshold 0.65 \
    --k_sigma 4 \
    --runlen 5 \
    --smooth_ms 20.0 \
    --tag trialonset_comprehensive_20mssw
```

### Key Parameters

| Parameter | Category (stim) | Saccade (sacc) |
|-----------|-----------------|----------------|
| alignment | stim | sacc |
| orientation | vertical | horizontal |
| axes_tag | axes_peakbin_stimCR-stim-vertical-20mssw | axes_peakbin_saccS-sacc-horizontal-20mssw |
| feature | C | S |
| baseline window | -0.20 to 0.00 s | -0.35 to -0.20 s |
| search window | 0.00 to 0.50 s | -0.30 to 0.20 s |
| threshold | baseline_mean + 4σ | baseline_mean + 4σ |
| runlen | 5 consecutive bins | 5 consecutive bins |
| smoothing | 20 ms Gaussian | 20 ms Gaussian |
| **QC threshold** | **0.65** | **0.65** |

### Onset Detection Algorithm
1. Project neural activity onto encoding axis (sC for category, sS_inv for saccade)
2. Sign activity by trial label (e.g., multiply by ±1 based on category)
3. Smooth with Gaussian kernel (σ = 20 ms)
4. Calculate threshold: `baseline_mean + k_sigma * baseline_std` per trial
5. Find first time point in search window where activity exceeds threshold for `runlen` consecutive bins

### Figure Specifications
- **Plot area**: 5 × 5 inches (square)
- **Figure size**: 6.5 × 6.5 inches (with margins)
- **Axis limits**: Based on search window (stim: 0-500 ms, sacc: -300 to 200 ms)
- **Font sizes**: xlabel/ylabel 18pt, tick labels 16pt, legend 15pt
- **Markers**: scatter dots 7pt, mean (red circle) 12pt, median (blue square) 12pt

### Statistics Reported
- **Median dt**: Median of (area2_latency - area1_latency) across trials
- **Mean dt**: Mean of (area2_latency - area1_latency) across trials
- **p (two-sided)**: Sign-flip permutation test (20,000 permutations)
- **p (area2 later)**: One-sided test for area2 encoding later

---

## Figure 5 & 6: QC AUC Figures

### Description
These figures show the Area Under the Curve (AUC) for decoding accuracy over time:
- **Category AUC**: How well category (left vs right) can be decoded from neural activity
- **Saccade AUC**: How well saccade direction can be decoded from neural activity
- Higher AUC indicates stronger encoding of the feature
- Chance level is 0.5 (dashed horizontal line)

### Source Script
`cli/plot_qc_paper.py`

### Input Data Location
- Stim category QC: `out/stim/<session_id>/qc/axes_peakbin_stimCR-stim-vertical-20mssw/qc_axes_<area>.json`
- Sacc saccade QC: `out/sacc/<session_id>/qc/axes_peakbin_saccS-sacc-horizontal-20mssw/qc_axes_<area>.json`

### Output Location
- Category AUC (Monkey M): `out/paper_figures/qc/qc_stim_category_M_20mssw_summary.{pdf,png,svg}`
- Category AUC (Monkey S): `out/paper_figures/qc/qc_stim_category_S_20mssw_summary.{pdf,png,svg}`
- Saccade AUC (Monkey M): `out/paper_figures/qc/qc_sacc_M_20mssw_summary.{pdf,png,svg}`
- Saccade AUC (Monkey S): `out/paper_figures/qc/qc_sacc_S_20mssw_summary.{pdf,png,svg}`

### How to Reproduce

```bash
python cli/plot_qc_paper.py \
    --out_root out \
    --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw \
    --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw \
    --suffix _20mssw \
    --smooth_ms 20
```

### Key Parameters

| Parameter | Category (stim) | Saccade (sacc) |
|-----------|-----------------|----------------|
| qc_tag | axes_peakbin_stimCR-stim-vertical-20mssw | axes_peakbin_saccS-sacc-horizontal-20mssw |
| metric | auc_C | auc_S_inv |
| time range | -100 to 500 ms | -300 to 200 ms |
| y-axis limits | 0.35 to 1.0 | 0.35 to 1.0 |
| chance level | 0.5 | 0.5 |
| smooth_ms | 20 | 20 |

### Area Colors
| Area | Color | Hex Code |
|------|-------|----------|
| FEF | Blue | #0e87cc |
| LIP | Red | #f10c45 |
| SC | Purple | #9b5fc0 |

### Figure Specifications
- **Plot area**: 10 × 5 inches
- **Figure size**: 11.5 × 6.3 inches (with margins)
- **Font sizes**: xlabel 18pt, ylabel 20pt, tick labels 18pt, legend 20pt
- **Line width**: 3pt

---

## Prerequisite: Session Data

### Session List
All sessions are listed in `sid_list.txt` (23 sessions total):

**Monkey M (2020)**: 20200327, 20200328, 20200401, 20200402, 20200926, 20200929, 20201001, 20201004, 20201204, 20201211, 20201216

**Monkey S (2023)**: 20230622, 20230627, 20230705, 20230707, 20230710, 20231025, 20231103, 20231109, 20231121, 20231123, 20231130, 20231205

### Required Preprocessing Steps

1. **Build caches** (neural data preprocessing):
   ```bash
   python cli/build_caches_all.py --align stim --sid_list sid_list.txt
   python cli/build_caches_all.py --align sacc --sid_list sid_list.txt
   ```

2. **Run 20ms sliding window workflow** (axes + QC + flow):
   ```bash
   sbatch jobs/peakbin_axes_qc_flow_array_20ms_sliding.sbatch
   ```

3. **Summarize across sessions**:
   ```bash
   # Stim (category) - lag 80ms
   python cli/summarize_flow_across_sessions.py \
       --out_root out --align stim \
       --tags evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial \
       --qc_threshold 0.65 \
       --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw \
       --smooth_ms 20 --group_diff_p

   # Sacc (saccade) - lag 50ms
   python cli/summarize_flow_across_sessions.py \
       --out_root out --align sacc \
       --tags evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial \
       --qc_threshold 0.65 \
       --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw \
       --smooth_ms 20 --group_diff_p
   ```

4. **Generate trial onset figures**:
   ```bash
   python cli/trial_onset_comprehensive.py \
       --out_root out --sid_list sid_list.txt \
       --align stim sacc \
       --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw \
       --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw \
       --sliding_window_ms_stim 20 --sliding_step_ms_stim 10 \
       --sliding_window_ms_sacc 20 --sliding_step_ms_sacc 10 \
       --qc_threshold 0.65 \
       --tag trialonset_comprehensive_20mssw
   ```

5. **Generate QC paper figures**:
   ```bash
   python cli/plot_qc_paper.py \
       --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw \
       --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw \
       --suffix _20mssw --smooth_ms 20
   ```

---

## Complete Reproduction Command Sequence

```bash
# From paper_project_final/ directory

# Step 1: Build caches (if not already done)
python cli/build_caches_all.py --align stim --sid_list sid_list.txt
python cli/build_caches_all.py --align sacc --sid_list sid_list.txt

# Step 2: Run 20ms sliding window workflow for all sessions
sbatch jobs/peakbin_axes_qc_flow_array_20ms_sliding.sbatch

# Step 3: Summarize flows across sessions
# Stim (lag 80ms)
python cli/summarize_flow_across_sessions.py \
    --out_root out --align stim \
    --tags evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial \
    --qc_threshold 0.65 \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw \
    --smooth_ms 20 --group_diff_p

# Sacc (lag 50ms)
python cli/summarize_flow_across_sessions.py \
    --out_root out --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial \
    --qc_threshold 0.65 \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw \
    --smooth_ms 20 --group_diff_p

# Step 4: Generate trial onset latency figures
python cli/trial_onset_comprehensive.py \
    --out_root out --sid_list sid_list.txt \
    --align stim sacc \
    --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw \
    --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw \
    --sliding_window_ms_stim 20 --sliding_step_ms_stim 10 \
    --sliding_window_ms_sacc 20 --sliding_step_ms_sacc 10 \
    --qc_threshold 0.65 \
    --tag trialonset_comprehensive_20mssw

# Step 5: Generate QC paper figures
python cli/plot_qc_paper.py \
    --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw \
    --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw \
    --suffix _20mssw --smooth_ms 20
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
- `session_ids`: List of included sessions
- `meta_json`: Metadata (tag, align, feature, pair, N sessions, etc.)

### Trial Onset NPZ Files
Located at: `out/<align>/trialtiming/<tag>/monkey_<M>_<area1>_vs_<area2>_<feature>_summary.npz`

Contents:
- `t1_ms`, `t2_ms`: Onset times for each area (in ms)
- `dt_ms`: Difference (t2 - t1) per trial
- `sids`: Session IDs for each trial
- `meta`: Statistics (median_dt, mean_dt, p-values, etc.)

### QC JSON Files
Located at: `out/<align>/<sid>/qc/<axes_tag>/qc_axes_<area>.json`

Contents:
- `time`: Time bins in seconds
- `auc_C`: Category AUC over time
- `auc_S_inv`: Saccade AUC (invariant) over time
- `auc_S_raw`: Saccade AUC (raw) over time

---

## Workflow Comparison

| Aspect | Original 10ms | 20ms Rebin | **20ms Sliding (Current)** |
|--------|---------------|------------|----------------------------|
| Output bin spacing | 10ms | 20ms | **10ms** |
| STIM lag | 80ms | 80ms | **80ms** |
| SACC lag | 50ms | 60ms | **50ms** |
| QC threshold | 0.60 | 0.60 | **0.65** |
| smooth_ms | 30ms | 40ms | **20ms** |
| Tag suffix | (none) | -20msbin | **-20mssw** |

---

## Notes

1. **Monkey naming convention**: 
   - Monkey M sessions start with "2020" (areas: MFEF, MLIP, MSC)
   - Monkey S sessions start with "2023" (areas: SFEF, SLIP, SSC)

2. **Orientation conventions**:
   - "vertical" = up/down saccade targets (used for stim-aligned category)
   - "horizontal" = left/right saccade targets (used for sacc-aligned saccade)
   - "pooled" = both orientations combined

3. **QC filtering**: Sessions where either area fails QC (AUC < 0.65) are excluded symmetrically from both directions to ensure fair comparison.

4. **Group-level significance**: Uses permutation-based null distribution by sampling from per-session null distributions (4096 replicates by default).

5. **20ms sliding window advantage**: Maintains 10ms output resolution (same as original workflow) while integrating over 20ms windows for smoother estimates.

---

## File Modification History

- Created: Dec 18, 2025
- Updated: Jan 16, 2026 — Updated to 20ms sliding window workflow with lag 80ms (stim), 50ms (sacc), and QC threshold 0.65
