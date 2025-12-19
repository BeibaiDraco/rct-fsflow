# Paper Figures Documentation

This README documents the six figures used in the paper and how to reproduce them.

---

## Overview of Figures

| Figure | File Path | Description |
|--------|-----------|-------------|
| 1 | `out/stim/summary/evoked_subtract_vert-stim-none-trial/C/figs/MFEF_vs_MLIP_panel_c.png` | Net Flow: Category information flow (FEF→LIP vs LIP→FEF), stim-aligned |
| 2 | `out/sacc/summary/evoked_subtract_horiz-sacc-none-trial/S/figs/MFEF_vs_MLIP_panel_c.png` | Net Flow: Saccade information flow (FEF→LIP vs LIP→FEF), sacc-aligned |
| 3 | `out/stim/trialtiming/trialonset_comprehensive/monkey_M_FEF_vs_LIP_category_summary.png` | Trial onset latency scatter: Category encoding FEF vs LIP |
| 4 | `out/sacc/trialtiming/trialonset_comprehensive/monkey_M_FEF_vs_LIP_saccade_summary.png` | Trial onset latency scatter: Saccade encoding FEF vs LIP |
| 5 | `out/paper_figures/qc/qc_stim_category.png` | QC: Category AUC over time (FEF, LIP, SC), stim-aligned |
| 6 | `out/paper_figures/qc/qc_sacc.png` | QC: Saccade AUC over time (FEF, LIP, SC), sacc-aligned |

---

## Figure 1 & 2: Net Flow Panel C Figures

### Description
These figures show the **Net Flow** (ΔΔLL in bits) between brain areas:
- **Panel C** displays the difference in information flow: `(A→B) - (B→A)`
- Positive values indicate information flows from area A to area B
- Significance dots (black) mark time points where p < α (group-level permutation test)

### Source Script
`cli/summarize_flow_across_sessions.py`

### Generating Function
`plot_panel_c_paper()` in the summarize script

### Input Data Location
The script reads flow results from:
- Stim-aligned: `out/stim/<session_id>/flow/evoked_subtract_vert-stim-none-trial/C/flow_C_<area1>to<area2>.npz`
- Sacc-aligned: `out/sacc/<session_id>/flow/evoked_subtract_horiz-sacc-none-trial/S/flow_S_<area1>to<area2>.npz`

### Output Location
- Stim category: `out/stim/summary/evoked_subtract_vert-stim-none-trial/C/figs/MFEF_vs_MLIP_panel_c.{pdf,png,svg}`
- Sacc saccade: `out/sacc/summary/evoked_subtract_horiz-sacc-none-trial/S/figs/MFEF_vs_MLIP_panel_c.{pdf,png,svg}`

### How to Reproduce

**Step 1: Run per-session flow analysis (if not already done)**

For all sessions, run the evoked subtraction flow analysis:

```bash
# Run for each session in sid_list.txt
# Stim-aligned (vertical orientation for category)
python cli/run_one_session_evoked.py \
    --sid <SESSION_ID> \
    --orientation vertical \
    --axes_orientation vertical \
    --stim_feature C \
    --sacc_feature none

# Sacc-aligned (horizontal orientation for saccade)
python cli/run_one_session_evoked.py \
    --sid <SESSION_ID> \
    --orientation horizontal \
    --axes_orientation horizontal \
    --stim_feature none \
    --sacc_feature S
```

Or use the batch job scripts in `jobs/`:
- `jobs/flow_evoked_array.sbatch` - for stim-aligned vertical evoked flows
- `jobs/flow_evoked_horiz_array.sbatch` - for sacc-aligned horizontal evoked flows

**Step 2: Summarize across sessions**

```bash
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align both \
    --tag_prefix evoked \
    --group_diff_p \
    --save_null_samples
```

### Key Parameters

| Parameter | Stim (Category) | Sacc (Saccade) |
|-----------|-----------------|----------------|
| alignment | stim | sacc |
| orientation | vertical | horizontal |
| axes_orientation | vertical | horizontal |
| feature | C (category) | S (saccade) |
| flow_tag | evoked_subtract_vert-stim-none-trial | evoked_subtract_horiz-sacc-none-trial |
| time range | -100 to 500 ms | -300 to 200 ms |
| y-axis limits | -10 to +20 bits | -10 to +20 bits |

### Figure Specifications
- **Plot area**: 10 × 5 inches
- **Figure size**: 11.5 × 6.3 inches (with margins)
- **Font sizes**: xlabel/ylabel 18-20pt, tick labels 18pt, legend 20pt
- **Colors**: darkcyan for Net Flow line, black for significance dots
- **Reference lines**: dashed vertical at t=0, dotted horizontal at y=0

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
- Axes: `out/<align>/<session_id>/axes/<axes_tag>/axes_<area>.npz`
- Neural caches: `out/<align>/<session_id>/caches/area_<area>.npz`

### Output Location
- Category: `out/stim/trialtiming/trialonset_comprehensive/monkey_M_FEF_vs_LIP_category_summary.{png,pdf,svg,npz}`
- Saccade: `out/sacc/trialtiming/trialonset_comprehensive/monkey_M_FEF_vs_LIP_saccade_summary.{png,pdf,svg,npz}`

### How to Reproduce

```bash
# Run trial onset comprehensive analysis
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align stim sacc \
    --orientation_stim vertical \
    --orientation_sacc horizontal \
    --axes_tag_stim axes_sweep-stim-vertical \
    --axes_tag_sacc axes_sweep-sacc-horizontal \
    --qc_threshold 0.75 \
    --k_sigma 6 \
    --runlen 5 \
    --smooth_ms 20.0
```

### Key Parameters

| Parameter | Category (stim) | Saccade (sacc) |
|-----------|-----------------|----------------|
| alignment | stim | sacc |
| orientation | vertical | horizontal |
| axes_tag | axes_sweep-stim-vertical | axes_sweep-sacc-horizontal |
| feature | C | S |
| baseline window | -0.20 to 0.00 s | -0.35 to -0.20 s |
| search window | 0.00 to 0.50 s | -0.30 to 0.20 s |
| threshold | baseline_mean + 6σ | baseline_mean + 6σ |
| runlen | 5 consecutive bins | 5 consecutive bins |
| smoothing | 20 ms Gaussian | 20 ms Gaussian |
| QC threshold | 0.75 | 0.75 |

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
- Stim category QC: `out/stim/<session_id>/qc/axes_sweep-vertical/qc_axes_<area>.json`
- Sacc saccade QC: `out/sacc/<session_id>/qc/axes_sweep-sacc-horizontal/qc_axes_<area>.json`

**Example session used**: `20201211` (hardcoded in the script)

### Output Location
- Category AUC: `out/paper_figures/qc/qc_stim_category.{pdf,png,svg}`
- Saccade AUC: `out/paper_figures/qc/qc_sacc.{pdf,png,svg}`

### How to Reproduce

```bash
python cli/plot_qc_paper.py
```

### Key Parameters

| Parameter | Category (stim) | Saccade (sacc) |
|-----------|-----------------|----------------|
| session | 20201211 | 20201211 |
| qc_tag | axes_sweep-vertical | axes_sweep-sacc-horizontal |
| metric | auc_C | auc_S_inv |
| time range | -100 to 500 ms | -300 to 200 ms |
| y-axis limits | 0.35 to 1.0 | 0.35 to 1.0 |
| chance level | 0.5 | 0.5 |

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
   python cli/qc_axes.py --align stim --axes_tag axes_sweep-stim-vertical
   python cli/qc_axes.py --align sacc --axes_tag axes_sweep-sacc-horizontal
   ```

4. **Run flow analysis** (per-session):
   Use batch jobs in `jobs/` directory or `cli/run_one_session_evoked.py`

5. **Summarize across sessions**:
   ```bash
   python cli/summarize_flow_across_sessions.py --align both --tag_prefix evoked --group_diff_p
   ```

6. **Generate trial onset figures**:
   ```bash
   python cli/trial_onset_comprehensive.py --align stim sacc
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

# Step 2: Train axes (if not already done)
# This is typically done via batch jobs or run_one_session_sweep.py

# Step 3: Run evoked flow for all sessions (typically via SLURM batch jobs)
# See jobs/flow_evoked_array.sbatch and jobs/flow_evoked_horiz_array.sbatch

# Step 4: Summarize flows across sessions
python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align both \
    --tag_prefix evoked \
    --group_diff_p \
    --alpha 0.05

# Step 5: Generate trial onset latency figures
python cli/trial_onset_comprehensive.py \
    --out_root out \
    --sid_list sid_list.txt \
    --align stim sacc \
    --tag trialonset_comprehensive

# Step 6: Generate QC paper figures
python cli/plot_qc_paper.py
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

## Notes

1. **Monkey naming convention**: 
   - Monkey M sessions start with "2020" (areas: MFEF, MLIP, MSC)
   - Monkey S sessions start with "2023" (areas: SFEF, SLIP, SSC)

2. **Orientation conventions**:
   - "vertical" = up/down saccade targets
   - "horizontal" = left/right saccade targets
   - "pooled" = both orientations combined

3. **QC filtering**: Sessions where either area fails QC (AUC < 0.75) are excluded symmetrically from both directions to ensure fair comparison.

4. **Group-level significance**: Uses permutation-based null distribution by sampling from per-session null distributions (4096 replicates by default).

---

## File Modification History

- Created: This documentation was generated to record how to reproduce the six paper figures.
- Scripts last modified: See individual script headers for version information.


