# QC Summary (Threshold = 0.65)

QC tags used:
- **Stim**: `axes_peakbin_stimCR-stim-vertical-20mssw` (metric: auc_C)
- **Sacc**: `axes_peakbin_saccS-sacc-horizontal-20mssw` (metric: auc_S_inv)

---

## 1. Flow Pair Analysis (`summarize_flow_across_sessions.py`)

Sessions where **both areas** pass the 0.65 QC threshold:

| Pair | Stim (C) | Sacc (S) |
|------|----------|----------|
| MFEF vs MLIP | 11 | 11 |
| MFEF vs MSC | 6 | 7 |
| MLIP vs MSC | 6 | 7 |
| SFEF vs SLIP | 12 | 11 |
| SFEF vs SSC | 7 | 6 |
| SLIP vs SSC | 7 | 6 |

**Totals:**
- Monkey M (FEF-LIP): 11 stim, 11 sacc
- Monkey M (with SC): 6 stim, 7 sacc
- Monkey S (FEF-LIP): 12 stim, 11 sacc
- Monkey S (with SC): 7 stim, 6 sacc

---

## 2. QC Paper Plots (`plot_qc_paper.py`)

Sessions with **at least one area** passing 0.65:

### Stim (auc_C)

| Monkey | Sessions Plotted | FEF | LIP | SC |
|--------|------------------|-----|-----|-----|
| M (2020) | 11/11 | 11 | 11 | 6 |
| S (2023) | 12/12 | 12 | 12 | 7 |
| **Total** | **23/23** | 23 | 23 | 13 |

### Sacc (auc_S_inv)

| Monkey | Sessions Plotted | FEF | LIP | SC |
|--------|------------------|-----|-----|-----|
| M (2020) | 11/11 | 11 | 11 | 7 |
| S (2023) | 11/12 | 11 | 11 | 6 |
| **Total** | **22/23** | 22 | 22 | 13 |

---

## QC Failures (peak AUC < 0.65)

| Condition | Session | Area | Peak AUC | Reason |
|-----------|---------|------|----------|--------|
| Stim (C) | 20200929 | MSC | 0.645 | Just below threshold |
| Sacc (S) | 20231130 | all | NULL | Only 18 horizontal saccade trials |

All other session-area combinations pass the 0.65 threshold.

---

## Notes

- **No QC Data**: Early sessions (2020 for M, early 2023 for S) lack SC recordings
- **FEF vs LIP pairs**: 11 sessions for M, 11-12 for S consistently pass
- **SC pairs**: Limited to 6-7 sessions due to missing data and QC failures

## Commands to Run with QC Filtering

**Note:** The script now requires `--qc_tag` when QC filtering is enabled. It will error and stop if no QC tag is provided and auto-detection fails.

```bash
# Stim summaries with QC filtering
python cli/summarize_flow_across_sessions.py --align stim \
    --tags evoked_peakbin_stimC_vertical_lag30ms_20mssw-none-trial \
           evoked_peakbin_stimC_vertical_lag40ms_20mssw-none-trial \
           evoked_peakbin_stimC_vertical_lag50ms_20mssw-none-trial \
           evoked_peakbin_stimC_vertical_lag80ms_20mssw-none-trial \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw \
    --features C

# Sacc summaries with QC filtering
python cli/summarize_flow_across_sessions.py --align sacc \
    --tags evoked_peakbin_saccS_horizontal_lag50ms_20mssw-none-trial \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw \
    --features S
```
