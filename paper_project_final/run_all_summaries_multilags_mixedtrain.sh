#!/bin/bash
# =============================================================================
# Run all downstream analysis for MIXED TRAINING results
# (axes trained on unfiltered data, QC/flow on correct-only data)
# =============================================================================

set -e

# Set environment variables (don't source homepath.txt due to conda issues)
export PAPER_HOME="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final"
export PAPER_DATA="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final/RCT_02"
export PYTHONPATH="$PAPER_HOME:$PYTHONPATH"

cd "$PAPER_HOME"

echo "============================================================"
echo "MIXED TRAINING SUMMARIES"
echo "Axes trained on: unfiltered (all trials)"
echo "QC/Flow run on:  correct-only"
echo "Results in:      out/"
echo "============================================================"

echo ""
echo "============================================================"
echo "Step 1: Summarize flow across sessions for ALL STIM lags (mixedtrain)"
echo "============================================================"

# STIM C lags: 30, 40, 50, 60, 80, 100ms
for LAG in 30 40 50 60 80 100; do
  TAG="evoked_peakbin_stimC_vertical_lag${LAG}ms_20mssw_mixedtrain-none-trial"
  echo ""
  echo "[STIM] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags "$TAG" \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 2: Summarize flow across sessions for ALL SACC lags (mixedtrain)"
echo "============================================================"

# SACC S lags: 30, 50, 60, 80ms
for LAG in 30 50 60 80; do
  TAG="evoked_peakbin_saccS_horizontal_lag${LAG}ms_20mssw_mixedtrain-none-trial"
  echo ""
  echo "[SACC] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags "$TAG" \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw_mixedtrain \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 3: Plot QC paper figures (mixedtrain)"
echo "============================================================"

python cli/plot_qc_paper.py \
  --out_root out \
  --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain \
  --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw_mixedtrain \
  --suffix _20mssw_mixedtrain \
  --smooth_ms 30 \
  --qc_threshold 0.65

echo ""
echo "============================================================"
echo "Step 4: Trial onset comprehensive analysis (STIM, mixedtrain)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out \
  --align stim \
  --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain \
  --orientation_stim vertical \
  --pt_min_ms_stim 200 \
  --sliding_window_ms_stim 20 \
  --sliding_step_ms_stim 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw_mixedtrain

echo ""
echo "============================================================"
echo "Step 5: Trial onset comprehensive analysis (SACC, mixedtrain)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out \
  --align sacc \
  --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw_mixedtrain \
  --orientation_sacc horizontal \
  --pt_min_ms_sacc 200 \
  --sliding_window_ms_sacc 20 \
  --sliding_step_ms_sacc 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw_mixedtrain

echo ""
echo "============================================================"
echo "[DONE] MIXED TRAINING summaries complete!"
echo "============================================================"
echo ""
echo "Output locations (all in out/):"
echo "  - Flow summaries: out/stim/summary/evoked_peakbin_stimC_vertical_lag*_20mssw_mixedtrain-none-trial/"
echo "  - Flow summaries: out/sacc/summary/evoked_peakbin_saccS_horizontal_lag*_20mssw_mixedtrain-none-trial/"
echo "  - QC figures: out/paper_figures/qc/*_mixedtrain*"
echo "  - Trial onset: out/stim/trialtiming/trialonset_comprehensive_20mssw_mixedtrain/"
echo "  - Trial onset: out/sacc/trialtiming/trialonset_comprehensive_20mssw_mixedtrain/"
echo ""
echo "Compare with original (no suffix) to assess impact of training on contaminated data."
