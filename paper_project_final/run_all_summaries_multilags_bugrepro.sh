#!/bin/bash
# =============================================================================
# Run all downstream analysis for BUG REPRODUCTION results
# (replicates old bug: all trials with lab_is_correct=True)
# =============================================================================

set -e

# Set environment variables (don't source homepath.txt due to conda issues)
export PAPER_HOME="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final"
export PAPER_DATA="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final/RCT_02"
export PYTHONPATH="$PAPER_HOME:$PYTHONPATH"

cd "$PAPER_HOME"

echo "============================================================"
echo "BUG REPRODUCTION SUMMARIES"
echo "This replicates the OLD buggy behavior:"
echo "  - All trials kept (correct + incorrect)"
echo "  - lab_is_correct = True for ALL trials"
echo "  - Downstream uses ALL trials"
echo "Results in: out_bugrepro/"
echo "============================================================"

echo ""
echo "============================================================"
echo "Step 1: Summarize flow across sessions for ALL STIM lags (bugrepro)"
echo "============================================================"

# STIM C lags: 30, 40, 50, 60, 80, 100ms
for LAG in 30 40 50 60 80 100; do
  TAG="evoked_peakbin_stimC_vertical_lag${LAG}ms_20mssw_bugrepro-none-trial"
  echo ""
  echo "[STIM] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out_bugrepro \
    --align stim \
    --tags "$TAG" \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw_bugrepro \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 2: Summarize flow across sessions for ALL SACC lags (bugrepro)"
echo "============================================================"

# SACC S lags: 30, 50, 60, 80ms
for LAG in 30 50 60 80; do
  TAG="evoked_peakbin_saccS_horizontal_lag${LAG}ms_20mssw_bugrepro-none-trial"
  echo ""
  echo "[SACC] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out_bugrepro \
    --align sacc \
    --tags "$TAG" \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw_bugrepro \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 3: Plot QC paper figures (bugrepro)"
echo "============================================================"

python cli/plot_qc_paper.py \
  --out_root out_bugrepro \
  --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw_bugrepro \
  --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw_bugrepro \
  --suffix _20mssw_bugrepro \
  --smooth_ms 30 \
  --qc_threshold 0.65

echo ""
echo "============================================================"
echo "Step 4: Trial onset comprehensive analysis (STIM, bugrepro)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out_bugrepro \
  --align stim \
  --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw_bugrepro \
  --orientation_stim vertical \
  --pt_min_ms_stim 200 \
  --sliding_window_ms_stim 20 \
  --sliding_step_ms_stim 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw_bugrepro

echo ""
echo "============================================================"
echo "Step 5: Trial onset comprehensive analysis (SACC, bugrepro)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out_bugrepro \
  --align sacc \
  --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw_bugrepro \
  --orientation_sacc horizontal \
  --pt_min_ms_sacc 200 \
  --sliding_window_ms_sacc 20 \
  --sliding_step_ms_sacc 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw_bugrepro

echo ""
echo "============================================================"
echo "[DONE] BUG REPRODUCTION summaries complete!"
echo "============================================================"
echo ""
echo "Output locations (all in out_bugrepro/):"
echo "  - Flow summaries: out_bugrepro/stim/summary/evoked_peakbin_stimC_vertical_lag*_20mssw_bugrepro-none-trial/"
echo "  - Flow summaries: out_bugrepro/sacc/summary/evoked_peakbin_saccS_horizontal_lag*_20mssw_bugrepro-none-trial/"
echo "  - QC figures: out_bugrepro/paper_figures/qc/*_bugrepro*"
echo "  - Trial onset: out_bugrepro/stim/trialtiming/trialonset_comprehensive_20mssw_bugrepro/"
echo "  - Trial onset: out_bugrepro/sacc/trialtiming/trialonset_comprehensive_20mssw_bugrepro/"
echo ""
echo "Compare with your OLD results to verify they match!"
