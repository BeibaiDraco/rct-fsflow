#!/bin/bash
# =============================================================================
# Run all downstream analysis after multi-lag flow jobs complete
# =============================================================================

set -e

# Set environment variables (don't source homepath.txt due to conda issues)
export PAPER_HOME="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final"
export PAPER_DATA="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final/RCT_02"
export PYTHONPATH="$PAPER_HOME:$PYTHONPATH"

cd "$PAPER_HOME"

echo "============================================================"
echo "Step 1: Summarize flow across sessions for ALL STIM lags"
echo "============================================================"

# STIM C lags: 30, 40, 50, 60, 80, 100ms
for LAG in 30 40 50 60 80 100; do
  TAG="evoked_peakbin_stimC_vertical_lag${LAG}ms_20mssw-none-trial"
  echo ""
  echo "[STIM] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align stim \
    --tags "$TAG" \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 2: Summarize flow across sessions for ALL SACC lags"
echo "============================================================"

# SACC S lags: 30, 50, 60, 80ms
for LAG in 30 50 60 80; do
  TAG="evoked_peakbin_saccS_horizontal_lag${LAG}ms_20mssw-none-trial"
  echo ""
  echo "[SACC] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out \
    --align sacc \
    --tags "$TAG" \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 3: Plot QC paper figures"
echo "============================================================"

python cli/plot_qc_paper.py \
  --out_root out \
  --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw \
  --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw \
  --suffix _20mssw \
  --smooth_ms 30 \
  --qc_threshold 0.65

echo ""
echo "============================================================"
echo "Step 4: Trial onset comprehensive analysis (STIM)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out \
  --align stim \
  --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw \
  --orientation_stim vertical \
  --pt_min_ms_stim 200 \
  --sliding_window_ms_stim 20 \
  --sliding_step_ms_stim 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw

echo ""
echo "============================================================"
echo "Step 5: Trial onset comprehensive analysis (SACC)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out \
  --align sacc \
  --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw \
  --orientation_sacc horizontal \
  --pt_min_ms_sacc 200 \
  --sliding_window_ms_sacc 20 \
  --sliding_step_ms_sacc 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw

echo ""
echo "============================================================"
echo "[DONE] All summaries, QC plots, and trial onset analysis complete!"
echo "============================================================"
echo ""
echo "Output locations:"
echo "  - Flow summaries: out/stim/summary/evoked_peakbin_stimC_vertical_lag*_20mssw-none-trial/"
echo "  - Flow summaries: out/sacc/summary/evoked_peakbin_saccS_horizontal_lag*_20mssw-none-trial/"
echo "  - QC figures: out/paper_figures/qc/"
echo "  - Trial onset: out/stim/trialtiming/trialonset_comprehensive_20mssw/"
echo "  - Trial onset: out/sacc/trialtiming/trialonset_comprehensive_20mssw/"
