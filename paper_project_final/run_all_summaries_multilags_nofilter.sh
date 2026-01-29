#!/bin/bash
# =============================================================================
# Run all downstream analysis for NOFILTER results
# (axes trained AND QC/flow on unfiltered data - all trials including incorrect)
# =============================================================================

set -e

# Set environment variables (don't source homepath.txt due to conda issues)
export PAPER_HOME="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final"
export PAPER_DATA="/Users/dracoxu/Documents/Research/rct-fsflow/paper_project_final/RCT_02"
export PYTHONPATH="$PAPER_HOME:$PYTHONPATH"

cd "$PAPER_HOME"

echo "============================================================"
echo "NOFILTER SUMMARIES"
echo "Axes trained on: unfiltered (all trials)"
echo "QC/Flow run on:  unfiltered (all trials)"
echo "Results in:      out_nofilter/"
echo "============================================================"

echo ""
echo "============================================================"
echo "Step 1: Summarize flow across sessions for ALL STIM lags (nofilter)"
echo "============================================================"

# STIM C lags: 30, 40, 50, 60, 80, 100ms
for LAG in 30 40 50 60 80 100; do
  TAG="evoked_peakbin_stimC_vertical_lag${LAG}ms_20mssw_nofilter-none-trial"
  echo ""
  echo "[STIM] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out_nofilter \
    --align stim \
    --tags "$TAG" \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw_nofilter \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 2: Summarize flow across sessions for ALL SACC lags (nofilter)"
echo "============================================================"

# SACC S lags: 30, 50, 60, 80ms
for LAG in 30 50 60 80; do
  TAG="evoked_peakbin_saccS_horizontal_lag${LAG}ms_20mssw_nofilter-none-trial"
  echo ""
  echo "[SACC] Summarizing tag: $TAG"
  python cli/summarize_flow_across_sessions.py \
    --out_root out_nofilter \
    --align sacc \
    --tags "$TAG" \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw_nofilter \
    --qc_threshold 0.65 \
    --smooth_ms 30 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 3: Plot QC paper figures (nofilter)"
echo "============================================================"

python cli/plot_qc_paper.py \
  --out_root out_nofilter \
  --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw_nofilter \
  --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw_nofilter \
  --suffix _20mssw_nofilter \
  --smooth_ms 30 \
  --qc_threshold 0.65

echo ""
echo "============================================================"
echo "Step 4: Trial onset comprehensive analysis (STIM, nofilter)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out_nofilter \
  --align stim \
  --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw_nofilter \
  --orientation_stim vertical \
  --pt_min_ms_stim 200 \
  --sliding_window_ms_stim 20 \
  --sliding_step_ms_stim 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw_nofilter

echo ""
echo "============================================================"
echo "Step 5: Trial onset comprehensive analysis (SACC, nofilter)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out_nofilter \
  --align sacc \
  --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw_nofilter \
  --orientation_sacc horizontal \
  --pt_min_ms_sacc 200 \
  --sliding_window_ms_sacc 20 \
  --sliding_step_ms_sacc 10 \
  --qc_threshold 0.65 \
  --smooth_ms 30 \
  --tag trialonset_comprehensive_20mssw_nofilter

echo ""
echo "============================================================"
echo "[DONE] NOFILTER summaries complete!"
echo "============================================================"
echo ""
echo "Output locations (all in out_nofilter/):"
echo "  - Flow summaries: out_nofilter/stim/summary/evoked_peakbin_stimC_vertical_lag*_20mssw_nofilter-none-trial/"
echo "  - Flow summaries: out_nofilter/sacc/summary/evoked_peakbin_saccS_horizontal_lag*_20mssw_nofilter-none-trial/"
echo "  - QC figures: out_nofilter/paper_figures/qc/*_nofilter*"
echo "  - Trial onset: out_nofilter/stim/trialtiming/trialonset_comprehensive_20mssw_nofilter/"
echo "  - Trial onset: out_nofilter/sacc/trialtiming/trialonset_comprehensive_20mssw_nofilter/"
echo ""
echo "Compare with original to assess impact of including incorrect trials in training AND evaluation."
