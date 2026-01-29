#!/bin/bash
#SBATCH --job-name=summaries_mixedtrain
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/summaries_mixedtrain_%j.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/summaries_mixedtrain_%j.err

# =============================================================================
# Run all downstream analysis for MIXED TRAINING results
# (axes trained on unfiltered data, QC/flow on correct-only data)
# Cluster version of run_all_summaries_multilags_mixedtrain.sh
# =============================================================================

set -euo pipefail

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# Thread hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# paper_project_final canonical paths
export PAPER_HOME="$PROJECT"
export PAPER_DATA="/project/bdoiron/dracoxu/rct-fsflow/saccade/RCT_02"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

echo "============================================================"
echo "MIXED TRAINING SUMMARIES"
echo "Axes trained on: unfiltered (all trials)"
echo "QC/Flow run on:  correct-only"
echo "Results in:      out/"
echo "============================================================"

echo ""
echo "============================================================"
echo "Step 1: Trial onset comprehensive analysis (STIM, mixedtrain)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out \
  --align stim \
  --axes_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain \
  --orientation_stim vertical \
  --pt_min_ms_stim 200 \
  --sliding_window_ms_stim 20 \
  --sliding_step_ms_stim 10 \
  --qc_root out_nofilter \
  --qc_tag_stim axes_peakbin_stimCR-stim-vertical-20mssw_nofilter \
  --qc_threshold 0.65 \
  --smooth_ms 20 \
  --tag trialonset_comprehensive_20mssw_mixedtrain

echo ""
echo "============================================================"
echo "Step 2: Trial onset comprehensive analysis (SACC, mixedtrain)"
echo "============================================================"

python cli/trial_onset_comprehensive.py \
  --out_root out \
  --align sacc \
  --axes_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw_mixedtrain \
  --orientation_sacc horizontal \
  --pt_min_ms_sacc 200 \
  --sliding_window_ms_sacc 20 \
  --sliding_step_ms_sacc 10 \
  --qc_root out_nofilter \
  --qc_tag_sacc axes_peakbin_saccS-sacc-horizontal-20mssw_nofilter \
  --qc_threshold 0.65 \
  --smooth_ms 20 \
  --tag trialonset_comprehensive_20mssw_mixedtrain

echo ""
echo "============================================================"
echo "Step 3: Summarize flow across sessions for ALL STIM lags (mixedtrain)"
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
    --qc_root out_nofilter \
    --qc_tag axes_peakbin_stimCR-stim-vertical-20mssw_nofilter \
    --qc_threshold 0.65 \
    --smooth_ms 20 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 4: Summarize flow across sessions for ALL SACC lags (mixedtrain)"
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
    --qc_root out_nofilter \
    --qc_tag axes_peakbin_saccS-sacc-horizontal-20mssw_nofilter \
    --qc_threshold 0.65 \
    --smooth_ms 20 \
    --alpha 0.05 \
    --group_diff_p
done

echo ""
echo "============================================================"
echo "Step 5: Plot QC paper figures (mixedtrain)"
echo "============================================================"

python cli/plot_qc_paper.py \
  --out_root out \
  --stim_qc_subdir axes_peakbin_stimCR-stim-vertical-20mssw_mixedtrain \
  --sacc_qc_subdir axes_peakbin_saccS-sacc-horizontal-20mssw_mixedtrain \
  --suffix _20mssw_mixedtrain \
  --smooth_ms 20 \
  --qc_threshold 0.65

echo ""
echo "============================================================"
echo "[DONE] MIXED TRAINING summaries complete!"
echo "============================================================"
