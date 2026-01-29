#!/bin/bash
#SBATCH --job-name=align_sum
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_sum_%j.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_sum_%j.err

# =============================================================================
# AXIS ALIGNMENT ANALYSIS - SUMMARY
#
# This script aggregates per-session alignment results and generates:
#   - Group-level statistics with sign-flip tests
#   - Summary JSON files
#   - Publication-ready figures
#
# THREE analyses summarized:
#   A) Alignment Index (PCA subspaces + covariance null)
#   B) Axis Shuffle (trained axes + label-shuffle null)
#   C) Axis Covariance (trained axes + covariance null) - DAVE'S QUESTION
#
# Prerequisites:
#   - Run jobs/axis_alignment_array.sbatch first (all tasks must complete)
#   - By default, axis_alignment_array.sbatch runs BOTH correctonly and nofilter
#
# Usage:
#   sbatch jobs/axis_alignment_summary.sbatch
#
# Or submit as dependency:
#   ARRAY_JOB=$(sbatch --parsable jobs/axis_alignment_array.sbatch)
#   sbatch --dependency=afterok:$ARRAY_JOB jobs/axis_alignment_summary.sbatch
# =============================================================================

set -euo pipefail

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
OUT_ROOT="$PROJECT/out"
OUT_NOFILTER="$PROJECT/out_nofilter"

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# Thread hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PAPER_HOME="$PROJECT"
export PAPER_DATA="$PROJECT/RCT_02"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

echo "=============================================================="
echo "AXIS ALIGNMENT ANALYSIS - SUMMARY"
echo "=============================================================="

# =============================================================================
# 1. Summarize CORRECT-ONLY results
# =============================================================================
TAG_CORRECT="correctonly"

if [[ -d "$OUT_ROOT/sacc/alignment/$TAG_CORRECT" ]]; then
  echo ""
  echo "[1] Summarizing CORRECT-ONLY results ($TAG_CORRECT)"
  echo ""
  
  python cli/summarize_alignment.py \
    --out_root "$OUT_ROOT" \
    --tag "$TAG_CORRECT" \
    --mode all
else
  echo "[WARN] No results found for tag=$TAG_CORRECT in $OUT_ROOT"
fi

# =============================================================================
# 2. Summarize NOFILTER results (if exists)
# =============================================================================
TAG_NOFILTER="nofilter"

if [[ -d "$OUT_NOFILTER/sacc/alignment/$TAG_NOFILTER" ]]; then
  echo ""
  echo "[2] Summarizing NOFILTER results ($TAG_NOFILTER)"
  echo ""
  
  python cli/summarize_alignment.py \
    --out_root "$OUT_NOFILTER" \
    --tag "$TAG_NOFILTER" \
    --mode all
else
  echo "[WARN] No results found for tag=$TAG_NOFILTER in $OUT_NOFILTER (optional)"
fi

echo ""
echo "=============================================================="
echo "[done] Summary complete"
echo "=============================================================="
echo ""
echo "Results saved to:"
echo "  $OUT_ROOT/sacc/summary/alignment_$TAG_CORRECT/"
if [[ -d "$OUT_NOFILTER/sacc/alignment/$TAG_NOFILTER" ]]; then
  echo "  $OUT_NOFILTER/sacc/summary/alignment_$TAG_NOFILTER/"
fi
echo ""
echo "Key outputs:"
echo "  - summary_alignment_index.json: AI analysis (PCA subspaces)"
echo "  - summary_axis_shuffle.json: Axis-shuffle analysis (label-specificity)"
echo "  - summary_axis_covariance.json: Axis-covariance analysis (DAVE'S QUESTION)"
echo "  - figs/: Publication-ready figures"
echo ""
echo "Interpretation guide:"
echo "  Analysis A (AI): group_p_less < 0.05 means PCA subspaces are MORE ORTHOGONAL than chance"
echo "  Analysis B (shuffle): group_p_greater < 0.05 means alignment is LABEL-SPECIFIC"
echo "  Analysis C (cov): group_p_orth < 0.05 means trained AXES are MORE ORTHOGONAL than chance"
echo "  *** Analysis C is the primary test for Dave's question about trained axes ***"
