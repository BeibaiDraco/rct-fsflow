#!/bin/bash
# =============================================================================
# DEPRECATED - Use the new parallelized workflow instead:
#
#   # Step 1: Run per-session analysis (job array)
#   sbatch jobs/axis_alignment_array.sbatch        # correct-only
#   sbatch jobs/axis_alignment_array_nofilter.sbatch  # all trials
#
#   # Step 2: After all sessions complete, run summary
#   sbatch jobs/axis_alignment_summary.sbatch
#
# Or use dependency:
#   ARRAY_JOB=$(sbatch --parsable jobs/axis_alignment_array.sbatch)
#   sbatch --dependency=afterok:$ARRAY_JOB jobs/axis_alignment_summary.sbatch
#
# =============================================================================
# This legacy script runs all sessions sequentially (SLOW).
# Use only for debugging or testing on a few sessions.
# =============================================================================

#SBATCH --job-name=axis_align_legacy
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/axis_align_legacy_%j.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/axis_align_legacy_%j.err

set -euo pipefail

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
OUT_ROOT="$PROJECT/out"
SID_LIST="$PROJECT/sid_list.txt"

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
echo "WARNING: Using legacy sequential workflow"
echo "Consider using the parallelized workflow instead:"
echo "  sbatch jobs/axis_alignment_array.sbatch"
echo "  sbatch jobs/axis_alignment_summary.sbatch"
echo "=============================================================="

# ------------ Parameters ------------
N_PERMS=1000
SEED=42
QC_THR_C=0.60
QC_THR_S=0.60
PT_MIN=200
SLIDING_WINDOW_MS=20
SLIDING_STEP_MS=10

WIN_C_START=-0.30
WIN_C_END=-0.10
WIN_S_START=-0.10
WIN_S_END=-0.03
K=6

AXES_TAG="axes_peakbin_saccCS-sacc-horizontal-20mssw"
TAG="correctonly"

# Run per-session analysis
python cli/axis_alignment_analysis.py \
  --sid_list "$SID_LIST" \
  --out_root "$OUT_ROOT" \
  --mode both \
  --win_C "$WIN_C_START" "$WIN_C_END" \
  --win_S "$WIN_S_START" "$WIN_S_END" \
  --k "$K" \
  --axes_tag "$AXES_TAG" \
  --orientation horizontal \
  --pt_min_ms "$PT_MIN" \
  --n_perms "$N_PERMS" \
  --seed "$SEED" \
  --qc_threshold_C "$QC_THR_C" \
  --qc_threshold_S "$QC_THR_S" \
  --sliding_window_ms "$SLIDING_WINDOW_MS" \
  --sliding_step_ms "$SLIDING_STEP_MS" \
  --tag "$TAG"

# Run summary
python cli/summarize_alignment.py \
  --out_root "$OUT_ROOT" \
  --tag "$TAG" \
  --mode both

echo ""
echo "[done] Legacy axis alignment analysis complete"
echo "Results: $OUT_ROOT/sacc/summary/alignment_$TAG/"
