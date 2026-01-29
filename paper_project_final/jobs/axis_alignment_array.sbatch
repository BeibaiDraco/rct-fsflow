#!/bin/bash
#SBATCH --job-name=align_arr
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Adjust 1-23 to number of sessions; %32 limits concurrency
#SBATCH --array=1-23%32
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_arr_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_arr_%A_%a.err

# =============================================================================
# AXIS ALIGNMENT ANALYSIS - JOB ARRAY (Per-Session)
#
# This script runs alignment analysis for ONE session.
# Use jobs/axis_alignment_summary.sbatch after all sessions complete.
#
# THREE complementary analyses per session:
#   A) Alignment Index + Covariance-Matched Null (PCA subspaces)
#   B) Axis Angle + Label-Shuffle Null (label-specificity test)
#   C) Axis Angle + Covariance-Constrained Null (DAVE'S QUESTION - axis orthogonality)
#
# By default, runs BOTH:
#   1. Correct-only trials (out/)
#   2. All trials (out_nofilter/) - still applies PT > 200ms, orientation filters
#
# Prerequisites:
#   - Sacc-aligned caches must exist in both out/ and out_nofilter/
#   - For Analysis B & C: Run train_saccCS_axes.sbatch first (for both)
#
# Usage:
#   # Run both correctonly and nofilter (default)
#   sbatch jobs/axis_alignment_array.sbatch
#
#   # Run only correct-only trials
#   FILTER_MODE=correctonly sbatch jobs/axis_alignment_array.sbatch
#
#   # Run only all trials (nofilter)
#   FILTER_MODE=nofilter sbatch jobs/axis_alignment_array.sbatch
#
#   # Run for specific session (task 5)
#   sbatch --array=5 jobs/axis_alignment_array.sbatch
# =============================================================================

set -euo pipefail

# ------------ Filter mode (both, correctonly, or nofilter) ------------
# Set via environment variable: FILTER_MODE=correctonly sbatch ...
FILTER_MODE="${FILTER_MODE:-both}"

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
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

# Get session ID from array task
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$PROJECT/out/sacc" | grep -E '^[0-9]{8}$' > "$SID_LIST" 2>/dev/null || \
  ls -1 "$PAPER_DATA" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

SID="$(sed -n "${TASK_ID}p" "$SID_LIST" || true)"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $TASK_ID in $SID_LIST" >&2
  exit 0
fi

echo "=============================================================="
echo "AXIS ALIGNMENT ANALYSIS - Session $SID (task $TASK_ID)"
echo "Filter mode: $FILTER_MODE"
echo "=============================================================="

# ------------ Common Parameters ------------
N_PERMS=1000
SEED=42
QC_THR_C=0.60
QC_THR_S=0.60
PT_MIN=200
SLIDING_WINDOW_MS=20
SLIDING_STEP_MS=10

# Analysis A: Alignment Index windows
WIN_C_START=-0.30
WIN_C_END=-0.10
WIN_S_START=-0.10
WIN_S_END=-0.03
K=6

# =============================================================================
# Function to run alignment analysis
# =============================================================================
run_alignment() {
  local OUT_ROOT="$1"
  local AXES_TAG="$2"
  local TAG="$3"
  local LABEL="$4"
  
  if [[ -d "$OUT_ROOT/sacc/$SID/caches" ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[info] Running alignment analysis for $SID ($LABEL)"
    echo "       Mode: all (AI + axis_shuffle + axis_cov)"
    echo "       OUT_ROOT: $OUT_ROOT"
    echo "       AXES_TAG: $AXES_TAG"
    echo "       n_perms: $N_PERMS"
    echo "------------------------------------------------------------"
    
    python cli/axis_alignment_analysis.py \
      --sid "$SID" \
      --out_root "$OUT_ROOT" \
      --mode all \
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
    
    echo "[done] $LABEL complete → $OUT_ROOT/sacc/alignment/$TAG/"
  else
    echo "[WARN] No sacc caches for SID=$SID in $OUT_ROOT → skipping $LABEL"
  fi
}

# =============================================================================
# Run alignment analysis based on filter mode
# =============================================================================

# 1. Correct-only trials
if [[ "$FILTER_MODE" == "both" || "$FILTER_MODE" == "correctonly" ]]; then
  run_alignment \
    "$PROJECT/out" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw" \
    "correctonly" \
    "CORRECT-ONLY trials"
fi

# 2. All trials (nofilter - still applies PT > 200ms, orientation filters)
if [[ "$FILTER_MODE" == "both" || "$FILTER_MODE" == "nofilter" ]]; then
  run_alignment \
    "$PROJECT/out_nofilter" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw_nofilter" \
    "nofilter" \
    "ALL trials (nofilter)"
fi

echo ""
echo "=============================================================="
echo "[done] Alignment analysis complete for SID=$SID"
echo "=============================================================="
