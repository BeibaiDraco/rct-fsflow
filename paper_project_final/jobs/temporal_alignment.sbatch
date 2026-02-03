#!/bin/bash
#SBATCH --job-name=temp_align
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-23%16
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/temp_align_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/temp_align_%A_%a.err

# =============================================================================
# TEMPORAL C-S ALIGNMENT ANALYSIS
#
# Studies how C-S axis alignment changes over time by training C axes at
# multiple time points using a sliding window.
#
# For each trial type (horizontal, pooled, vertical):
#   1. Load pre-trained S axis (saccade-aligned, window search optimal)
#   2. Get optimal C window length from trained axes
#   3. Slide this window across time (step = 20ms)
#   4. Train C axis at each position, compute alignment with S
#   5. Output: alignment vs time curve
#
# Three analysis cases (matching trial types):
#   - Case i/iv: horizontal trials
#   - Case ii/v: pooled trials
#   - Case iii/vi: vertical trials
#
# Run for both correct-only (out/) and all-trials (out_nofilter/).
#
# Usage:
#   sbatch jobs/temporal_alignment.sbatch
#
#   # Run specific session
#   sbatch --array=5 jobs/temporal_alignment.sbatch
# =============================================================================

set -euo pipefail

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
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

# Get session ID from array task
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$PROJECT/out/sacc" | grep -E '^[0-9]{8}$' > "$SID_LIST" 2>/dev/null || \
  ls -1 "$PROJECT/RCT_02" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

SID="$(sed -n "${TASK_ID}p" "$SID_LIST" || true)"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $TASK_ID in $SID_LIST" >&2
  exit 0
fi

echo "=============================================================="
echo "TEMPORAL C-S ALIGNMENT ANALYSIS"
echo "Session: $SID (task $TASK_ID)"
echo "=============================================================="

# ------------ Parameters ------------
STEP_MS=20
TIME_RANGE="-350 200"
PT_MIN=200
SW_MS=20
SW_STEP=10

# =============================================================================
# Run for all orientations and filter modes
# =============================================================================

run_orientation() {
  local ORIENT="$1"
  local OUT_ROOT="$2"
  local FILTER_TAG="$3"
  local AXES_TAG="axes_peakbin_saccCS-sacc-${ORIENT}-20mssw"
  
  if [[ "$OUT_ROOT" == *"nofilter"* ]]; then
    AXES_TAG="${AXES_TAG}_nofilter"
  fi
  
  echo ""
  echo "------------------------------------------------------------"
  echo "Orientation: $ORIENT, Filter: $FILTER_TAG"
  echo "Axes tag: $AXES_TAG"
  echo "------------------------------------------------------------"
  
  python cli/temporal_alignment_analysis.py \
    --sid "$SID" \
    --orientation "$ORIENT" \
    --out_root "$OUT_ROOT" \
    --axes_tag "$AXES_TAG" \
    --step_ms "$STEP_MS" \
    --time_range $TIME_RANGE \
    --pt_min_ms "$PT_MIN" \
    --sliding_window_ms "$SW_MS" \
    --sliding_step_ms "$SW_STEP" \
    --tag "temporal_alignment_${ORIENT}_${FILTER_TAG}" \
    2>&1 || echo "[WARN] Failed for $ORIENT ($FILTER_TAG)"
}

# CORRECT-ONLY TRIALS
echo ""
echo "=============================================================="
echo "CORRECT-ONLY TRIALS (out/)"
echo "=============================================================="

run_orientation "horizontal" "$PROJECT/out" "correctonly"
run_orientation "pooled" "$PROJECT/out" "correctonly"
run_orientation "vertical" "$PROJECT/out" "correctonly"

# ALL TRIALS
echo ""
echo "=============================================================="
echo "ALL TRIALS (out_nofilter/)"
echo "=============================================================="

run_orientation "horizontal" "$PROJECT/out_nofilter" "alltrials"
run_orientation "pooled" "$PROJECT/out_nofilter" "alltrials"
run_orientation "vertical" "$PROJECT/out_nofilter" "alltrials"

echo ""
echo "=============================================================="
echo "[done] Temporal alignment analysis complete for SID=$SID"
echo "=============================================================="
