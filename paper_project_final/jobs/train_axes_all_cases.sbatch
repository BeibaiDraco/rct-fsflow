#!/bin/bash
#SBATCH --job-name=train_axes_all
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Adjust 1-23 to number of sessions; %16 limits concurrency
#SBATCH --array=1-23%16
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/train_axes_all_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/train_axes_all_%A_%a.err

# =============================================================================
# TRAIN AXES FOR ALL FOUR CASES
#
# Trains the necessary axes for running axis_alignment_all_cases.sbatch.
#
# Required axes for CORRECT-ONLY (out/):
#   Case i:   axes_peakbin_saccCS-sacc-horizontal-20mssw
#   Case ii:  axes_peakbin_saccCS-sacc-pooled-20mssw
#   Case iii: axes_peakbin_stimC-stim-vertical-20mssw (C only)
#             + axes_peakbin_saccCS-sacc-horizontal-20mssw (S from same as Case i)
#   Case iv:  axes_peakbin_stimC-stim-pooled-20mssw (C only)
#             + axes_peakbin_saccCS-sacc-pooled-20mssw (S from same as Case ii)
#
# Required axes for ALL-TRIALS (out_nofilter/):
#   Same as above but with _nofilter suffix
#
# Usage:
#   # Train for both trial modes (default)
#   sbatch jobs/train_axes_all_cases.sbatch
#
#   # Train only for correct-only
#   FILTER_MODE=correctonly sbatch jobs/train_axes_all_cases.sbatch
#
#   # Train only for all-trials
#   FILTER_MODE=alltrials sbatch jobs/train_axes_all_cases.sbatch
# =============================================================================

set -euo pipefail

FILTER_MODE="${FILTER_MODE:-both}"

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
SID_LIST="$PROJECT/sid_list.txt"

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PAPER_HOME="$PROJECT"
export PAPER_DATA="$PROJECT/RCT_02"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$PROJECT/out/sacc" | grep -E '^[0-9]{8}$' > "$SID_LIST" 2>/dev/null || \
  ls -1 "$PAPER_DATA" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

SID="$(sed -n "${TASK_ID}p" "$SID_LIST" || true)"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $TASK_ID" >&2
  exit 0
fi

echo "=============================================================="
echo "TRAIN AXES FOR ALL FOUR CASES - Session $SID"
echo "Filter mode: $FILTER_MODE"
echo "=============================================================="

# ------------ Common Parameters ------------
PT_MIN=200
SLIDING_WINDOW_MS=20
SLIDING_STEP_MS=10

# =============================================================================
# Helper function to train axes
# =============================================================================
train_axes() {
  local OUT_ROOT="$1"
  local ALIGN="$2"
  local FEATURES="$3"
  local ORIENTATION="$4"
  local TAG="$5"
  local LABEL="$6"
  
  local AXES_DIR="$OUT_ROOT/$ALIGN/$SID/axes/$TAG"
  
  # Skip if already exists
  if [[ -d "$AXES_DIR" ]]; then
    echo "[skip] $LABEL - already exists"
    return 0
  fi
  
  echo ""
  echo "------------------------------------------------------------"
  echo "[train] $LABEL"
  echo "  Align: $ALIGN, Features: $FEATURES, Orientation: $ORIENTATION"
  echo "  Tag: $TAG"
  echo "  OUT_ROOT: $OUT_ROOT"
  echo "------------------------------------------------------------"
  
  python cli/train_axes.py \
    --out_root "$OUT_ROOT" \
    --align "$ALIGN" \
    --sid "$SID" \
    --features $FEATURES \
    --orientation "$ORIENTATION" \
    --tag "$TAG" \
    --searchC --searchS \
    --search_score_mode peak_bin_auc \
    --search_tiebreak shortest_then_earliest \
    --search_tol 0.01 \
    --sliding_window_ms "$SLIDING_WINDOW_MS" \
    --sliding_step_ms "$SLIDING_STEP_MS" \
    --pt_min_ms_sacc "$PT_MIN" \
    --pt_min_ms_stim "$PT_MIN"
  
  echo "[done] $TAG"
}

# =============================================================================
# CORRECT-ONLY (out/)
# =============================================================================
if [[ "$FILTER_MODE" == "both" || "$FILTER_MODE" == "correctonly" ]]; then
  echo ""
  echo "=============================================================="
  echo "TRAINING AXES - CORRECT-ONLY (out/)"
  echo "=============================================================="
  
  # Case i: Sacc-aligned C+S, horizontal
  train_axes \
    "$PROJECT/out" \
    "sacc" \
    "C S" \
    "horizontal" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw" \
    "Case i: Sacc C+S, horizontal (correct-only)"
  
  # Case ii: Sacc-aligned C+S, pooled
  train_axes \
    "$PROJECT/out" \
    "sacc" \
    "C S" \
    "pooled" \
    "axes_peakbin_saccCS-sacc-pooled-20mssw" \
    "Case ii: Sacc C+S, pooled (correct-only)"
  
  # Case iii: Stim-aligned C only, vertical (already exists as axes_peakbin_stimC-stim-vertical-20mssw)
  train_axes \
    "$PROJECT/out" \
    "stim" \
    "C" \
    "vertical" \
    "axes_peakbin_stimC-stim-vertical-20mssw" \
    "Case iii: Stim C, vertical (correct-only)"
  
  # Case iv: Stim-aligned C only, pooled (already exists as axes_peakbin_stimC-stim-pooled-20mssw)
  train_axes \
    "$PROJECT/out" \
    "stim" \
    "C" \
    "pooled" \
    "axes_peakbin_stimC-stim-pooled-20mssw" \
    "Case iv: Stim C, pooled (correct-only)"
fi

# =============================================================================
# ALL-TRIALS (out_nofilter/)
# =============================================================================
if [[ "$FILTER_MODE" == "both" || "$FILTER_MODE" == "alltrials" ]]; then
  echo ""
  echo "=============================================================="
  echo "TRAINING AXES - ALL-TRIALS (out_nofilter/)"
  echo "=============================================================="
  
  # Case i: Sacc-aligned C+S, horizontal
  train_axes \
    "$PROJECT/out_nofilter" \
    "sacc" \
    "C S" \
    "horizontal" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw_nofilter" \
    "Case i: Sacc C+S, horizontal (all-trials)"
  
  # Case ii: Sacc-aligned C+S, pooled
  train_axes \
    "$PROJECT/out_nofilter" \
    "sacc" \
    "C S" \
    "pooled" \
    "axes_peakbin_saccCS-sacc-pooled-20mssw_nofilter" \
    "Case ii: Sacc C+S, pooled (all-trials)"
  
  # Case iii: Stim-aligned C only, vertical
  train_axes \
    "$PROJECT/out_nofilter" \
    "stim" \
    "C" \
    "vertical" \
    "axes_peakbin_stimC-stim-vertical-20mssw_nofilter" \
    "Case iii: Stim C, vertical (all-trials)"
  
  # Case iv: Stim-aligned C only, pooled
  train_axes \
    "$PROJECT/out_nofilter" \
    "stim" \
    "C" \
    "pooled" \
    "axes_peakbin_stimC-stim-pooled-20mssw_nofilter" \
    "Case iv: Stim C, pooled (all-trials)"
fi

echo ""
echo "=============================================================="
echo "[done] All axes training complete for SID=$SID"
echo "=============================================================="
