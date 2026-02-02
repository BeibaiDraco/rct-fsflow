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
# TRAIN AXES FOR ALL SIX CASES
#
# Trains the necessary axes for running axis_alignment_all_cases.sbatch.
#
# THE SIX CASES:
#   Case i:   Sacc-aligned C & S, horizontal trials
#   Case ii:  Sacc-aligned C & S, pooled trials
#   Case iii: Stim-aligned C (vertical) vs Sacc-aligned S (horizontal)
#   Case iv:  Stim-aligned C (pooled) vs Sacc-aligned S (pooled)
#   Case v:   Stim-aligned C (horizontal) vs Sacc-aligned S (horizontal)
#   Case vi:  Sacc-aligned C & S, vertical trials
#
# Required axes for CORRECT-ONLY (out/):
#   Case i:   axes_peakbin_saccCS-sacc-horizontal-20mssw
#   Case ii:  axes_peakbin_saccCS-sacc-pooled-20mssw
#   Case iii: axes_peakbin_stimC-stim-vertical-20mssw (C only)
#             + axes_peakbin_saccCS-sacc-horizontal-20mssw (S from Case i)
#   Case iv:  axes_peakbin_stimC-stim-pooled-20mssw (C only)
#             + axes_peakbin_saccCS-sacc-pooled-20mssw (S from Case ii)
#   Case v:   axes_peakbin_stimC-stim-horizontal-20mssw (C only)
#             + axes_peakbin_saccCS-sacc-horizontal-20mssw (S from Case i)
#   Case vi:  axes_peakbin_saccCS-sacc-vertical-20mssw
#
# Required axes for ALL-TRIALS (out_nofilter/):
#   Same as above but with _nofilter suffix
#
# Usage:
#   # Train for both trial modes (default, skip existing)
#   sbatch jobs/train_axes_all_cases.sbatch
#
#   # Force retrain (overwrite existing axes)
#   FORCE_RETRAIN=1 sbatch jobs/train_axes_all_cases.sbatch
#
#   # Train only for correct-only
#   FILTER_MODE=correctonly sbatch jobs/train_axes_all_cases.sbatch
#
#   # Train only for all-trials
#   FILTER_MODE=alltrials sbatch jobs/train_axes_all_cases.sbatch
#
#   # Force retrain only correct-only
#   FORCE_RETRAIN=1 FILTER_MODE=correctonly sbatch jobs/train_axes_all_cases.sbatch
# =============================================================================

set -euo pipefail

FILTER_MODE="${FILTER_MODE:-both}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"  # Set to 1 to overwrite existing axes

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
echo "Force retrain: $FORCE_RETRAIN"
echo "=============================================================="

# ------------ Common Parameters ------------
PT_MIN=200
SLIDING_WINDOW_MS=20
SLIDING_STEP_MS=10

# === SEARCH PARAMETERS (ALIGNMENT-BASED, NOT FEATURE-BASED) ===
# Key design choice: Use the SAME search range for ALL features within an alignment.
# This ensures C and S axes are trained on comparable time windows, which is 
# important for the alignment analysis null construction.
#
# STIM-aligned: Search within [0.00, 0.50] s relative to stimulus onset
# SACC-aligned: Search within [-0.20, 0.05] s relative to saccade onset
#
SEARCH_RANGE_STIM="0.00:0.50"
SEARCH_RANGE_SACC="-0.20:0.05"

# Window search parameters
SEARCH_STEP_MS=20
SEARCH_LEN_MS="50 100 150 200 250"

# Peak-bin scoring + tie-break
SEARCH_SCORE_MODE="peak_bin_auc"
SEARCH_TOL="0.01"
STIM_TIEBREAK="shortest_then_earliest"
SACC_TIEBREAK="shortest_then_closest0"

# Classifier settings
NORM="global"
CLF="logreg"
C_GRID="0.1 0.3 1 3 10"
LDA_SHRINKAGE="auto"

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
  
  # Handle existing axes
  if [[ -d "$AXES_DIR" ]]; then
    if [[ "$FORCE_RETRAIN" -eq 1 ]]; then
      echo "[overwrite] $LABEL - removing existing axes"
      rm -rf "$AXES_DIR"
    else
      echo "[skip] $LABEL - already exists (use FORCE_RETRAIN=1 to overwrite)"
      return 0
    fi
  fi
  
  # Determine search range and tiebreak based on ALIGNMENT (not feature)
  local SEARCH_RANGE TIEBREAK
  if [[ "$ALIGN" == "stim" ]]; then
    SEARCH_RANGE="$SEARCH_RANGE_STIM"
    TIEBREAK="$STIM_TIEBREAK"
  else
    SEARCH_RANGE="$SEARCH_RANGE_SACC"
    TIEBREAK="$SACC_TIEBREAK"
  fi
  
  echo ""
  echo "------------------------------------------------------------"
  echo "[train] $LABEL"
  echo "  Align: $ALIGN, Features: $FEATURES, Orientation: $ORIENTATION"
  echo "  Search range: $SEARCH_RANGE (alignment-based)"
  echo "  Window lengths: $SEARCH_LEN_MS ms"
  echo "  Tag: $TAG"
  echo "  OUT_ROOT: $OUT_ROOT"
  echo "------------------------------------------------------------"
  
  # Build search flags based on features
  local SEARCH_FLAGS=""
  local RANGE_FLAGS=""
  if [[ "$FEATURES" == *"C"* ]]; then
    SEARCH_FLAGS="$SEARCH_FLAGS --searchC"
    RANGE_FLAGS="$RANGE_FLAGS --search_range_C=$SEARCH_RANGE"
  fi
  if [[ "$FEATURES" == *"S"* ]]; then
    SEARCH_FLAGS="$SEARCH_FLAGS --searchS"
    RANGE_FLAGS="$RANGE_FLAGS --search_range_S=$SEARCH_RANGE"
  fi
  if [[ "$FEATURES" == *"R"* ]]; then
    SEARCH_FLAGS="$SEARCH_FLAGS --searchR"
    RANGE_FLAGS="$RANGE_FLAGS --search_range_R=$SEARCH_RANGE"
  fi
  
  python cli/train_axes.py \
    --out_root "$OUT_ROOT" \
    --align "$ALIGN" \
    --sid "$SID" \
    --features $FEATURES \
    --orientation "$ORIENTATION" \
    --tag "$TAG" \
    --norm "$NORM" \
    --clf_binary "$CLF" \
    --C_grid $C_GRID \
    --lda_shrinkage "$LDA_SHRINKAGE" \
    $SEARCH_FLAGS \
    $RANGE_FLAGS \
    --search_step_ms "$SEARCH_STEP_MS" \
    --search_len_ms $SEARCH_LEN_MS \
    --search_score_mode "$SEARCH_SCORE_MODE" \
    --search_tiebreak "$TIEBREAK" \
    --search_tol "$SEARCH_TOL" \
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
  
  # Case v: Stim-aligned C (horizontal) - S uses sacc-aligned from Case i
  train_axes \
    "$PROJECT/out" \
    "stim" \
    "C" \
    "horizontal" \
    "axes_peakbin_stimC-stim-horizontal-20mssw" \
    "Case v: Stim C, horizontal (correct-only)"
  
  # Case vi: Sacc-aligned C+S, vertical
  train_axes \
    "$PROJECT/out" \
    "sacc" \
    "C S" \
    "vertical" \
    "axes_peakbin_saccCS-sacc-vertical-20mssw" \
    "Case vi: Sacc C+S, vertical (correct-only)"
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
  
  # Case v: Stim-aligned C (horizontal) - S uses sacc-aligned from Case i
  train_axes \
    "$PROJECT/out_nofilter" \
    "stim" \
    "C" \
    "horizontal" \
    "axes_peakbin_stimC-stim-horizontal-20mssw_nofilter" \
    "Case v: Stim C, horizontal (all-trials)"
  
  # Case vi: Sacc-aligned C+S, vertical
  train_axes \
    "$PROJECT/out_nofilter" \
    "sacc" \
    "C S" \
    "vertical" \
    "axes_peakbin_saccCS-sacc-vertical-20mssw_nofilter" \
    "Case vi: Sacc C+S, vertical (all-trials)"
fi

echo ""
echo "=============================================================="
echo "[done] All axes training complete for SID=$SID"
echo "=============================================================="
