#!/bin/bash
#SBATCH --job-name=train_saccCS
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-23%32
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/train_saccCS_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/train_saccCS_%A_%a.err

# =============================================================================
# TRAIN C+S AXES ON SACC-ALIGNED DATA
#
# Purpose: Train both Category (C) and Saccade (S) axes on sacc-aligned data
#          for the axis alignment analysis (Dave's multiplexing question).
#
# This creates axes where both C and S are defined in the same coordinate
# system (same trials, same normalization, same time reference).
#
# Two versions:
#   1. Correct-only (out/) - tag: axes_peakbin_saccCS-sacc-horizontal-20mssw
#   2. All trials (out_nofilter/) - tag: axes_peakbin_saccCS-sacc-horizontal-20mssw_nofilter
#
# Usage:
#   sbatch jobs/train_saccCS_axes.sbatch
# =============================================================================

set -euo pipefail

# ------------------- Debug controls -------------------
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

run_cmd () {
  local -a cmd=("$@")
  echo ""
  echo "[RUN] ${cmd[*]}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "${cmd[@]}"
}

run_step () {
  local name="$1"; shift
  echo ""
  echo "========== ${name} =========="
  if run_cmd "$@"; then
    return 0
  else
    local rc=$?
    echo "[ERROR] step failed rc=${rc}: ${name}" >&2
    if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
      echo "[WARN] continuing despite failure (CONTINUE_ON_ERROR=1)" >&2
      return 0
    fi
    exit $rc
  fi
}

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
OUT_ROOT="$PROJECT/out"
OUT_NOFILTER="$PROJECT/out_nofilter"
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

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$OUT_ROOT/sacc" | grep -E '^[0-9]{8}$' > "$SID_LIST" 2>/dev/null || \
  ls -1 "$PAPER_DATA" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

SID="$(sed -n "${TASK_ID}p" "$SID_LIST" || true)"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $TASK_ID in $SID_LIST" >&2
  exit 0
fi

echo "[info] task=${TASK_ID} SID=${SID}"
echo "[info] Training C+S on sacc-aligned data for axis alignment analysis"

# ------------ Parameters (matching existing sacc jobs) ------------
ORI_SACC="horizontal"
PT_MIN=200

# Sliding window
WINDOW_MS=20
STEP_MS=10

# Window search grid
SEARCH_STEP_MS=20
SEARCH_LEN_MS=(50 100 150 200 250)

# Peak-bin scoring + tie-break
SEARCH_SCORE_MODE="peak_bin_auc"
SEARCH_TOL="0.01"
SACC_TIEBREAK="shortest_then_closest0"

# Search ranges for sacc-aligned data
# C: pre-saccade window where category information persists
# S: pre-saccade window for saccade direction
SEARCH_RANGE_SACC_C="-0.30:-0.05"
SEARCH_RANGE_SACC_S="-0.20:0.05"

# Classifier settings
NORM="global"
CLF="logreg"
C_GRID=(0.1 0.3 1 3 10)
LDA_SHRINKAGE="auto"

# QC settings
QC_THR=0.60
QC_K=5

# ------------ Tags ------------
AXES_TAG_CORRECT="axes_peakbin_saccCS-sacc-${ORI_SACC}-20mssw"
AXES_TAG_NOFILTER="axes_peakbin_saccCS-sacc-${ORI_SACC}-20mssw_nofilter"

# Helper: expand arrays
SEARCH_LEN_ARGS=()
for L in "${SEARCH_LEN_MS[@]}"; do SEARCH_LEN_ARGS+=("$L"); done

CGRID_ARGS=()
for C in "${C_GRID[@]}"; do CGRID_ARGS+=("$C"); done

# =============================================================================
# 1. CORRECT-ONLY: Train C+S on sacc-aligned data (out/)
# =============================================================================
if [[ -d "$OUT_ROOT/sacc/$SID/caches" ]]; then
  echo ""
  echo "==================== CORRECT-ONLY: Train C+S on sacc-aligned ===================="
  
  run_step "Train C+S axes on sacc-aligned (correct-only)" \
    python cli/train_axes.py \
      --out_root "$OUT_ROOT" \
      --align sacc --sid "$SID" \
      --orientation "$ORI_SACC" \
      --features C S \
      --tag "$AXES_TAG_CORRECT" \
      --norm "$NORM" \
      --clf_binary "$CLF" \
      --C_grid "${CGRID_ARGS[@]}" \
      --lda_shrinkage "$LDA_SHRINKAGE" \
      --searchC --searchS \
      --search_range_C="$SEARCH_RANGE_SACC_C" \
      --search_range_S="$SEARCH_RANGE_SACC_S" \
      --search_step_ms "$SEARCH_STEP_MS" \
      --search_len_ms "${SEARCH_LEN_ARGS[@]}" \
      --search_score_mode "$SEARCH_SCORE_MODE" \
      --search_tiebreak "$SACC_TIEBREAK" \
      --search_tol "$SEARCH_TOL" \
      --pt_min_ms_sacc "$PT_MIN" \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  run_step "QC for C+S axes (correct-only)" \
    python cli/qc_axes.py \
      --out_root "$OUT_ROOT" \
      --align sacc --sid "$SID" \
      --orientation "$ORI_SACC" \
      --tag "$AXES_TAG_CORRECT" \
      --thr "$QC_THR" --k "$QC_K" \
      --pt_min_ms_sacc "$PT_MIN" \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"
else
  echo "[WARN] No sacc caches in $OUT_ROOT for SID=$SID"
fi

# =============================================================================
# 2. NOFILTER: Train C+S on sacc-aligned data (out_nofilter/)
# =============================================================================
if [[ -d "$OUT_NOFILTER/sacc/$SID/caches" ]]; then
  echo ""
  echo "==================== NOFILTER: Train C+S on sacc-aligned ===================="
  
  run_step "Train C+S axes on sacc-aligned (nofilter)" \
    python cli/train_axes.py \
      --out_root "$OUT_NOFILTER" \
      --align sacc --sid "$SID" \
      --orientation "$ORI_SACC" \
      --features C S \
      --tag "$AXES_TAG_NOFILTER" \
      --norm "$NORM" \
      --clf_binary "$CLF" \
      --C_grid "${CGRID_ARGS[@]}" \
      --lda_shrinkage "$LDA_SHRINKAGE" \
      --searchC --searchS \
      --search_range_C="$SEARCH_RANGE_SACC_C" \
      --search_range_S="$SEARCH_RANGE_SACC_S" \
      --search_step_ms "$SEARCH_STEP_MS" \
      --search_len_ms "${SEARCH_LEN_ARGS[@]}" \
      --search_score_mode "$SEARCH_SCORE_MODE" \
      --search_tiebreak "$SACC_TIEBREAK" \
      --search_tol "$SEARCH_TOL" \
      --pt_min_ms_sacc "$PT_MIN" \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  run_step "QC for C+S axes (nofilter)" \
    python cli/qc_axes.py \
      --out_root "$OUT_NOFILTER" \
      --align sacc --sid "$SID" \
      --orientation "$ORI_SACC" \
      --tag "$AXES_TAG_NOFILTER" \
      --thr "$QC_THR" --k "$QC_K" \
      --pt_min_ms_sacc "$PT_MIN" \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"
else
  echo "[WARN] No sacc caches in $OUT_NOFILTER for SID=$SID"
fi

echo ""
echo "[done] C+S axes training complete for SID=$SID"
