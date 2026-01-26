#!/bin/bash
#SBATCH --job-name=peakbin_20ms_lag40
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Adjust 1-23 to number of sessions; %32 limits concurrency
#SBATCH --array=1-23%32
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/peakbin_20ms_lag40_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/peakbin_20ms_lag40_%A_%a.err

# =============================================================================
# 20ms SLIDING WINDOW WORKFLOW - LAG40ms
#
# Purpose: Run STIM C analysis with lag40ms.
#
# Tags:
#   - Axes: axes_peakbin_stimCR-stim-vertical-20mssw (shared)
#   - Flow: evoked_peakbin_stimC_vertical_lag40ms_20mssw-none-trial
#
# Usage:
#   sbatch jobs/peakbin_axes_qc_flow_array_20ms_sliding_lag30_50.sbatch
#   sbatch --export=DRY_RUN=1 --array=1 jobs/peakbin_axes_qc_flow_array_20ms_sliding_lag30_50.sbatch
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
SID_LIST="$PROJECT/sid_list.txt"   # newline-delimited list of 8-digit SIDs

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
export PAPER_DATA="/project/bdoiron/dracoxu/rct-fsflow/paper_project_final/RCT_02"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

# Allow local dry-run (no SLURM env)
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# Build sid_list.txt if missing (from stim directory)
if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$OUT_ROOT/stim" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

SID="$(sed -n "${TASK_ID}p" "$SID_LIST" || true)"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $TASK_ID in $SID_LIST" >&2
  exit 0
fi

echo "[info] task=${TASK_ID} SID=${SID}"
echo "[info] PAPER_HOME=$PAPER_HOME"
echo "[info] OUT_ROOT=$OUT_ROOT"
echo "[info] DRY_RUN=$DRY_RUN CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"

# ------------ Shared analysis knobs ------------
ORI_STIM="vertical"      # vertical|horizontal|pooled for stim alignment
PT_MIN=200               # ms
QC_THR=0.60
QC_K=5

PERMS=500
RIDGE="1e-2"
PERM_WITHIN="CR"

# Evoked sigma: with 10ms output resolution, 10ms = 1 bin
EVOKED_SIGMA_MS=10

# === SLIDING WINDOW PARAMETERS ===
WINDOW_MS=20
STEP_MS=10

# Window search grid
SEARCH_STEP_MS=20
SEARCH_LEN_MS=(50 100 150 200 250)

# Peak-bin scoring + tie-break
SEARCH_SCORE_MODE="peak_bin_auc"
SEARCH_TOL="0.01"
STIM_TIEBREAK="shortest_then_earliest"

# Search ranges
SEARCH_RANGE_STIM_C="0.00:0.50"
SEARCH_RANGE_STIM_R="0.00:0.50"

# Norm / classifier
NORM="global"            # global|baseline|none
CLF="logreg"             # logreg|svm|lda
C_GRID=(0.1 0.3 1 3 10)
LDA_SHRINKAGE="auto"

# ------------ Tags (shared axes) ------------
AXES_TAG_STIM="axes_peakbin_stimCR-stim-${ORI_STIM}-20mssw"

# Helper: expand arrays into args
SEARCH_LEN_ARGS=()
for L in "${SEARCH_LEN_MS[@]}"; do SEARCH_LEN_ARGS+=("$L"); done

CGRID_ARGS=()
for C in "${C_GRID[@]}"; do CGRID_ARGS+=("$C"); done

# ======================================================================================
# STIM (C,R): train axes → QC (once, shared across lags)
# ======================================================================================
if [[ -d "$OUT_ROOT/stim/$SID/caches" ]]; then

  run_step "STIM 20ms sliding: train axes (C,R) peakbin" \
    python cli/train_axes.py \
      --out_root "$OUT_ROOT" \
      --align stim --sid "$SID" \
      --orientation "$ORI_STIM" \
      --features C R \
      --tag "$AXES_TAG_STIM" \
      --norm "$NORM" \
      --clf_binary "$CLF" \
      --C_grid "${CGRID_ARGS[@]}" \
      --lda_shrinkage "$LDA_SHRINKAGE" \
      --searchC --searchR \
      --search_range_C="$SEARCH_RANGE_STIM_C" \
      --search_range_R="$SEARCH_RANGE_STIM_R" \
      --search_step_ms "$SEARCH_STEP_MS" \
      --search_len_ms "${SEARCH_LEN_ARGS[@]}" \
      --search_score_mode "$SEARCH_SCORE_MODE" \
      --search_tiebreak "$STIM_TIEBREAK" \
      --search_tol "$SEARCH_TOL" \
      --pt_min_ms_stim "$PT_MIN" \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  run_step "STIM 20ms sliding: QC" \
    python cli/qc_axes.py \
      --out_root "$OUT_ROOT" \
      --align stim --sid "$SID" \
      --orientation "$ORI_STIM" \
      --tag "$AXES_TAG_STIM" \
      --thr "$QC_THR" --k "$QC_K" \
      --pt_min_ms_stim "$PT_MIN" \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  # ====================================================================================
  # LAG 40ms: flow(C) + flow(R)
  # ====================================================================================
  LAGS_STIM_MS=40
  FLOW_TAG_STIM_C="evoked_peakbin_stimC_${ORI_STIM}_lag${LAGS_STIM_MS}ms_20mssw"
  FLOW_TAG_STIM_R="evoked_peakbin_stimR_${ORI_STIM}_lag${LAGS_STIM_MS}ms_20mssw"

  echo ""
  echo "==================== LAG 40ms ===================="

  run_step "STIM lag40ms: flow feature=C (save null)" \
    python cli/flow_session.py \
      --out_root "$OUT_ROOT" \
      --align stim --sid "$SID" \
      --feature C \
      --orientation "$ORI_STIM" \
      --lags_ms "$LAGS_STIM_MS" \
      --ridge "$RIDGE" \
      --perms "$PERMS" \
      --perm-within "$PERM_WITHIN" \
      --null_method trial_shuffle \
      --standardize_mode none \
      --pt_min_ms "$PT_MIN" \
      --no_induced \
      --evoked_subtract --evoked_sigma_ms "$EVOKED_SIGMA_MS" \
      --axes_tag "$AXES_TAG_STIM" \
      --tag "$FLOW_TAG_STIM_C" \
      --flow_tag_base "$FLOW_TAG_STIM_C" \
      --save_null_samples \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  run_step "STIM lag40ms: flow feature=R (save null)" \
    python cli/flow_session.py \
      --out_root "$OUT_ROOT" \
      --align stim --sid "$SID" \
      --feature R \
      --orientation "$ORI_STIM" \
      --lags_ms "$LAGS_STIM_MS" \
      --ridge "$RIDGE" \
      --perms "$PERMS" \
      --perm-within "$PERM_WITHIN" \
      --null_method trial_shuffle \
      --standardize_mode none \
      --pt_min_ms "$PT_MIN" \
      --no_induced \
      --evoked_subtract --evoked_sigma_ms "$EVOKED_SIGMA_MS" \
      --axes_tag "$AXES_TAG_STIM" \
      --tag "$FLOW_TAG_STIM_R" \
      --flow_tag_base "$FLOW_TAG_STIM_R" \
      --save_null_samples \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

else
  echo "[WARN] No stim caches for SID=$SID → skipping stim"
fi

echo ""
echo "[done] 20ms sliding window peakbin LAG40ms complete for SID=$SID"
echo ""
echo "============================================================"
echo "LAG40ms workflow completed. Next steps:"
echo "  1. Run summarize_flow_across_sessions.py with:"
echo "     --tags evoked_peakbin_stimC_vertical_lag40ms_20mssw-none-trial"
echo "     --smooth_ms 30"
echo "============================================================"
