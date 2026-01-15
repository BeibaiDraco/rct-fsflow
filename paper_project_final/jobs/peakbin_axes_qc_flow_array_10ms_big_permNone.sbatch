#!/bin/bash
#SBATCH --job-name=peakbin_permNone
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Adjust 1-23 to number of sessions; %32 limits concurrency
#SBATCH --array=1-23%32
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/peakbin_permNone_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/peakbin_permNone_%A_%a.err

# =============================================================================
# STIM (C,R) + SACC 10ms (S) - PERM-WITHIN "none" (unrestricted shuffle)
#
# Combines:
#   - STIM from peakbin_axes_qc_flow_array_big.sbatch (vertical, 80ms lag)
#   - SACC from peakbin_axes_qc_flow_array_10ms_big.sbatch (horizontal, 50ms, 10ms bins)
#
# Uses --perm-within none (unrestricted). Flow tags include "_permNone" suffix.
#
# Usage:
#   sbatch jobs/peakbin_axes_qc_flow_array_10ms_big_permNone.sbatch
#   sbatch --export=DRY_RUN=1 --array=1 jobs/peakbin_axes_qc_flow_array_10ms_big_permNone.sbatch
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
SID_LIST="$PROJECT/sid_list.txt"

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

# Build sid_list.txt if missing
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
ORI_STIM="vertical"      # for stim alignment
ORI_SACC="horizontal"    # for sacc alignment
PT_MIN=200
QC_THR=0.60
QC_K=5

PERMS=500
RIDGE="1e-2"
EVOKED_SIGMA_MS=10

# *** KEY DIFFERENCE: perm-within "none" (unrestricted shuffle) ***
PERM_WITHIN="none"

# Lags
LAGS_STIM_MS=80
LAGS_SACC_MS=50

# REBIN FACTOR for SACC only: 2 = combine 2 adjacent 5ms bins → 10ms bins
REBIN_FACTOR=2

# Window search grid
SEARCH_STEP_MS=20
SEARCH_LEN_MS=(50 100 150 200 250)

# Peak-bin scoring + tie-break
SEARCH_SCORE_MODE="peak_bin_auc"
SEARCH_TOL="0.01"
STIM_TIEBREAK="shortest_then_earliest"
SACC_TIEBREAK="shortest_then_closest0"

# Search ranges
SEARCH_RANGE_STIM_C="0.00:0.50"
SEARCH_RANGE_STIM_R="0.00:0.50"
SEARCH_RANGE_SACC_S="-0.20:0.05"

# Norm / classifier
NORM="global"
CLF="logreg"
C_GRID=(0.1 0.3 1 3 10)
LDA_SHRINKAGE="auto"

# ------------ Tags ------------
# Axes tags: SAME as original (axes don't depend on permutation strategy)
AXES_TAG_STIM="axes_peakbin_stimCR-stim-${ORI_STIM}"
AXES_TAG_SACC="axes_peakbin_saccS-sacc-${ORI_SACC}-10msbin"

# Flow tags: include _permNone suffix to distinguish from CR results
FLOW_TAG_BASE_STIM_C="evoked_peakbin_stimC_${ORI_STIM}_lag${LAGS_STIM_MS}ms_permNone"
FLOW_TAG_BASE_STIM_R="evoked_peakbin_stimR_${ORI_STIM}_lag${LAGS_STIM_MS}ms_permNone"
FLOW_TAG_BASE_SACC_S="evoked_peakbin_saccS_${ORI_SACC}_lag${LAGS_SACC_MS}ms_10msbin_permNone"

# Helper: expand arrays into args
SEARCH_LEN_ARGS=()
for L in "${SEARCH_LEN_MS[@]}"; do SEARCH_LEN_ARGS+=("$L"); done

CGRID_ARGS=()
for C in "${C_GRID[@]}"; do CGRID_ARGS+=("$C"); done

# ======================================================================================
# STIM (C,R): reuse existing axes → flow with perm-within "none"
# ======================================================================================
if [[ -d "$OUT_ROOT/stim/$SID/caches" ]]; then

  AXES_PATH="$OUT_ROOT/stim/$SID/axes/$AXES_TAG_STIM"
  if [[ -d "$AXES_PATH" ]]; then
    echo "[info] STIM axes already exist at $AXES_PATH, reusing"
  else
    run_step "STIM: train axes (C,R) peakbin" \
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
        --pt_min_ms_stim "$PT_MIN"

    run_step "STIM: QC" \
      python cli/qc_axes.py \
        --out_root "$OUT_ROOT" \
        --align stim --sid "$SID" \
        --orientation "$ORI_STIM" \
        --tag "$AXES_TAG_STIM" \
        --thr "$QC_THR" --k "$QC_K" \
        --pt_min_ms_stim "$PT_MIN" \
        --norm auto
  fi

  run_step "STIM: flow feature=C perm-within=none (save null)" \
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
      --tag "$FLOW_TAG_BASE_STIM_C" \
      --flow_tag_base "$FLOW_TAG_BASE_STIM_C" \
      --save_null_samples \
      --norm auto

  run_step "STIM: flow feature=R perm-within=none (save null)" \
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
      --tag "$FLOW_TAG_BASE_STIM_R" \
      --flow_tag_base "$FLOW_TAG_BASE_STIM_R" \
      --save_null_samples \
      --norm auto

else
  echo "[WARN] No stim caches for SID=$SID → skipping stim"
fi

# ======================================================================================
# SACC 10ms (S): reuse existing axes → flow with perm-within "none"
# ======================================================================================
if [[ -d "$OUT_ROOT/sacc/$SID/caches" ]]; then

  AXES_PATH="$OUT_ROOT/sacc/$SID/axes/$AXES_TAG_SACC"
  if [[ -d "$AXES_PATH" ]]; then
    echo "[info] SACC 10ms axes already exist at $AXES_PATH, reusing"
  else
    run_step "SACC 10ms: train axes (S) peakbin" \
      python cli/train_axes.py \
        --out_root "$OUT_ROOT" \
        --align sacc --sid "$SID" \
        --orientation "$ORI_SACC" \
        --features S \
        --tag "$AXES_TAG_SACC" \
        --norm "$NORM" \
        --clf_binary "$CLF" \
        --C_grid "${CGRID_ARGS[@]}" \
        --lda_shrinkage "$LDA_SHRINKAGE" \
        --searchS \
        --search_range_S="$SEARCH_RANGE_SACC_S" \
        --search_step_ms "$SEARCH_STEP_MS" \
        --search_len_ms "${SEARCH_LEN_ARGS[@]}" \
        --search_score_mode "$SEARCH_SCORE_MODE" \
        --search_tiebreak "$SACC_TIEBREAK" \
        --search_tol "$SEARCH_TOL" \
        --pt_min_ms_sacc "$PT_MIN" \
        --rebin_factor "$REBIN_FACTOR"

    run_step "SACC 10ms: QC" \
      python cli/qc_axes.py \
        --out_root "$OUT_ROOT" \
        --align sacc --sid "$SID" \
        --orientation "$ORI_SACC" \
        --tag "$AXES_TAG_SACC" \
        --thr "$QC_THR" --k "$QC_K" \
        --pt_min_ms_sacc "$PT_MIN" \
        --norm auto \
        --rebin_factor "$REBIN_FACTOR"
  fi

  run_step "SACC 10ms: flow feature=S lag=${LAGS_SACC_MS}ms perm-within=none (save null)" \
    python cli/flow_session.py \
      --out_root "$OUT_ROOT" \
      --align sacc --sid "$SID" \
      --feature S \
      --orientation "$ORI_SACC" \
      --lags_ms "$LAGS_SACC_MS" \
      --ridge "$RIDGE" \
      --perms "$PERMS" \
      --perm-within "$PERM_WITHIN" \
      --null_method trial_shuffle \
      --standardize_mode none \
      --pt_min_ms "$PT_MIN" \
      --no_induced \
      --evoked_subtract --evoked_sigma_ms "$EVOKED_SIGMA_MS" \
      --axes_tag "$AXES_TAG_SACC" \
      --tag "$FLOW_TAG_BASE_SACC_S" \
      --flow_tag_base "$FLOW_TAG_BASE_SACC_S" \
      --save_null_samples \
      --norm auto \
      --rebin_factor "$REBIN_FACTOR"

else
  echo "[WARN] No sacc caches for SID=$SID → skipping sacc"
fi

echo ""
echo "[done] STIM + SACC 10ms peakbin flow (perm-within=none) complete for SID=$SID"
