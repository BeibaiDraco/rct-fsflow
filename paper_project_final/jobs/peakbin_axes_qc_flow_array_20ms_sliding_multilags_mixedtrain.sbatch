#!/bin/bash
#SBATCH --job-name=peakbin_20ms_mixedtrain
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Adjust 1-23 to number of sessions; %32 limits concurrency
#SBATCH --array=1-23%32
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/peakbin_20ms_mixedtrain_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/peakbin_20ms_mixedtrain_%A_%a.err

# =============================================================================
# 20ms SLIDING WINDOW WORKFLOW - MIXED TRAINING
#
# Purpose: Train axes on UNFILTERED data (all trials, lab_is_correct=True for all),
#          but run QC and flow on CORRECT-ONLY caches.
#
# This tests: "Does contaminated training (with incorrect trials) affect
#             the results when evaluated on clean correct-only data?"
#
# Cache parameters MATCH OLD exactly:
#   STIM: t0=-0.25, t1=0.80, bin_ms=10
#   SACC: t0=-0.40, t1=0.30, bin_ms=5
#
# WORKFLOW:
#   1. Build unfiltered caches in out_nofilter/ (with --force_all_correct)
#   2. Train axes on out_nofilter/ caches (ALL trials)
#   3. Copy trained axes to out/ with _mixedtrain tag
#   4. Run QC on out/ caches (CORRECT-ONLY) using mixed-trained axes
#   5. Run flow on out/ caches (CORRECT-ONLY) using mixed-trained axes
#
# Usage:
#   sbatch jobs/peakbin_axes_qc_flow_array_20ms_sliding_multilags_mixedtrain.sbatch
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
OUT_ROOT="$PROJECT/out"              # <-- CORRECT-ONLY caches for QC/flow
OUT_NOFILTER="$PROJECT/out_nofilter" # <-- UNFILTERED caches for axes training
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
mkdir -p "$OUT_NOFILTER"

# Allow local dry-run (no SLURM env)
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# Build sid_list.txt if missing
if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$OUT_ROOT/stim" | grep -E '^[0-9]{8}$' > "$SID_LIST" 2>/dev/null || \
  ls -1 "$PAPER_DATA" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

SID="$(sed -n "${TASK_ID}p" "$SID_LIST" || true)"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $TASK_ID in $SID_LIST" >&2
  exit 0
fi

echo "[info] task=${TASK_ID} SID=${SID}"
echo "[info] PAPER_HOME=$PAPER_HOME"
echo "[info] OUT_ROOT=$OUT_ROOT (CORRECT-ONLY for QC/flow)"
echo "[info] OUT_NOFILTER=$OUT_NOFILTER (UNFILTERED for axes training)"
echo "[info] DRY_RUN=$DRY_RUN CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"

# ------------ Shared analysis knobs (MATCHING OLD) ------------
ORI_STIM="vertical"
ORI_SACC="horizontal"
PT_MIN=200
QC_THR=0.60
QC_K=5

PERMS=500
RIDGE="1e-2"
PERM_WITHIN="CR"

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
# Training tag (used in out_nofilter/)
AXES_TAG_TRAIN_STIM="axes_peakbin_stimCR-stim-${ORI_STIM}-20mssw_nofilter"
AXES_TAG_TRAIN_SACC="axes_peakbin_saccS-sacc-${ORI_SACC}-20mssw_nofilter"

# Evaluation tag (used in out/ for QC/flow) - suffix indicates mixed training
AXES_TAG_EVAL_STIM="axes_peakbin_stimCR-stim-${ORI_STIM}-20mssw_mixedtrain"
AXES_TAG_EVAL_SACC="axes_peakbin_saccS-sacc-${ORI_SACC}-20mssw_mixedtrain"

# ------------ LAG VALUES TO TEST ------------
STIM_LAGS=(30 40 50 60 80 100)
SACC_LAGS=(30 50 60 80)

# Helper: expand arrays into args
SEARCH_LEN_ARGS=()
for L in "${SEARCH_LEN_MS[@]}"; do SEARCH_LEN_ARGS+=("$L"); done

CGRID_ARGS=()
for C in "${C_GRID[@]}"; do CGRID_ARGS+=("$C"); done

# ======================================================================================
# STEP 0: Build caches
# UNFILTERED caches (for axes training): --no_correct_filter --force_all_correct
# CORRECT-ONLY caches (for QC/flow): default behavior
# Cache parameters MATCH OLD exactly.
# ======================================================================================
echo ""
echo "==================== BUILD CACHES ===================="
echo "STIM: t0=-0.25, t1=0.80, bin_ms=10 (matching OLD)"
echo "SACC: t0=-0.40, t1=0.30, bin_ms=5 (matching OLD)"

# Build UNFILTERED STIM caches (for axes training)
run_step "Build STIM caches (unfiltered, for axes training)" \
  python cli/build_cache.py \
    --root "$PAPER_DATA" \
    --out_root "$OUT_NOFILTER" \
    --align stim \
    --sid "$SID" \
    --stim_t0 -0.25 \
    --stim_bin_ms 10 \
    --no_correct_filter \
    --force_all_correct

# Build UNFILTERED SACC caches (for axes training)
run_step "Build SACC caches (unfiltered, for axes training)" \
  python cli/build_cache.py \
    --root "$PAPER_DATA" \
    --out_root "$OUT_NOFILTER" \
    --align sacc \
    --sid "$SID" \
    --sacc_t1 0.30 \
    --no_correct_filter \
    --force_all_correct

# Build CORRECT-ONLY STIM caches (for QC/flow)
run_step "Build STIM caches (correct only, for QC/flow)" \
  python cli/build_cache.py \
    --root "$PAPER_DATA" \
    --out_root "$OUT_ROOT" \
    --align stim \
    --sid "$SID" \
    --stim_t0 -0.25 \
    --stim_bin_ms 10

# Build CORRECT-ONLY SACC caches (for QC/flow)
run_step "Build SACC caches (correct only, for QC/flow)" \
  python cli/build_cache.py \
    --root "$PAPER_DATA" \
    --out_root "$OUT_ROOT" \
    --align sacc \
    --sid "$SID" \
    --sacc_t1 0.30

# ======================================================================================
# STIM: train axes on UNFILTERED → copy to OUT → QC/flow on CORRECT-ONLY
# ======================================================================================
if [[ -d "$OUT_NOFILTER/stim/$SID/caches" ]] && [[ -d "$OUT_ROOT/stim/$SID/caches" ]]; then

  # --- Train axes on unfiltered data ---
  run_step "STIM 20ms sliding: train axes (C,R) on UNFILTERED data" \
    python cli/train_axes.py \
      --out_root "$OUT_NOFILTER" \
      --align stim --sid "$SID" \
      --orientation "$ORI_STIM" \
      --features C R \
      --tag "$AXES_TAG_TRAIN_STIM" \
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

  # --- Copy axes to out/ with _mixedtrain tag ---
  SRC_AXES_STIM="$OUT_NOFILTER/stim/$SID/axes/$AXES_TAG_TRAIN_STIM"
  DST_AXES_STIM="$OUT_ROOT/stim/$SID/axes/$AXES_TAG_EVAL_STIM"
  
  if [[ -d "$SRC_AXES_STIM" ]]; then
    echo ""
    echo "========== Copy STIM axes to out/ with _mixedtrain tag =========="
    mkdir -p "$DST_AXES_STIM"
    cp -v "$SRC_AXES_STIM"/*.npz "$DST_AXES_STIM/" 2>/dev/null || true
    cp -v "$SRC_AXES_STIM"/*.json "$DST_AXES_STIM/" 2>/dev/null || true
    echo "[info] Copied axes from $SRC_AXES_STIM to $DST_AXES_STIM"
  fi

  # --- Run QC on CORRECT-ONLY caches using mixed-trained axes ---
  run_step "STIM 20ms sliding: QC on CORRECT-ONLY (with MIXED-TRAINED axes)" \
    python cli/qc_axes.py \
      --out_root "$OUT_ROOT" \
      --align stim --sid "$SID" \
      --orientation "$ORI_STIM" \
      --tag "$AXES_TAG_EVAL_STIM" \
      --thr "$QC_THR" --k "$QC_K" \
      --pt_min_ms_stim "$PT_MIN" \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  # --- Flow for each lag ---
  for LAG in "${STIM_LAGS[@]}"; do
    FLOW_TAG_STIM_C="evoked_peakbin_stimC_${ORI_STIM}_lag${LAG}ms_20mssw_mixedtrain"
    
    echo ""
    echo "==================== STIM C: LAG ${LAG}ms [MIXEDTRAIN] ===================="
    
    run_step "STIM lag${LAG}ms: flow feature=C on CORRECT-ONLY (mixed-trained axes)" \
      python cli/flow_session.py \
        --out_root "$OUT_ROOT" \
        --align stim --sid "$SID" \
        --feature C \
        --orientation "$ORI_STIM" \
        --lags_ms "$LAG" \
        --ridge "$RIDGE" \
        --perms "$PERMS" \
        --perm-within "$PERM_WITHIN" \
        --null_method trial_shuffle \
        --standardize_mode none \
        --pt_min_ms "$PT_MIN" \
        --no_induced \
        --evoked_subtract --evoked_sigma_ms "$EVOKED_SIGMA_MS" \
        --axes_tag "$AXES_TAG_EVAL_STIM" \
        --tag "$FLOW_TAG_STIM_C" \
        --flow_tag_base "$FLOW_TAG_STIM_C" \
        --save_null_samples \
        --norm auto \
        --sliding_window_ms "$WINDOW_MS" \
        --sliding_step_ms "$STEP_MS"
  done

else
  echo "[WARN] Missing caches for STIM SID=$SID → skipping stim"
fi

# ======================================================================================
# SACC: train axes on UNFILTERED → copy to OUT → QC/flow on CORRECT-ONLY
# ======================================================================================
if [[ -d "$OUT_NOFILTER/sacc/$SID/caches" ]] && [[ -d "$OUT_ROOT/sacc/$SID/caches" ]]; then

  # --- Train axes on unfiltered data ---
  run_step "SACC 20ms sliding: train axes (S) on UNFILTERED data" \
    python cli/train_axes.py \
      --out_root "$OUT_NOFILTER" \
      --align sacc --sid "$SID" \
      --orientation "$ORI_SACC" \
      --features S \
      --tag "$AXES_TAG_TRAIN_SACC" \
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
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  # --- Copy axes to out/ with _mixedtrain tag ---
  SRC_AXES_SACC="$OUT_NOFILTER/sacc/$SID/axes/$AXES_TAG_TRAIN_SACC"
  DST_AXES_SACC="$OUT_ROOT/sacc/$SID/axes/$AXES_TAG_EVAL_SACC"
  
  if [[ -d "$SRC_AXES_SACC" ]]; then
    echo ""
    echo "========== Copy SACC axes to out/ with _mixedtrain tag =========="
    mkdir -p "$DST_AXES_SACC"
    cp -v "$SRC_AXES_SACC"/*.npz "$DST_AXES_SACC/" 2>/dev/null || true
    cp -v "$SRC_AXES_SACC"/*.json "$DST_AXES_SACC/" 2>/dev/null || true
    echo "[info] Copied axes from $SRC_AXES_SACC to $DST_AXES_SACC"
  fi

  # --- Run QC on CORRECT-ONLY caches using mixed-trained axes ---
  run_step "SACC 20ms sliding: QC on CORRECT-ONLY (with MIXED-TRAINED axes)" \
    python cli/qc_axes.py \
      --out_root "$OUT_ROOT" \
      --align sacc --sid "$SID" \
      --orientation "$ORI_SACC" \
      --tag "$AXES_TAG_EVAL_SACC" \
      --thr "$QC_THR" --k "$QC_K" \
      --pt_min_ms_sacc "$PT_MIN" \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"

  # --- Flow for each lag ---
  for LAG in "${SACC_LAGS[@]}"; do
    FLOW_TAG_SACC_S="evoked_peakbin_saccS_${ORI_SACC}_lag${LAG}ms_20mssw_mixedtrain"
    
    echo ""
    echo "==================== SACC S: LAG ${LAG}ms [MIXEDTRAIN] ===================="
    
    run_step "SACC lag${LAG}ms: flow feature=S on CORRECT-ONLY (mixed-trained axes)" \
      python cli/flow_session.py \
        --out_root "$OUT_ROOT" \
        --align sacc --sid "$SID" \
        --feature S \
        --orientation "$ORI_SACC" \
        --lags_ms "$LAG" \
        --ridge "$RIDGE" \
        --perms "$PERMS" \
        --perm-within "$PERM_WITHIN" \
        --null_method trial_shuffle \
        --standardize_mode none \
        --pt_min_ms "$PT_MIN" \
        --no_induced \
        --evoked_subtract --evoked_sigma_ms "$EVOKED_SIGMA_MS" \
        --axes_tag "$AXES_TAG_EVAL_SACC" \
        --tag "$FLOW_TAG_SACC_S" \
        --flow_tag_base "$FLOW_TAG_SACC_S" \
        --save_null_samples \
        --norm auto \
        --sliding_window_ms "$WINDOW_MS" \
        --sliding_step_ms "$STEP_MS"
  done

else
  echo "[WARN] Missing caches for SACC SID=$SID → skipping sacc"
fi

echo ""
echo "[done] 20ms sliding window peakbin MULTI-LAG (MIXED TRAINING) complete for SID=$SID"
echo "============================================================"
