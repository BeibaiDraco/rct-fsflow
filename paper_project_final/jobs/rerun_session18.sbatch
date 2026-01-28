#!/bin/bash
#SBATCH --job-name=rerun_s18
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/rerun_s18_%j.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/rerun_s18_%j.err

# =============================================================================
# Rerun session 18 (SID=20231103) - full pipeline: C, R (stim) and S (sacc)
# Previously failed due to R-feature SVD bug (now fixed in paperflow/axes.py)
# =============================================================================

set -euo pipefail

SID="20231103"

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
OUT_ROOT="$PROJECT/out"

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# Thread hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PAPER_HOME="$PROJECT"
export PAPER_DATA="/project/bdoiron/dracoxu/rct-fsflow/paper_project_final/RCT_02"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"

# ------------ Shared analysis knobs (same as main job) ------------
ORI_STIM="vertical"
ORI_SACC="horizontal"
PT_MIN=200
QC_THR=0.60
QC_K=5
PERMS=500
RIDGE="1e-2"
PERM_WITHIN="CR"
EVOKED_SIGMA_MS=10
WINDOW_MS=20
STEP_MS=10
SEARCH_STEP_MS=20
SEARCH_LEN_MS=(50 100 150 200 250)
SEARCH_SCORE_MODE="peak_bin_auc"
SEARCH_TOL="0.01"
STIM_TIEBREAK="shortest_then_earliest"
SACC_TIEBREAK="shortest_then_closest0"
SEARCH_RANGE_STIM_C="0.00:0.50"
SEARCH_RANGE_SACC_S="-0.20:0.05"
NORM="global"
CLF="logreg"
C_GRID=(0.1 0.3 1 3 10)
LDA_SHRINKAGE="auto"

# Use the SAME tags as main job for consistency
AXES_TAG_STIM="axes_peakbin_stimCR-stim-${ORI_STIM}-20mssw"
AXES_TAG_SACC="axes_peakbin_saccS-sacc-${ORI_SACC}-20mssw"

STIM_LAGS=(30 40 50 60 80 100)
SACC_LAGS=(30 50 60 80)

SEARCH_LEN_ARGS=()
for L in "${SEARCH_LEN_MS[@]}"; do SEARCH_LEN_ARGS+=("$L"); done
CGRID_ARGS=()
for C in "${C_GRID[@]}"; do CGRID_ARGS+=("$C"); done

echo "[info] Rerunning SID=${SID} with C, R (stim) and S (sacc) - bug fixed"
echo "[info] OUT_ROOT=$OUT_ROOT"

# =============================================================================
# STIM: train axes (C and R - now should work with bug fix)
# =============================================================================
if [[ -d "$OUT_ROOT/stim/$SID/caches" ]]; then

  echo "========== STIM: train axes (C,R) peakbin =========="
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
    --search_range_R="$SEARCH_RANGE_STIM_C" \
    --search_step_ms "$SEARCH_STEP_MS" \
    --search_len_ms "${SEARCH_LEN_ARGS[@]}" \
    --search_score_mode "$SEARCH_SCORE_MODE" \
    --search_tiebreak "$STIM_TIEBREAK" \
    --search_tol "$SEARCH_TOL" \
    --pt_min_ms_stim "$PT_MIN" \
    --sliding_window_ms "$WINDOW_MS" \
    --sliding_step_ms "$STEP_MS"

  echo "========== STIM: QC =========="
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

  # Flow for all STIM lags
  for LAG in "${STIM_LAGS[@]}"; do
    FLOW_TAG_STIM_C="evoked_peakbin_stimC_${ORI_STIM}_lag${LAG}ms_20mssw"
    echo "========== STIM C: LAG ${LAG}ms =========="
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
      --axes_tag "$AXES_TAG_STIM" \
      --tag "$FLOW_TAG_STIM_C" \
      --flow_tag_base "$FLOW_TAG_STIM_C" \
      --save_null_samples \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"
  done
else
  echo "[WARN] No stim caches for SID=$SID"
fi

# =============================================================================
# SACC: train axes → QC → flow (S)
# =============================================================================
if [[ -d "$OUT_ROOT/sacc/$SID/caches" ]]; then

  echo "========== SACC: train axes (S) peakbin =========="
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
    --sliding_window_ms "$WINDOW_MS" \
    --sliding_step_ms "$STEP_MS"

  echo "========== SACC: QC =========="
  python cli/qc_axes.py \
    --out_root "$OUT_ROOT" \
    --align sacc --sid "$SID" \
    --orientation "$ORI_SACC" \
    --tag "$AXES_TAG_SACC" \
    --thr "$QC_THR" --k "$QC_K" \
    --pt_min_ms_sacc "$PT_MIN" \
    --norm auto \
    --sliding_window_ms "$WINDOW_MS" \
    --sliding_step_ms "$STEP_MS"

  # Flow for all SACC lags
  for LAG in "${SACC_LAGS[@]}"; do
    FLOW_TAG_SACC_S="evoked_peakbin_saccS_${ORI_SACC}_lag${LAG}ms_20mssw"
    echo "========== SACC S: LAG ${LAG}ms =========="
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
      --axes_tag "$AXES_TAG_SACC" \
      --tag "$FLOW_TAG_SACC_S" \
      --flow_tag_base "$FLOW_TAG_SACC_S" \
      --save_null_samples \
      --norm auto \
      --sliding_window_ms "$WINDOW_MS" \
      --sliding_step_ms "$STEP_MS"
  done
else
  echo "[WARN] No sacc caches for SID=$SID"
fi

echo ""
echo "[done] Rerun complete for SID=$SID"
