#!/bin/bash
#SBATCH --job-name=peakbin_axes_qc_flow
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Adjust 1-23 to number of sessions; %32 limits concurrency
#SBATCH --array=1-23%32
#SBATCH -o logs/peakbin_axes_qc_flow_%A_%a.out
#SBATCH -e logs/peakbin_axes_qc_flow_%A_%a.err

set -euo pipefail

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
export PAPER_DATA="/project/bdoiron/dracoxu/rct-fsflow/saccade/RCT_02"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

# Build sid_list.txt if missing (from stim directory)
if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$OUT_ROOT/stim" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

# Pick SID for this array index (1-based)
SID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SID_LIST" || true)"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $SLURM_ARRAY_TASK_ID in $SID_LIST" >&2
  exit 0
fi

echo "[info] task=$SLURM_ARRAY_TASK_ID SID=$SID"
echo "[info] PAPER_HOME=$PAPER_HOME"
echo "[info] OUT_ROOT=$OUT_ROOT"

# ------------ Shared analysis knobs ------------
ORI="vertical"           # change if you want horizontal/pooled
PT_MIN=200               # ms
QC_THR=0.60              # your session-level filtering used 0.6
QC_K=5                   # consecutive bins for latency

PERMS=500
RIDGE="1e-2"
PERM_WITHIN="CR"
EVOKED_SIGMA_MS=10

# Lags (explicitly included in flow tags)
LAGS_STIM_MS=50
LAGS_SACC_MS=30

# Window search grid (shared)
SEARCH_STEP_MS=20
SEARCH_LEN_MS=(50 100 150 200 250)

# Peak-bin scoring + tie-break
SEARCH_SCORE_MODE="peak_bin_auc"
SEARCH_TOL="0.01"

# Stim: encourage earlier/shorter if within tol
STIM_TIEBREAK="shortest_then_earliest"
# Sacc: encourage window centered near 0 if within tol
SACC_TIEBREAK="shortest_then_closest0"

# Search ranges
SEARCH_RANGE_STIM_C="0.00:0.50"
SEARCH_RANGE_STIM_R="0.00:0.50"
SEARCH_RANGE_SACC_S="-0.20:0.05"

# Normalization / classifier (edit if desired)
NORM="global"          # global|baseline|none  (use baseline if you want)
CLF="logreg"           # logreg|svm|lda
C_GRID=(0.1 0.3 1 3 10)
LDA_SHRINKAGE="auto"

# ------------ Tags (axes/QC + flow) ------------
AXES_TAG_BASE_STIM="axes_peakbin_stimCR"
AXES_TAG_BASE_SACC="axes_peakbin_saccS"

AXES_TAG_STIM="${AXES_TAG_BASE_STIM}-stim-${ORI}"
AXES_TAG_SACC="${AXES_TAG_BASE_SACC}-sacc-${ORI}"

FLOW_TAG_BASE_STIM_C="evoked_peakbin_stimC_${ORI}_lag${LAGS_STIM_MS}ms"
FLOW_TAG_BASE_STIM_R="evoked_peakbin_stimR_${ORI}_lag${LAGS_STIM_MS}ms"
FLOW_TAG_BASE_SACC_S="evoked_peakbin_saccS_${ORI}_lag${LAGS_SACC_MS}ms"

# ------------ Helper: stringify arrays for CLI ------------
SEARCH_LEN_MS_ARGS=()
for L in "${SEARCH_LEN_MS[@]}"; do SEARCH_LEN_MS_ARGS+=("$L"); done

C_GRID_ARGS=()
for C in "${C_GRID[@]}"; do C_GRID_ARGS+=("$C"); done

# ======================================================================================
# STIM: Train axes (C,R) with peak-bin window search, QC, then flow for C and R
# ======================================================================================
if [[ -d "$OUT_ROOT/stim/$SID/caches" ]]; then
  echo ""
  echo "========== STIM: train axes (C,R) peakbin =========="
  python cli/train_axes.py \
    --out_root "$OUT_ROOT" \
    --align stim --sid "$SID" \
    --orientation "$ORI" \
    --features C R \
    --tag "$AXES_TAG_STIM" \
    --norm "$NORM" \
    --clf_binary "$CLF" \
    --C_grid "${C_GRID_ARGS[@]}" \
    --lda_shrinkage "$LDA_SHRINKAGE" \
    --searchC --searchR \
    --search_range_C "$SEARCH_RANGE_STIM_C" \
    --search_range_R "$SEARCH_RANGE_STIM_R" \
    --search_step_ms "$SEARCH_STEP_MS" \
    --search_len_ms "${SEARCH_LEN_MS_ARGS[@]}" \
    --search_score_mode "$SEARCH_SCORE_MODE" \
    --search_tiebreak "$STIM_TIEBREAK" \
    --search_tol "$SEARCH_TOL" \
    --pt_min_ms_stim "$PT_MIN"

  echo ""
  echo "========== STIM: QC =========="
  python cli/qc_axes.py \
    --out_root "$OUT_ROOT" \
    --align stim --sid "$SID" \
    --orientation "$ORI" \
    --tag "$AXES_TAG_STIM" \
    --thr "$QC_THR" --k "$QC_K" \
    --pt_min_ms_stim "$PT_MIN" \
    --norm auto

  echo ""
  echo "========== STIM: flow feature=C (save null) =========="
  python cli/flow_session.py \
    --out_root "$OUT_ROOT" \
    --align stim --sid "$SID" \
    --feature C \
    --orientation "$ORI" \
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

  echo ""
  echo "========== STIM: flow feature=R (save null) =========="
  python cli/flow_session.py \
    --out_root "$OUT_ROOT" \
    --align stim --sid "$SID" \
    --feature R \
    --orientation "$ORI" \
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
# SACC: Train axes (S) with peakbin window search, QC, then flow for S
# ======================================================================================
if [[ -d "$OUT_ROOT/sacc/$SID/caches" ]]; then
  echo ""
  echo "========== SACC: train axes (S) peakbin =========="
  python cli/train_axes.py \
    --out_root "$OUT_ROOT" \
    --align sacc --sid "$SID" \
    --orientation "$ORI" \
    --features S \
    --tag "$AXES_TAG_SACC" \
    --norm "$NORM" \
    --clf_binary "$CLF" \
    --C_grid "${C_GRID_ARGS[@]}" \
    --lda_shrinkage "$LDA_SHRINKAGE" \
    --searchS \
    --search_range_S "$SEARCH_RANGE_SACC_S" \
    --search_step_ms "$SEARCH_STEP_MS" \
    --search_len_ms "${SEARCH_LEN_MS_ARGS[@]}" \
    --search_score_mode "$SEARCH_SCORE_MODE" \
    --search_tiebreak "$SACC_TIEBREAK" \
    --search_tol "$SEARCH_TOL" \
    --pt_min_ms_sacc "$PT_MIN"

  echo ""
  echo "========== SACC: QC =========="
  python cli/qc_axes.py \
    --out_root "$OUT_ROOT" \
    --align sacc --sid "$SID" \
    --orientation "$ORI" \
    --tag "$AXES_TAG_SACC" \
    --thr "$QC_THR" --k "$QC_K" \
    --pt_min_ms_sacc "$PT_MIN" \
    --norm auto

  echo ""
  echo "========== SACC: flow feature=S (save null) =========="
  python cli/flow_session.py \
    --out_root "$OUT_ROOT" \
    --align sacc --sid "$SID" \
    --feature S \
    --orientation "$ORI" \
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
    --norm auto
else
  echo "[WARN] No sacc caches for SID=$SID → skipping sacc"
fi

echo ""
echo "[done] peakbin axes+QC+flow complete for SID=$SID"
