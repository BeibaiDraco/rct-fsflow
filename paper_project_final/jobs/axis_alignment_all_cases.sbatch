#!/bin/bash
#SBATCH --job-name=align_all
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# Adjust 1-23 to number of sessions; %16 limits concurrency
#SBATCH --array=1-23%16
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_all_%A_%a.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/align_all_%A_%a.err

# =============================================================================
# AXIS ALIGNMENT ANALYSIS - ALL SIX CASES
#
# Runs alignment analysis for ALL six cases defined in the analysis plan:
#
#   Case i:   Sacc-aligned C & S, horizontal trials
#   Case ii:  Sacc-aligned C & S, pooled trials
#   Case iii: Stim-aligned C (vertical) vs Sacc-aligned S (horizontal)
#   Case iv:  Stim-aligned C (pooled) vs Sacc-aligned S (pooled)
#   Case v:   Stim-aligned C (horizontal) vs Sacc-aligned S (horizontal)
#   Case vi:  Sacc-aligned C & S, vertical trials
#
# Each case is run for BOTH:
#   - correctonly: Uses out/ caches (correct trials only)
#   - alltrials:   Uses out_nofilter/ caches (all trials)
#
# Prerequisites:
#   - Sacc-aligned caches in both out/ and out_nofilter/
#   - Stim-aligned caches in both out/ and out_nofilter/
#   - Required axes trained (run train_axes_all_cases.sbatch first):
#     * out/sacc: axes_peakbin_saccCS-sacc-horizontal-20mssw
#     * out/sacc: axes_peakbin_saccCS-sacc-pooled-20mssw
#     * out/sacc: axes_peakbin_saccCS-sacc-vertical-20mssw
#     * out/stim: axes_peakbin_stimC-stim-vertical-20mssw
#     * out/stim: axes_peakbin_stimC-stim-pooled-20mssw
#     * out/stim: axes_peakbin_stimC-stim-horizontal-20mssw
#     * Same for out_nofilter/ with _nofilter suffix
#
# Usage:
#   # Run all cases for both trial modes (default)
#   sbatch jobs/axis_alignment_all_cases.sbatch
#
#   # Run only for correct-only trials
#   FILTER_MODE=correctonly sbatch jobs/axis_alignment_all_cases.sbatch
#
#   # Run only for all trials
#   FILTER_MODE=alltrials sbatch jobs/axis_alignment_all_cases.sbatch
#
#   # Run specific session (task 5)
#   sbatch --array=5 jobs/axis_alignment_all_cases.sbatch
# =============================================================================

set -euo pipefail

# ------------ Filter mode (both, correctonly, or alltrials) ------------
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
echo "AXIS ALIGNMENT ANALYSIS - ALL FOUR CASES"
echo "Session: $SID (task $TASK_ID)"
echo "Filter mode: $FILTER_MODE"
echo "=============================================================="

# ------------ Common Parameters ------------
N_PERMS=1000
SEED=42
QC_THR_C=0.65
QC_THR_S=0.65
PT_MIN=200
SLIDING_WINDOW_MS=20
SLIDING_STEP_MS=10

# =============================================================================
# Windows for Covariance Estimation (Null Construction)
# =============================================================================
# IMPORTANT: These windows should MATCH the search ranges used for axes training
# in train_axes_all_cases.sbatch. The null construction samples random axes from 
# the covariance structure of the neural manifold, and this covariance should be
# computed over the same time periods where the axes were trained.
#
# Design principle: Use ALIGNMENT-BASED windows (same for C and S within alignment)
#   - SACC-aligned: Use [-0.20, 0.05] for ALL features (matches SEARCH_RANGE_SACC)
#   - STIM-aligned: Use [0.00, 0.50] for ALL features (matches SEARCH_RANGE_STIM)
#
# This ensures the null distribution properly reflects "what alignment would we
# expect if axes were random directions on the neural manifold during this epoch?"
# =============================================================================

# SACC-aligned windows (Cases i & ii: both C and S on sacc alignment)
# Both C and S axes were searched within [-0.20, 0.05], so use this for covariance
WIN_SACC_START=-0.20
WIN_SACC_END=0.05

# STIM-aligned windows (Cases iii & iv: C on stim, S on sacc)
# C was searched within [0.00, 0.50] on stim alignment
WIN_STIM_START=0.00
WIN_STIM_END=0.50

# For cross-alignment cases (iii & iv), S is still on sacc alignment
# So S window uses the sacc range: [-0.20, 0.05]

# =============================================================================
# Function to run Case i or ii (same-alignment)
# =============================================================================
run_case_same_align() {
  local CASE_NUM="$1"        # "i" or "ii"
  local ORIENTATION="$2"     # "horizontal" or "pooled"
  local OUT_ROOT="$3"        # out/ or out_nofilter/
  local AXES_TAG="$4"        # axes tag for sacc-aligned
  local TAG="$5"             # output tag
  local LABEL="$6"           # description
  
  if [[ -d "$OUT_ROOT/sacc/$SID/axes/$AXES_TAG" ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[Case $CASE_NUM] $LABEL"
    echo "  Axes: $AXES_TAG"
    echo "  Orientation: $ORIENTATION"
    echo "  OUT_ROOT: $OUT_ROOT"
    echo "  Covariance windows (SACC-aligned): C=[$WIN_SACC_START, $WIN_SACC_END], S=[$WIN_SACC_START, $WIN_SACC_END]"
    echo "------------------------------------------------------------"
    
    # For same-alignment cases (i & ii), both C and S are sacc-aligned
    # Use the SAME window for both (the sacc search range)
    python cli/axis_alignment_analysis.py \
      --sid "$SID" \
      --out_root "$OUT_ROOT" \
      --mode axis_cov \
      --win_C "$WIN_SACC_START" "$WIN_SACC_END" \
      --win_S "$WIN_SACC_START" "$WIN_SACC_END" \
      --axes_tag "$AXES_TAG" \
      --orientation "$ORIENTATION" \
      --pt_min_ms "$PT_MIN" \
      --n_perms "$N_PERMS" \
      --seed "$SEED" \
      --qc_threshold_C "$QC_THR_C" \
      --qc_threshold_S "$QC_THR_S" \
      --sliding_window_ms "$SLIDING_WINDOW_MS" \
      --sliding_step_ms "$SLIDING_STEP_MS" \
      --no_rich_cond \
      --tag "$TAG"
    
    echo "[done] Case $CASE_NUM → $OUT_ROOT/sacc/alignment/$TAG/"
  else
    echo "[WARN] Axes not found: $OUT_ROOT/sacc/$SID/axes/$AXES_TAG → skipping Case $CASE_NUM"
  fi
}

# =============================================================================
# Function to run Case iii or iv (cross-alignment)
# =============================================================================
run_case_cross_align() {
  local CASE_NUM="$1"        # "iii" or "iv"
  local ORI_C="$2"           # "vertical" or "pooled" for C (stim)
  local ORI_S="$3"           # "horizontal" or "pooled" for S (sacc)
  local OUT_ROOT="$4"        # out/ or out_nofilter/
  local AXES_TAG_C="$5"      # axes tag for stim-aligned C
  local AXES_TAG_S="$6"      # axes tag for sacc-aligned S
  local TAG="$7"             # output tag
  local LABEL="$8"           # description
  
  if [[ -d "$OUT_ROOT/stim/$SID/axes/$AXES_TAG_C" ]] && [[ -d "$OUT_ROOT/sacc/$SID/axes/$AXES_TAG_S" ]]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[Case $CASE_NUM] $LABEL"
    echo "  C axes: stim/$AXES_TAG_C (orientation: $ORI_C)"
    echo "  S axes: sacc/$AXES_TAG_S (orientation: $ORI_S)"
    echo "  OUT_ROOT: $OUT_ROOT"
    echo "  Covariance windows: C(stim)=[$WIN_STIM_START, $WIN_STIM_END], S(sacc)=[$WIN_SACC_START, $WIN_SACC_END]"
    echo "------------------------------------------------------------"
    
    # For cross-alignment cases (iii & iv):
    # - C is stim-aligned, use STIM search range for its covariance window
    # - S is sacc-aligned, use SACC search range for its covariance window
    python cli/axis_alignment_cross.py \
      --sid "$SID" \
      --out_root "$OUT_ROOT" \
      --align_C stim \
      --axes_tag_C "$AXES_TAG_C" \
      --orientation_C "$ORI_C" \
      --win_C "$WIN_STIM_START" "$WIN_STIM_END" \
      --align_S sacc \
      --axes_tag_S "$AXES_TAG_S" \
      --orientation_S "$ORI_S" \
      --win_S "$WIN_SACC_START" "$WIN_SACC_END" \
      --pt_min_ms "$PT_MIN" \
      --n_perms "$N_PERMS" \
      --seed "$SEED" \
      --qc_threshold_C "$QC_THR_C" \
      --qc_threshold_S "$QC_THR_S" \
      --no_rich_cond \
      --tag "$TAG"
    
    echo "[done] Case $CASE_NUM → $OUT_ROOT/cross_alignment/$TAG/"
  else
    echo "[WARN] Axes not found for Case $CASE_NUM → skipping"
    [[ ! -d "$OUT_ROOT/stim/$SID/axes/$AXES_TAG_C" ]] && echo "       Missing: $OUT_ROOT/stim/$SID/axes/$AXES_TAG_C"
    [[ ! -d "$OUT_ROOT/sacc/$SID/axes/$AXES_TAG_S" ]] && echo "       Missing: $OUT_ROOT/sacc/$SID/axes/$AXES_TAG_S"
  fi
}

# =============================================================================
# Run all cases for CORRECT-ONLY trials
# =============================================================================
if [[ "$FILTER_MODE" == "both" || "$FILTER_MODE" == "correctonly" ]]; then
  echo ""
  echo "=============================================================="
  echo "CORRECT-ONLY TRIALS (out/)"
  echo "=============================================================="
  
  # Case i: Sacc-aligned C & S, horizontal trials
  run_case_same_align \
    "i" \
    "horizontal" \
    "$PROJECT/out" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw" \
    "case_i_correctonly" \
    "Sacc-aligned C & S, horizontal trials (correct-only)"
  
  # Case ii: Sacc-aligned C & S, pooled trials
  run_case_same_align \
    "ii" \
    "pooled" \
    "$PROJECT/out" \
    "axes_peakbin_saccCS-sacc-pooled-20mssw" \
    "case_ii_correctonly" \
    "Sacc-aligned C & S, pooled trials (correct-only)"
  
  # Case iii: Stim-aligned C (vertical) vs Sacc-aligned S (horizontal)
  run_case_cross_align \
    "iii" \
    "vertical" \
    "horizontal" \
    "$PROJECT/out" \
    "axes_peakbin_stimC-stim-vertical-20mssw" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw" \
    "case_iii_correctonly" \
    "Stim-C (vertical) vs Sacc-S (horizontal) (correct-only)"
  
  # Case iv: Stim-aligned C (pooled) vs Sacc-aligned S (pooled)
  run_case_cross_align \
    "iv" \
    "pooled" \
    "pooled" \
    "$PROJECT/out" \
    "axes_peakbin_stimC-stim-pooled-20mssw" \
    "axes_peakbin_saccCS-sacc-pooled-20mssw" \
    "case_iv_correctonly" \
    "Stim-C (pooled) vs Sacc-S (pooled) (correct-only)"
  
  # Case v: Stim-aligned C (horizontal) vs Sacc-aligned S (horizontal)
  run_case_cross_align \
    "v" \
    "horizontal" \
    "horizontal" \
    "$PROJECT/out" \
    "axes_peakbin_stimC-stim-horizontal-20mssw" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw" \
    "case_v_correctonly" \
    "Stim-C (horizontal) vs Sacc-S (horizontal) (correct-only)"
  
  # Case vi: Sacc-aligned C & S, vertical trials
  run_case_same_align \
    "vi" \
    "vertical" \
    "$PROJECT/out" \
    "axes_peakbin_saccCS-sacc-vertical-20mssw" \
    "case_vi_correctonly" \
    "Sacc-aligned C & S, vertical trials (correct-only)"
fi

# =============================================================================
# Run all cases for ALL TRIALS (nofilter)
# =============================================================================
if [[ "$FILTER_MODE" == "both" || "$FILTER_MODE" == "alltrials" ]]; then
  echo ""
  echo "=============================================================="
  echo "ALL TRIALS (out_nofilter/)"
  echo "=============================================================="
  
  # Case i: Sacc-aligned C & S, horizontal trials
  run_case_same_align \
    "i" \
    "horizontal" \
    "$PROJECT/out_nofilter" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw_nofilter" \
    "case_i_alltrials" \
    "Sacc-aligned C & S, horizontal trials (all-trials)"
  
  # Case ii: Sacc-aligned C & S, pooled trials
  run_case_same_align \
    "ii" \
    "pooled" \
    "$PROJECT/out_nofilter" \
    "axes_peakbin_saccCS-sacc-pooled-20mssw_nofilter" \
    "case_ii_alltrials" \
    "Sacc-aligned C & S, pooled trials (all-trials)"
  
  # Case iii: Stim-aligned C (vertical) vs Sacc-aligned S (horizontal)
  run_case_cross_align \
    "iii" \
    "vertical" \
    "horizontal" \
    "$PROJECT/out_nofilter" \
    "axes_peakbin_stimC-stim-vertical-20mssw_nofilter" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw_nofilter" \
    "case_iii_alltrials" \
    "Stim-C (vertical) vs Sacc-S (horizontal) (all-trials)"
  
  # Case iv: Stim-aligned C (pooled) vs Sacc-aligned S (pooled)
  run_case_cross_align \
    "iv" \
    "pooled" \
    "pooled" \
    "$PROJECT/out_nofilter" \
    "axes_peakbin_stimC-stim-pooled-20mssw_nofilter" \
    "axes_peakbin_saccCS-sacc-pooled-20mssw_nofilter" \
    "case_iv_alltrials" \
    "Stim-C (pooled) vs Sacc-S (pooled) (all-trials)"
  
  # Case v: Stim-aligned C (horizontal) vs Sacc-aligned S (horizontal)
  run_case_cross_align \
    "v" \
    "horizontal" \
    "horizontal" \
    "$PROJECT/out_nofilter" \
    "axes_peakbin_stimC-stim-horizontal-20mssw_nofilter" \
    "axes_peakbin_saccCS-sacc-horizontal-20mssw_nofilter" \
    "case_v_alltrials" \
    "Stim-C (horizontal) vs Sacc-S (horizontal) (all-trials)"
  
  # Case vi: Sacc-aligned C & S, vertical trials
  run_case_same_align \
    "vi" \
    "vertical" \
    "$PROJECT/out_nofilter" \
    "axes_peakbin_saccCS-sacc-vertical-20mssw_nofilter" \
    "case_vi_alltrials" \
    "Sacc-aligned C & S, vertical trials (all-trials)"
fi

echo ""
echo "=============================================================="
echo "[done] All cases complete for SID=$SID"
echo "=============================================================="
