#!/bin/bash
#SBATCH --job-name=case_vi_inv
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/case_vi_inv_%j.out
#SBATCH -e /project/bdoiron/dracoxu/rct-fsflow/paper_project_final/logs/case_vi_inv_%j.err

# =============================================================================
# CASE VI INVESTIGATION
#
# This script runs comprehensive analyses to understand why Case vi shows
# unusually high C-S alignment:
#
# 1. Case vi driver analysis: Correlations, comparisons, and cross-orientation matrix
# 2. Reverse case: Stim-C(horiz) vs Sacc-S(vert) - opposite of case vi
# 3. Temporal cross-alignment: Time-resolved C-S alignment during stim period
# 4. Axis projection analysis: Which PCs do C and S axes share
# 5. Target location confound: Check if target location explains alignment
# 6. Comprehensive summary: Aggregate all results
#
# Each analysis is run for BOTH:
#   - correctonly: Uses out/ caches (correct trials only)
#   - alltrials:   Uses out_nofilter/ caches (all trials)
#
# All outputs go to: out/vi/
#
# Usage:
#   # Run for both trial modes (default)
#   sbatch jobs/vi/case_vi_investigation.sbatch
#
#   # Run only for correct-only trials
#   FILTER_MODE=correctonly sbatch jobs/vi/case_vi_investigation.sbatch
#
#   # Run only for all trials
#   FILTER_MODE=alltrials sbatch jobs/vi/case_vi_investigation.sbatch
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
# Add both project root and cli/ to PYTHONPATH for imports
export PYTHONPATH="$PAPER_HOME/cli:$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

# Create session list if needed
if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$PROJECT/out/sacc" | grep -E '^[0-9]{8}$' > "$SID_LIST" 2>/dev/null || \
  ls -1 "$PAPER_DATA" | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

echo "=============================================================="
echo "CASE VI INVESTIGATION"
echo "Filter mode: $FILTER_MODE"
echo "=============================================================="
echo "Start time: $(date)"
echo ""

# =============================================================================
# Define which modes to run
# =============================================================================
MODES_TO_RUN=()
if [[ "$FILTER_MODE" == "both" ]]; then
  MODES_TO_RUN=("correctonly" "alltrials")
elif [[ "$FILTER_MODE" == "correctonly" ]]; then
  MODES_TO_RUN=("correctonly")
elif [[ "$FILTER_MODE" == "alltrials" ]]; then
  MODES_TO_RUN=("alltrials")
else
  echo "[ERROR] Invalid FILTER_MODE: $FILTER_MODE"
  exit 1
fi

# =============================================================================
# Run analyses for each mode
# =============================================================================
for TRIAL_MODE in "${MODES_TO_RUN[@]}"; do
  echo ""
  echo "######################################################################"
  echo "RUNNING ANALYSES FOR: $TRIAL_MODE"
  echo "######################################################################"
  echo ""
  
  # Set paths based on trial mode
  if [[ "$TRIAL_MODE" == "correctonly" ]]; then
    OUT_ROOT="$PROJECT/out"
    SUFFIX="correctonly"
    AXES_SUFFIX="20mssw"
  else
    OUT_ROOT="$PROJECT/out_nofilter"
    SUFFIX="alltrials"
    AXES_SUFFIX="20mssw_nofilter"
  fi
  
  # =============================================================================
  # Clean up old outputs for this mode
  # =============================================================================
  echo "Cleaning up old outputs for $TRIAL_MODE..."
  rm -rf "$PROJECT/out/vi/case_vi_analysis_${TRIAL_MODE}"
  rm -rf "$PROJECT/out/vi/reverse_case_${SUFFIX}"
  rm -rf "$PROJECT/out/vi/temporal_cross_alignment_${TRIAL_MODE}"
  rm -rf "$PROJECT/out/vi/axis_projection_analysis_${TRIAL_MODE}"
  rm -rf "$PROJECT/out/vi/target_location_analysis_${TRIAL_MODE}"
  rm -f "$PROJECT/out/vi/diagnose_case_vi_matrix_${TRIAL_MODE}.json"
  echo "Done cleaning."
  echo ""

  # =============================================================================
  # Analysis 1: Case vi driver analysis (includes cross-orientation matrix)
  # =============================================================================
  echo "=============================================================="
  echo "[1/6] CASE VI DRIVER ANALYSIS ($TRIAL_MODE)"
  echo "=============================================================="
  
  python cli/vi/analyze_case_vi_drivers.py \
      --project "$PROJECT" \
      --trial_mode "$TRIAL_MODE"
  
  echo ""
  echo "[1/6] DONE"
  echo ""

  # =============================================================================
  # Analysis 2: Reverse case - Stim-C(horiz) vs Sacc-S(vert)
  # =============================================================================
  echo "=============================================================="
  echo "[2/6] REVERSE CASE ANALYSIS ($TRIAL_MODE)"
  echo "=============================================================="
  
  python cli/vi/cross_alignment_reverse.py \
      --sid_list "$SID_LIST" \
      --out_root "$OUT_ROOT" \
      --n_perms 1000 \
      --seed 42 \
      --pt_min_ms 200 \
      --tag "reverse_case_${SUFFIX}" \
      --axes_suffix "$AXES_SUFFIX"
  
  echo ""
  echo "[2/6] DONE"
  echo ""

  # =============================================================================
  # Analysis 3: Temporal cross-alignment
  # =============================================================================
  echo "=============================================================="
  echo "[3/6] TEMPORAL CROSS-ALIGNMENT ($TRIAL_MODE)"
  echo "=============================================================="
  
  # Run for Case vi configuration: C(vert-stim) vs S(horiz-sacc)
  python cli/vi/temporal_cross_alignment.py \
      --sid_list "$SID_LIST" \
      --out_root "$OUT_ROOT" \
      --orientation_C "vertical" \
      --orientation_S "horizontal" \
      --axes_tag_S "axes_peakbin_saccCS-sacc-horizontal-${AXES_SUFFIX}" \
      --window_length_ms 50 \
      --step_ms 20 \
      --time_range_start_ms 0 \
      --time_range_end_ms 500 \
      --pt_min_ms 200 \
      --n_perms 500 \
      --seed 42 \
      --tag "case_vi_temporal" \
      --output_subdir "temporal_cross_alignment_${TRIAL_MODE}"
  
  echo ""
  echo "[3/6] DONE"
  echo ""

  # =============================================================================
  # Analysis 4: Axis projection analysis (PC overlap)
  # =============================================================================
  echo "=============================================================="
  echo "[4/6] AXIS PROJECTION ANALYSIS ($TRIAL_MODE)"
  echo "=============================================================="
  
  python cli/vi/analyze_axis_projections.py \
      --all \
      --out_root "$OUT_ROOT" \
      --output_subdir "axis_projection_analysis_${TRIAL_MODE}" \
      --axes_suffix "$AXES_SUFFIX"
  
  echo ""
  echo "[4/6] DONE"
  echo ""

  # =============================================================================
  # Analysis 5: Target location confound
  # =============================================================================
  echo "=============================================================="
  echo "[5/6] TARGET LOCATION CONFOUND ($TRIAL_MODE)"
  echo "=============================================================="
  
  python cli/vi/analyze_target_location_confound.py \
      --all \
      --out_root "$OUT_ROOT" \
      --output_subdir "target_location_analysis_${TRIAL_MODE}" \
      --axes_suffix "$AXES_SUFFIX"
  
  echo ""
  echo "[5/6] DONE"
  echo ""

  # =============================================================================
  # Summary: Aggregate all results
  # =============================================================================
  echo "=============================================================="
  echo "[6/6] COMPREHENSIVE SUMMARY ($TRIAL_MODE)"
  echo "=============================================================="
  
  python cli/vi/summarize_case_vi_investigation.py \
      --project "$PROJECT" \
      --trial_mode "$TRIAL_MODE"
  
  echo ""
  echo "[6/6] DONE"
  echo ""
  
  echo "Results for $TRIAL_MODE saved to:"
  echo "  - $PROJECT/out/vi/case_vi_analysis_${TRIAL_MODE}/"
  echo "  - $PROJECT/out/vi/reverse_case_${SUFFIX}/"
  echo "  - $PROJECT/out/vi/temporal_cross_alignment_${TRIAL_MODE}/"
  echo "  - $PROJECT/out/vi/axis_projection_analysis_${TRIAL_MODE}/"
  echo "  - $PROJECT/out/vi/target_location_analysis_${TRIAL_MODE}/"
  echo ""

done

echo "=============================================================="
echo "ALL ANALYSES COMPLETE"
echo "=============================================================="
echo "End time: $(date)"
