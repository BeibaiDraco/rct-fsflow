#!/bin/bash
#SBATCH --job-name=qc_targ_axes
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Adjust 1-23 to number of sessions; %32 limits concurrency
#SBATCH --array=1-23%32
#SBATCH -o logs/qc_targ_axes_%A_%a.out
#SBATCH -e logs/qc_targ_axes_%A_%a.err

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
# Safely prepend PAPER_HOME even if PYTHONPATH is unset
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

# Build sid_list.txt if missing (from targ directory)
if [[ ! -f "$SID_LIST" ]]; then
  ls -1 "$OUT_ROOT/targ" 2>/dev/null | grep -E '^[0-9]{8}$' > "$SID_LIST" || true
  # Fallback to stim directory if targ is empty
  if [[ ! -s "$SID_LIST" ]]; then
    ls -1 "$OUT_ROOT/stim" | grep -E '^[0-9]{8}$' > "$SID_LIST"
  fi
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

# ------------ Run QC for targ-aligned T axes (all orientations) ------------
# The T axes for targ alignment were saved with axes_tag_base=axes_sweep, so tags are:
#   axes_sweep-targ-vertical
#   axes_sweep-targ-horizontal
#   axes_sweep-targ-pooled

echo ""
echo "========== QC: T axes (targ-aligned, vertical) =========="
python cli/qc_axes.py \
  --out_root "$OUT_ROOT" \
  --align targ \
  --sid "$SID" \
  --orientation vertical \
  --tag axes_sweep-targ-vertical \
  --pt_min_ms_targ 200

echo ""
echo "========== QC: T axes (targ-aligned, horizontal) =========="
python cli/qc_axes.py \
  --out_root "$OUT_ROOT" \
  --align targ \
  --sid "$SID" \
  --orientation horizontal \
  --tag axes_sweep-targ-horizontal \
  --pt_min_ms_targ 200

echo ""
echo "========== QC: T axes (targ-aligned, pooled) =========="
python cli/qc_axes.py \
  --out_root "$OUT_ROOT" \
  --align targ \
  --sid "$SID" \
  --orientation pooled \
  --tag axes_sweep-targ-pooled \
  --pt_min_ms_targ 200

echo ""
echo "[done] targ-aligned T axes QC complete for SID=$SID (vertical+horizontal+pooled)"




