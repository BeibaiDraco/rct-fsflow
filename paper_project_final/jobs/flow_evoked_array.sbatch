#!/bin/bash
#SBATCH --job-name=flow_evoked
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Adjust 1-23 to number of sessions; %32 limits concurrency
#SBATCH --array=1-23%32
#SBATCH -o logs/flow_evoked_%A_%a.out
#SBATCH -e logs/flow_evoked_%A_%a.err

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

# Build sid_list.txt if missing (from stim directory; sacc will be subset)
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

# ------------ Run evoked subtraction flow ------------
python cli/run_one_session_evoked.py \
  --out_root "$OUT_ROOT" \
  --sid "$SID" \
  --axes_tag_base axes_sweep \
  --flow_tag_base evoked_subtract_vert \
  --lags_ms_stim 50 \
  --lags_ms_sacc 30 \
  --ridge 1e-2 \
  --perms 500 \
  --pt_min_ms 200 \
  --perm-within CR \
  --evoked_sigma_ms 10

