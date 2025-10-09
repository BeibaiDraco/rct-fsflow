#!/bin/bash
#SBATCH --job-name=pairdiff
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Set this to your session count (e.g., 23)
#SBATCH --array=1-23
#SBATCH -o logs/18_pairdiff_%A_%a.out
#SBATCH -e logs/18_pairdiff_%A_%a.err

set -euo pipefail

# --- Paths & knobs ---
REPO="/project/bdoiron/dracoxu/rct-fsflow"
SID_LIST="$REPO/results/sid_list.txt"   # newline-delimited list of 8-digit SIDs
TAGS="${TAGS:-induced_k2_win016_p500 induced_k5_win016_p500}"  # override via --export=ALL,TAGS="..."
ALPHA="${ALPHA:-0.05}"
REDO="${REDO:-0}"  # set to 1 to force recompute

# --- Env ---
module purge
module load python/3.11.9
source "$REPO/runtime/venv/bin/activate"

# Thread hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"
mkdir -p logs

# Build sid_list.txt if missing
if [[ ! -f "$SID_LIST" ]]; then
  ls -1 results/session | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

# Pick the SID for this array index (1-based)
SID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SID_LIST")"
if [[ -z "${SID:-}" ]]; then
  echo "[WARN] No SID for task $SLURM_ARRAY_TASK_ID in $SID_LIST" >&2
  exit 0
fi
echo "[info] task=$SLURM_ARRAY_TASK_ID SID=$SID"
echo "[info] TAGS: $TAGS"

# Split TAGS string into an array so each tag becomes its own CLI arg
read -r -a TAG_ARR <<< "$TAGS"

CMD=( "$REPO/18_run_pairdiff_all_sessions.py"
      --sid-list-file "$SID_LIST"
      --tags "${TAG_ARR[@]}"
      --idx "$SLURM_ARRAY_TASK_ID" --one-based
      --alpha "$ALPHA"
)
if [[ "$REDO" == "1" ]]; then CMD+=( --redo ); fi

echo "[run] python ${CMD[*]}"
python "${CMD[@]}"
