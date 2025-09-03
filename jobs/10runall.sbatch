#!/bin/bash
#SBATCH --job-name=rct-all-sessions
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # match --n_jobs
#SBATCH --mem=128G
#SBATCH --array=0-22%4            # cap: run at most 3 sessions concurrently

set -euo pipefail

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# If worklist doesn't exist, create it once on array index 0 (others will see it)
if [[ ! -f results/worklists/all_pairs.json ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    python 02a_list_all_pairs.py --root RCT --out results/worklists/all_pairs.json
  fi
  # small barrier: give index 0 a head start to write the file
  sleep 5
fi

# Flow parameters (tune here if desired)
PERMS=500
N_JOBS=${SLURM_CPUS_PER_TASK}   # tie to cpus-per-task
WIN=0.16
STEP=0.02
K=5

TAG=win160_k5_perm500

python 10_run_all_sessions.py \
  --index ${SLURM_ARRAY_TASK_ID} \
  --root RCT \
  --perms ${PERMS} \
  --n_jobs ${N_JOBS} \
  --win ${WIN} \
  --step ${STEP} \
  --k ${K} \
  --run_tag ${TAG} --skip_if_exists