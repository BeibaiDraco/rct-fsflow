#!/bin/bash
#SBATCH --job-name=rct-all-sessions
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # match --n_jobs
#SBATCH --mem=128G
#SBATCH --array=0-22%4             # run at most 4 sessions concurrently

set -euo pipefail

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

echo "[info] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[info] CPUs per task = ${SLURM_CPUS_PER_TASK}"
echo "[info] Hostname = $(hostname)"

# Ensure worklist exists (task 0 creates it)
if [[ ! -f results/worklists/all_pairs.json ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    python 02a_list_all_pairs.py --root RCT --out results/worklists/all_pairs.json
  fi
  # brief barrier so others see it
  sleep 5
fi

# Flow parameters
PERMS=500
N_JOBS=${SLURM_CPUS_PER_TASK}   # tie to cpus-per-task
WIN=0.16
STEP=0.02
K=5
RIDGE=1e-2

# Tag for outputs (so runs don't overwrite each other)
TAG=win160_k5_perm500

python 10_run_all_sessions_integrated.py \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --root RCT \
  --perms "${PERMS}" --n_jobs "${N_JOBS}" \
  --win "${WIN}" --step "${STEP}" --k "${K}" --ridge "${RIDGE}" \
  --band_start 0.12 --band_end 0.28 \
  --run_tag "${TAG}" --skip_if_exists
