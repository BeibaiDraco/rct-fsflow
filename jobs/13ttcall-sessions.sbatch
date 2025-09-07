#!/bin/bash
#SBATCH --job-name=rct-ttc-all
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-22%4   # <-- set to 0-(N-1) per your session count

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
  sleep 5
fi

# TTC parameters
ALPHA=0.01
SMOOTH=3
MINCON=2
TMIN=0.00
TMAX=0.50
TAG=ttc_v1_dirEx
TARGET_AREAS=("MFEF" "MLIP" "MSC")

python 13_run_ttc_all_sessions.py \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --root_session results/session \
  --worklist results/worklists/all_pairs.json \
  --target_areas "${TARGET_AREAS[@]}" \
  --alpha "${ALPHA}" --smoothing_bins "${SMOOTH}" --min_consecutive "${MINCON}" \
  --tmin "${TMIN}" --tmax "${TMAX}" --exclude_test_direction \
  --tag "${TAG}" --skip_if_exists
