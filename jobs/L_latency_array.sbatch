#!/bin/bash
#SBATCH --job-name=rct-lat
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --array=0-22%2     # set 0-<N-1>, cap concurrency at 4

set -euo pipefail

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# Ensure worklist exists
if [[ ! -f results/worklists/all_pairs.json ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    python 02a_list_all_pairs.py --root RCT --out results/worklists/all_pairs.json
  fi
  sleep 5
fi

# Axes tag you want to use/read (optional)
AXES_TAG=win160_k5_perm500
OUT_TAG=latency

python L02_run_all_trial_latency.py \
  --index ${SLURM_ARRAY_TASK_ID} \
  --root RCT \
  --axes_tag ${AXES_TAG} \
  --out_tag ${OUT_TAG} \
  --thrC 0.75 --thrR 0.60 --hold_m 3 --tmin 0.0 --smooth_bins 3
