#!/bin/bash
#SBATCH --job-name=sacc-all
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-22%4
#SBATCH -o logs/41_all_sessions_%j.out
#SBATCH -e logs/41_all_sessions_%j.err

set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow/saccade
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/venv/bin/activate" || source "$BASE/runtime/venv/bin/activate" || true

echo "[info] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[info] Hostname=$(hostname)  CPUs=${SLURM_CPUS_PER_TASK}"

python scripts/41_run_all_sessions_sacc.py \
  --root RCT_02 \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --orientation vertical \
  --t0 -0.40 --t1 0.20 --bin_ms 5 \
  --trainC -0.30:-0.10 --trainS -0.10:-0.03 \
  --lags_ms 30 --perms 500 \
  --tag sacc_v1
