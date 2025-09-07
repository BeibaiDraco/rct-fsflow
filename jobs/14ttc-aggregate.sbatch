#!/bin/bash
#SBATCH --job-name=rct-ttc-agg
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

TAG=ttc_v1_dirEx

python 14_ttc_aggregate_across_sessions.py \
  --root_session results/session \
  --out_root results/ttc \
  --tag "${TAG}" \
  --stats_script 12_ttc_stats.py
