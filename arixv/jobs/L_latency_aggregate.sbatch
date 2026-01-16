#!/bin/bash
#SBATCH --job-name=rct-lat-agg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

OUT_TAG=latency_auto
GROUP_DIR=results/group_latency_auto

python L03_aggregate_latency.py --sess_root results/session --out_tag ${OUT_TAG} --out_dir ${GROUP_DIR}
