#!/bin/bash
#SBATCH --job-name=17-makeplots
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -euo pipefail

mkdir -p logs

module purge && module load python/3.11.9
source /project/bdoiron/dracoxu/rct-fsflow/runtime/venv/bin/activate
cd /project/bdoiron/dracoxu/rct-fsflow

python 17_make_session_overlays_induced.py \
  --tag induced_k2_win016_p500 \
  --mode both --labels C R \
  --annotate_p --p_stride 1

python 17_make_session_overlays_induced.py \
  --tag induced_k5_win016_p500 \
  --mode both --labels C R \
  --annotate_p --p_stride 1
