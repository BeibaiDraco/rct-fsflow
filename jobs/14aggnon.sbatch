#!/bin/bash
#SBATCH --job-name=14-agg-non
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

python 14_aggregate_noninduced.py --tag win160_k2_perm500_integrated --save_csv
# or aggregate a different non-induced tag:
python 14_aggregate_noninduced.py --tag win160_k5_perm500_integrated --save_csv