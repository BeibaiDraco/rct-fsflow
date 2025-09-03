#!/bin/bash
#SBATCH --job-name=rct-axesqc
#SBATCH --output=logs/08fsp_%j.out
#SBATCH --error=logs/08fsp_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd $BASE
module purge
module load python/3.11.9
source $BASE/runtime/venv/bin/activate

# Run two key pairs for your session (vertical-only cache already used upstream)
python 08p_fsflow_timesliding_parallel.py \
  --sid 20200926 \
  --all_pairs \
  --win 0.12 --step 0.02 --k 4 \
  --perms 500 --n_jobs 16 --ridge 1e-2