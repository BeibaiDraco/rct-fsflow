#!/bin/bash
#SBATCH --job-name=rct-axes
#SBATCH --output=rct-axes_%j.out
#SBATCH --error=rct-axes_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"
# make sure caches exist for the session first
python 04_build_axes.py --sid 20200926