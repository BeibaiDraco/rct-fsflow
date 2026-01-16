#!/bin/bash
#SBATCH --job-name=rct-cache
#SBATCH --output=rct-cache_%j.out
#SBATCH --error=rct-cache_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"
python 02a_list_all_pairs.py --root RCT