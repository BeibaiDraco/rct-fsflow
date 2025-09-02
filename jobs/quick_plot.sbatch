#!/bin/bash
#SBATCH --job-name=rct-plot
#SBATCH --output=rct-plot_%j.out
#SBATCH --error=rct-plot_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"
python 02_quick_plot.py --root RCT --area FEF --session 20200217 --unit-index 0