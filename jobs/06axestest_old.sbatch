#!/bin/bash
#SBATCH --job-name=rct-axesqc
#SBATCH --output=logs/rct-axesqc_%j.out
#SBATCH --error=logs/rct-axesqc_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# rebuild caches @ 10 ms
python 03_cache_binned.py --root RCT --session 20200926 --bin 0.010
# rebuild axes (auto-detect areas)
python 04_build_axes.py --sid 20200926
# QC again
python 06_axes_qc.py --sid 20200926 --area MFEF
python 06_axes_qc.py --sid 20200926 --area MLIP
python 06_axes_qc.py --sid 20200926 --area MSC