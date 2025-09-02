#!/bin/bash
#SBATCH --job-name=rct-axesqc
#SBATCH --output=logs/rct-raw_%j.out
#SBATCH --error=logs/rct-raw_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd $BASE
module purge
module load python/3.11.9
source $BASE/runtime/venv/bin/activate

# Example: test LIP first (often strongest for within-category direction)
python 07_decode_raw.py --sid 20200926 --area MLIP --pool_start 0.05 --pool_end 0.20 --timeseries

# Then FEF, SC
python 07_decode_raw.py --sid 20200926 --area MFEF --pool_start 0.05 --pool_end 0.20 --timeseries
python 07_decode_raw.py --sid 20200926 --area MSC  --pool_start 0.05 --pool_end 0.20 --timeseries