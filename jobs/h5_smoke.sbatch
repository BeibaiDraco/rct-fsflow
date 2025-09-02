#!/bin/bash
#SBATCH --job-name=rct-smoke
#SBATCH --output=rct-smoke_%j.out
#SBATCH --error=rct-smoke_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G

set -euo pipefail

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# Run smoke test against the exported dataset under $BASE/RCT
python 01_h5_smoke.py --root RCT --do-bin --area FEF