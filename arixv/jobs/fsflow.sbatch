#!/bin/bash
#SBATCH --job-name=rct-flow
#SBATCH --output=rct-flow_%j.out
#SBATCH --error=rct-flow_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# 1) ensure caches exist (once per session)
python 03_cache_binned.py --root RCT --session 20200217 --areas FEF LIP SC --bin 0.010
# 2) build axes
python 04_build_axes.py --sid 20200217 --areas FEF LIP SC
# 3) run flow (examples)
python 05_fsflow.py --sid 20200217 --pair LIP->FEF --condition_other
python 05_fsflow.py --sid 20200217 --pair FEF->SC  --condition_other
python 05_fsflow.py --sid 20200217 --pair LIP->SC  --condition_other