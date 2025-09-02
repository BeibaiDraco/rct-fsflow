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

# Make sure caches & axes already exist for this session & areas
python 05_fsflow.py --sid 20200926 --A MFEF --B MLIP --condition_other
python 05_fsflow.py --sid 20200926 --A MFEF --B MSC --condition_other
python 05_fsflow.py --sid 20200926 --A MLIP --B MFEF --condition_other
python 05_fsflow.py --sid 20200926 --A MSC --B MFEF --condition_other