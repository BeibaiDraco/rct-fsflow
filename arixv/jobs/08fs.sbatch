#!/bin/bash
#SBATCH --job-name=rct-axesqc
#SBATCH --output=logs/08fs_%j.out
#SBATCH --error=logs/08fs_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=02:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd $BASE
module purge
module load python/3.11.9
source $BASE/runtime/venv/bin/activate

# Run two key pairs for your session (vertical-only cache already used upstream)
python 08_fsflow_timesliding.py --sid 20200926 --A MLIP --B MFEF --win 0.12 --step 0.02 --k 4 --perms 20
python 08_fsflow_timesliding.py --sid 20200926 --A MFEF --B MSC  --win 0.12 --step 0.02 --k 4 --perms 20