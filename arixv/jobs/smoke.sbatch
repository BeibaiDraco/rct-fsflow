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
cd $BASE || { echo "Cannot cd into $BASE"; exit 1; }

module purge
module load python/3.11.9
source $BASE/runtime/venv/bin/activate

which python
python -V
ls -lh 00_smoke_test.py

python 00_smoke_test.py --mat data/raw/RCT_master_dataset_both_monkeys.mat --do_bin --area FEF
