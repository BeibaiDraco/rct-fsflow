#!/bin/bash
#SBATCH --job-name=rct-axesqc
#SBATCH --output=logs/rct-axesqc_%j.out
#SBATCH --error=logs/rct-axesqc_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

# env
set -euo pipefail
BASE=/project/bdoiron/dracoxu/rct-fsflow
cd $BASE
module purge
module load python/3.11.9
source $BASE/runtime/venv/bin/activate

# 1) Cache vertical-only trials, -0.25..+0.80 s, 10 ms bins
python 03_cache_binned.py --root RCT --session 20200926 --bin 0.010 --t0 -0.25 --t1 0.80 --targets_vert_only

python 04_build_axes.py --sid 20200926 \
  --trainC_start 0.10 --trainC_end 0.30 \
  --trainR_start 0.05 --trainR_end 0.20 \
  --C_dim 1 --R_dim 2

# 3) QC (subspace-based curves + pooled CV numbers)
python 06_axes_qc.py --sid 20200926 --area MLIP --smooth_bins 3
python 06_axes_qc.py --sid 20200926 --area MFEF --smooth_bins 3
python 06_axes_qc.py --sid 20200926 --area MSC  --smooth_bins 3