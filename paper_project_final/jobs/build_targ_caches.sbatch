#!/bin/bash
#SBATCH --job-name=build_targ
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/build_targ_caches_%j.out
#SBATCH -e logs/build_targ_caches_%j.err

set -euo pipefail

# ------------ Paths & env ------------
BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
OUT_ROOT="$PROJECT/out"

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# Thread hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# paper_project_final canonical paths
export PAPER_HOME="$PROJECT"
export PAPER_DATA="/project/bdoiron/dracoxu/rct-fsflow/paper_project_final/RCT_02"
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

echo "[info] Building targ caches for all sessions"
echo "[info] PAPER_HOME=$PAPER_HOME"
echo "[info] PAPER_DATA=$PAPER_DATA"
echo "[info] OUT_ROOT=$OUT_ROOT"

# ------------ Build targ caches for all sessions ------------
# Time window: -200ms to +350ms aligned to targets onset
# Bin size: 5ms
python cli/build_caches_all.py \
  --root "$PAPER_DATA" \
  --out_root "$OUT_ROOT" \
  --align targ \
  --targ_t0 -0.20 \
  --targ_t1 0.35 \
  --targ_bin_ms 5.0 \
  --min_areas 2

echo "[done] targ caches built"

