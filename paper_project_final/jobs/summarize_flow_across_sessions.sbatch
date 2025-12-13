#!/bin/bash
#SBATCH --job-name=summ_flow
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/summarize_flow_%j.out
#SBATCH -e logs/summarize_flow_%j.err

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
export PAPER_DATA="/project/bdoiron/dracoxu/rct-fsflow/saccade/RCT_02"
# Safely prepend PAPER_HOME even if PYTHONPATH is unset
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

echo "[info] PAPER_HOME=$PAPER_HOME"
echo "[info] OUT_ROOT=$OUT_ROOT"

# ------------ Run summary across sessions ------------
python cli/summarize_flow_across_sessions.py \
  --out_root "$OUT_ROOT" \
  --align both \
  --alpha 0.05 \
  --qc_threshold 0.75 \
  --group_diff_p \
  --group_null_B 4096 \
  --group_null_seed 12345 \
  --smooth_ms 50

