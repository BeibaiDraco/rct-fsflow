#!/bin/bash
# Local version of summarize_flow_across_sessions.sbatch
# Run this script from the paper_project_final directory

set -euo pipefail

# ------------ Paths & env ------------
# Adjust these paths for your local setup
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="$PROJECT/out"

# Activate your Python environment
# Option 1: If you have a venv in the parent directory
# BASE="$(dirname "$PROJECT")"
# source "$BASE/runtime/venv/bin/activate"

# Option 2: If you have a conda environment
# conda activate your_env_name

# Option 3: If using system Python or another venv, activate it here
# source /path/to/your/venv/bin/activate

# Thread hygiene (optional but recommended)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# paper_project_final canonical paths
export PAPER_HOME="$PROJECT"
# Adjust PAPER_DATA to your local data path if needed
# export PAPER_DATA="/path/to/your/local/data/RCT_02"
# Safely prepend PAPER_HOME even if PYTHONPATH is unset
export PYTHONPATH="$PAPER_HOME${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
mkdir -p logs

echo "[info] PAPER_HOME=$PAPER_HOME"
echo "[info] OUT_ROOT=$OUT_ROOT"

# ------------ Run summary across sessions ------------
# Use --align all to include stim, sacc, and targ alignments
python cli/summarize_flow_across_sessions.py \
  --out_root "$OUT_ROOT" \
  --align all \
  --alpha 0.05 \
  --qc_threshold 0.75 \
  --group_diff_p \
  --group_null_B 4096 \
  --group_null_seed 12345 \
  --smooth_ms 50
