#!/bin/bash
#SBATCH --job-name=rct-ind-all
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
# EDIT THIS RANGE to match your worklist length (0..N-1), and cap concurrency with %K
#SBATCH --array=0-22%2

set -euo pipefail

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# ----------------------------------------------------------------------
# Ensure worklist exists once (ok if it already exists)
# If you change RCT/* content, delete results/worklists/all_pairs.json and resubmit.
# ----------------------------------------------------------------------
if [[ ! -f results/worklists/all_pairs.json ]]; then
  # Let array index 0 build it, others wait briefly
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    python 02a_list_all_pairs.py --root RCT --out results/worklists/all_pairs.json
  fi
  sleep 5
fi

# ------------------ Tunable parameters (match your 12/13 scripts) ------------------
PERMS=500
N_JOBS=${SLURM_CPUS_PER_TASK}
WIN=0.16
STEP=0.02
K=4
RIDGE=1e-2

# Pre-registered bands
BANDC_START=0.12
BANDC_END=0.28
BANDR_START=0.08
BANDR_END=0.20

# Sliding integration window (rectangular)
INT_WIN=0.16

# Evoked smoothing sigma (ms) for evoked removal
EVOKED_SIGMA_MS=10

# Results go under results/session/<sid>/<TAG>/
TAG=induced_k${K}_win${WIN//./}_p${PERMS}

# Feature toggles (12/13 forward these to 12_induced_fsflow_timesliding.py)
NO_X0=--no_x0             # lagged-only (recommended)
COND_B_ONLY=--cond_b_only # condition on B-only (recommended)
PT_COV=                   # set to "--pt_cov" to enable robust PT covariate
ANNOTATE_P=               # set to "--annotate_p" to print per-time p's on INT plot
SKIP=--skip_if_exists     # skip if outputs already exist for this tag
# -----------------------------------------------------------------------------------

python 13_run_all_sessions_induced.py \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --root RCT \
  --perms "${PERMS}" --n_jobs "${N_JOBS}" \
  --win "${WIN}" --step "${STEP}" --k "${K}" --ridge "${RIDGE}" \
  --bandC_start "${BANDC_START}" --bandC_end "${BANDC_END}" \
  --bandR_start "${BANDR_START}" --bandR_end "${BANDR_END}" \
  --int_win "${INT_WIN}" --evoked_sigma_ms "${EVOKED_SIGMA_MS}" \
  --run_tag "${TAG}" ${SKIP} ${ANNOTATE_P} ${NO_X0} ${COND_B_ONLY} ${PT_COV}
