#!/bin/bash
#SBATCH --job-name=rct-ind-one
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

set -euo pipefail

# REQUIRED: pass the session id on submit:  sbatch --export=SID=20200926 jobs/12_induced_one.sbatch
SID=20200926

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

# ------------------ Tunable parameters ------------------
PERMS=300
N_JOBS=${SLURM_CPUS_PER_TASK}
WIN=0.16
STEP=0.02
K=2
RIDGE=1e-2

BANDC_START=0.12
BANDC_END=0.28
BANDR_START=0.08
BANDR_END=0.20

INT_WIN=0.16
EVOKED_SIGMA_MS=10

TAG=induced_k${K}_win${WIN//./}_p${PERMS}_test

# Feature toggles
NO_X0=--no_x0
COND_B_ONLY=--cond_b_only
PT_COV=                 # set to "--pt_cov" to enable robust PT covariate
ANNOTATE_P=             # set to "--annotate_p" to print per-time p's on INT plot
SKIP=                   # leave empty for a test; or set to "--skip_if_exists"
# --------------------------------------------------------

# (Optional) refresh cache/axes for this SID exactly like 13_* does:
python 03_cache_binned.py --root RCT --session "${SID}" \
  --bin 0.010 --t0 -0.25 --t1 0.80 --targets_vert_only --reuse_cache

python 04_build_axes.py --sid "${SID}" \
  --trainC_start 0.10 --trainC_end 0.30 \
  --trainR_start 0.05 --trainR_end 0.20 \
  --C_dim 1 --R_dim 2 \
  --out_tag "${TAG}"

# Run 12 on ALL ordered pairs in this session
python 12_induced_fsflow_timesliding.py \
  --sid "${SID}" --all_pairs \
  --win "${WIN}" --step "${STEP}" --k "${K}" --ridge "${RIDGE}" \
  --perms "${PERMS}" --n_jobs "${N_JOBS}" \
  --bandC_start "${BANDC_START}" --bandC_end "${BANDC_END}" \
  --bandR_start "${BANDR_START}" --bandR_end "${BANDR_END}" \
  --int_win "${INT_WIN}" --evoked_sigma_ms "${EVOKED_SIGMA_MS}" \
  --out_tag "${TAG}" ${SKIP} ${ANNOTATE_P} ${NO_X0} ${COND_B_ONLY} ${PT_COV}
