#!/bin/bash
#SBATCH --job-name=test-ttc-single
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

BASE=/project/bdoiron/dracoxu/rct-fsflow
cd "$BASE"
mkdir -p logs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"

echo "[info] Testing TTC analysis on single session"
echo "[info] Hostname = $(hostname)"
echo "[info] CPUs per task = ${SLURM_CPUS_PER_TASK}"

# Test parameters - modify these as needed
SESSION_DATE="20200926"                    # Actual session date
GC_CONFIG="win160_k4_perm500"              # GC analysis config: win=160ms, k=3, 500perms
SESSION_ID="${SESSION_DATE}/${GC_CONFIG}"  # Full path to subspace data
AREAS=("SFEF" "SLIP")
ALPHA=0.01
SMOOTH=3
MINCON=2
TMIN=0.00
TMAX=0.50
TAG="test_ttc"
MIN_THETA_C=0.55
MIN_THETA_R=0.60

echo "[info] Session date: ${SESSION_DATE}"
echo "[info] GC config: ${GC_CONFIG}"
echo "[info] Areas: ${AREAS[*]}"
echo "[info] Alpha: ${ALPHA}"
echo "[info] Smoothing bins: ${SMOOTH}"
echo "[info] Min consecutive: ${MINCON}"
echo "[info] Time window: ${TMIN} to ${TMAX}"
echo "[info] Tag: ${TAG}"

# Run the TTC analysis
python 11_trialwise_ttc.py \
  --sid "${SESSION_ID}" \
  --areas "${AREAS[@]}" \
  --root results/session \
  --alpha "${ALPHA}" \
  --smoothing_bins "${SMOOTH}" \
  --min_consecutive "${MINCON}" \
  --tmin "${TMIN}" \
  --tmax "${TMAX}" \
  --exclude_test_direction \
  --tag "${TAG}" \
  --min_theta_C "${MIN_THETA_C}" \
  --min_theta_R "${MIN_THETA_R}"

echo "[info] TTC analysis completed successfully!"
echo "[info] Check output files in: results/session/${SESSION_ID}/"
echo "[info] - ttc_${TAG}.npz (main results)"
echo "[info] - ttc_${TAG}.json (summary)"
