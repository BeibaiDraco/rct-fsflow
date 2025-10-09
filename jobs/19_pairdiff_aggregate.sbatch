#!/bin/bash
#SBATCH --job-name=pairdiff-agg
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH -o logs/19_pairdiff_agg_%j.out
#SBATCH -e logs/19_pairdiff_agg_%j.err

set -euo pipefail

REPO="/project/bdoiron/dracoxu/rct-fsflow"
SID_LIST="$REPO/results/sid_list.txt"  # produced earlier
# Edit or override TAGS at submit time: --export=ALL,TAGS="induced_k2_win016_p500 induced_k5_win016_p500"
TAGS="${TAGS:-induced_k2_win016_p500 induced_k5_win016_p500}"
ALPHA="${ALPHA:-0.05}"

module purge
module load python/3.11.9
source "$REPO/runtime/venv/bin/activate"

cd "$REPO"
mkdir -p logs

# Build sid list if missing
if [[ ! -f "$SID_LIST" ]]; then
  ls -1 results/session | grep -E '^[0-9]{8}$' > "$SID_LIST"
fi

echo "[info] TAGS=$TAGS"
python "$REPO/19_aggregate_pairdiff_metrics.py" \
  --sid-list-file "$SID_LIST" \
  --tags $TAGS \
  --alpha "$ALPHA"
