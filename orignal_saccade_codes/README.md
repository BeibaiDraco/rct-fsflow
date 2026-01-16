salloc --partition=voltron --qos=voltron --account=pi-bdoiron --time=02:00:00 --cpus-per-task=1 --mem=16G --job-name=interactive

srun --pty bash

module purge && module load python/3.11.9

source /project/bdoiron/dracoxu/rct-fsflow/runtime/venv/bin/activate
cd /project/bdoiron/dracoxu/rct-fsflow/saccade
export PYTHONPATH=$PWD:$PYTHONPATH



salloc --partition=voltron --qos=voltron --account=pi-bdoiron --time=02:00:00 --cpus-per-task=1 --mem=16G --job-name=interactive

srun --pty bash

BASE="/project/bdoiron/dracoxu/rct-fsflow"
PROJECT="$BASE/paper_project_final"
OUT_ROOT="$PROJECT/out"
SID_LIST="$PROJECT/sid_list.txt"   # newline-delimited list of 8-digit SIDs

module purge
module load python/3.11.9
source "$BASE/runtime/venv/bin/activate"