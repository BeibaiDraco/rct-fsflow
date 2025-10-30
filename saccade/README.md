salloc --partition=voltron --qos=voltron --account=pi-bdoiron --time=01:00:00 --cpus-per-task=1 --mem=16G --job-name=interactive

srun --pty bash

module purge && module load python/3.11.9

source /project/bdoiron/dracoxu/rct-fsflow/runtime/venv/bin/activate
cd /project/bdoiron/dracoxu/rct-fsflow/saccade
export PYTHONPATH=$PWD:$PYTHONPATH