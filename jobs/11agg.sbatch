#!/bin/bash
#SBATCH --job-name=rct-agg
#SBATCH --output=logs/11agg_%j.out
#SBATCH --error=logs/11agg_%j.err
#SBATCH --partition=voltron
#SBATCH --qos=voltron
#SBATCH --account=pi-bdoiron
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

module purge
module load python/3.11.9
source /project/bdoiron/dracoxu/rct-fsflow/runtime/venv/bin/activate
cd /project/bdoiron/dracoxu/rct-fsflow

python 11_aggregate_flows.py --save_csv
