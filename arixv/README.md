# rct-fsflow

## Interactive Node Setup

Use these steps when you need an interactive shell on the Voltron partition:

1. From a login node, request an allocation:
   ```bash
   salloc --partition=voltron --qos=voltron --account=pi-bdoiron --time=01:00:00 --cpus-per-task=1 --mem=16G --job-name=interactive
   ```
2. After the allocation is granted, start an interactive shell on the compute node:
   ```bash
   srun --pty bash
   ```
   (Optional) Run `hostname` to confirm you are on a `midway3-####` compute node.
3. Inside the allocation, set up the Python environment:
   ```bash
   module purge && module load python/3.11.9
   cd /project/bdoiron/dracoxu/rct-fsflow
   source /project/bdoiron/dracoxu/rct-fsflow/runtime/venv/bin/activate
   ```
4. You can now run project scripts, for example:
   ```bash
   python 18_pair_difference_test.py --help
   ```

When you finish, exit the shell to release the allocation.
