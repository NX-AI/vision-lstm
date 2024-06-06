#!/bin/bash -l
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=8
#SBATCH --partition=PARTITION
#SBATCH --time={time}
#SBATCH --chdir={chdir}
#SBATCH --output={output}

# set the first node name as master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT={master_port}
# add all hostnames info for logging
export ALL_HOST_NAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

# activate conda env
conda activate {env_name}

# write python command to log file -> easy check for which run crashed if there is some config issue
echo python main_train.py {cli_args}

# run
srun --cpus-per-task 16 python main_train.py {cli_args}