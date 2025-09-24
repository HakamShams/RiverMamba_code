#!/bin/bash

#SBATCH --job-name=test           # create a short name for your job
#SBATCH --nodes=2                 # node count
#SBATCH --ntasks-per-node=1       # total number of tasks per node
#SBATCH --cpus-per-task=80        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=500G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4              # number of allocated gpus per node
#SBATCH --time=3-00:00:00         # total run time limit (HH:MM:SS)
#SBATCH --account=xxxx
#SBATCH --output=./test.out
#SBATCH --error=./test.err
#SBATCH --partition=sgpu_long     # long  # devel  # medium

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo MASTER_PORT=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

eval "$(conda shell.bash hook)"
conda activate rivermamba

echo "SLURM_NTASKS="$SLURM_NTASKS
echo "SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE
echo "SLURM_NNODES="$SLURM_NNODES

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=${SLURM_GPUS_ON_NODE} --rdzv_id=${SLURM_JOBID} --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_multinode.py

