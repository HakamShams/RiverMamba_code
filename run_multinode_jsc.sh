#!/bin/bash

#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=48       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=xxx
#SBATCH --output=./test.out
#SBATCH --error=./test.err
#SBATCH --partition=booster

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo MASTER_PORT=$MASTER_PORT
#echo "WORLD_SIZE="$WORLD_SIZE

#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
#export MASTER_PORT=54123

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

#eval "$(conda shell.bash hook)"

#module load miniforge
#source activate rivermamba

module load Python
#source ./my_env/bin/activate  # path to your local virtual env on JSC

echo "SLURM_NTASKS="$SLURM_NTASKS
echo "SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE
echo "SLURM_NNODES="$SLURM_NNODES

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} torchrun_jsc --nnodes=${SLURM_NNODES} --nproc_per_node=${SLURM_GPUS_ON_NODE} --rdzv_id=${SLURM_JOBID} --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_multinode.py

