#!/bin/bash

export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Setting this to 0 showed considerably better performance for
# the nccl all_reduce test
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export FI_HMEM_CUDA_USE_GDRCOPY=1
export FI_MR_CACHE_MONITOR=userfaultfd

# Try setting up the info for torch distributed from slurm 
# Taken from:
# https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
# Set the master_addr using this command from the cmd-line
#master_addr=$(/usr/bin/scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR="$MASTER_ADDR

export RANK=$SLURM_PROCID
export SLURM_GPUS_ON_NODE=4

fi_settings=`env | grep FI_`
echo "FI_*: $fi_settings"
nccl_settings=`env | grep NCCL_`
echo "NCCL_*: $nccl_settings"

# Execute what we were told to execute
exec "${@}"
