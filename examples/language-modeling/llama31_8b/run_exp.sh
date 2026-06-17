#!/bin/bash

CONF=$1
SIF=$2
NUM_EXP=$3

# Vars without defaults
: "${CONF:?CONF is not set}"
: "${SIF:?SIF is not set}"

# Vars with defaults
: "${NUM_EXP:=1}"

export WORKDIR=$(dirname ${BASH_SOURCE[0]})

export MYROOT=$(realpath /lus/scratch/${USER})
export DATADIR=${MYROOT}/mlperf/data
export TRITON_CACHE_DIR=${MYROOT}/triton

export REMOUNT_WORKDIR=$(realpath $PWD)
export EXTRA_MOUNTS="${WORKDIR}/cuda-gdb-pipes:/workspace/cuda-gdb-pipes,${TRITON_CACHE_DIR}"

export CONT=$SIF

mkdir -p ${WORKDIR}/cuda-gdb-pipes

source $CONF

if [ $WALLTIME -gt 600 ]; then
    export WALLTIME=600
fi

export NEXP=$NUM_EXP

sbatch -p blancapeak -N ${DGXNNODES} --ntasks-per-node=${DGXNGPU} --time=${WALLTIME} run.sub
