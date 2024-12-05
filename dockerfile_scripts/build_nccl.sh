#!/usr/bin/env bash

set -x

cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`
CUDA_DIR="/usr/local/cuda-$cuda_ver_str"
    
git clone https://github.com/nvidia/nccl.git /tmp/nccl_src


(cd /tmp/nccl_src && git checkout v2.23.4-1)

export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
#make DEBUG=1 NVCC_GENCODE=${NVCC_GENCODE} CUDA_HOME=${CUDA_DIR} PREFIX=${HOROVOD_NCCL_HOME} -C /tmp/nccl_src -j 4 install
make CUDA_HOME=${CUDA_DIR} NVCC_GENCODE=${NVCC_GENCODE} PREFIX=${HOROVOD_NCCL_HOME} -C /tmp/nccl_src -j 4 install

rm -rf /tmp/nccl_src


