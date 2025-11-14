#!/usr/bin/env bash

set -x

NCCL_VERSION=$1

cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`
CUDA_DIR="/usr/local/cuda-$cuda_ver_str"
    
git clone https://github.com/nvidia/nccl.git /tmp/nccl_src

(cd /tmp/nccl_src && git checkout v${NCCL_VERSION}-1)

## Cuda Compute Capability
## 8.0: A100, A30
## 9.0: GH200, H200, H100
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90"
make CUDA_HOME=${CUDA_DIR} NVCC_GENCODE="${NVCC_GENCODE}" PREFIX=${HOROVOD_NCCL_HOME} -C /tmp/nccl_src -j 4 install

# Make the example nccl profiler
cd /tmp/nccl_src/ext-profiler/example && \
    make && \
    cp libnccl-profiler-example.so ${HOROVOD_NCCL_HOME}/lib

rm -rf /tmp/nccl_src


