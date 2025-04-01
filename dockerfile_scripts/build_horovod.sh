#!/bin/bash

# Make sure we have pip. Some base images, such as for HPL, will not.
if ! command -v pip &> /dev/null; then
    echo "Skipping Horovod install because pip not installed"
    exit 0;
fi

# Try and build a version of Horovod that works with c++-17, which is
# required by the latest PyTorch
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_CUDA_HOME=/usr/local/cuda
export HOROVOD_NCCL_LINK=SHARED
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITH_MPI=1
#export HOROVOD_WITH_PYTORCH=1
#export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITH_PYTORCH=$1
export HOROVOD_WITH_TENSORFLOW=$2
export HOROVOD_WITHOUT_MXNET=1
pip install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17

