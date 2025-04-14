#!/usr/bin/env bash

set -e

apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y pdsh libaio-dev

# Starting from the 25.03 release, the PyTorch container has implemented a
# pip constraints file at /etc/pip/constraint.txt. This file specifies the
# versions of all python packages used during the PyTorch container creation,
# and is included to prevent unintentional overwriting of any of the
# project's dependencies. To install a different version of one of the
# packages constrained here, the file /etc/pip/constraint.txt within the
# container must be modified. Simply remove the version constraints for any
# packages that you want to overwrite, keeping in mind that any other
# versions than those specified in the constraint file have not been fully
# tested in the container.
CONSTRAINT_FILE="/etc/pip/constraint.txt"
if [ -e $CONSTRAINT_FILE ]; then
    sed -i '/^dill/d' $CONSTRAINT_FILE
    sed -i '/^fsspec/d' $CONSTRAINT_FILE
fi

#Older versions of deepspeed require pinned pydantic version
python -m pip install pydantic

# Install some dependencies for the LLM test
pip install 'datasets>=3.5.0'
pip install accelerate arrow huggingface-hub packaging safetensors setuptools tokenizers transformers xxhash evaluate
#Precompile deepspeed ops except sparse_attn which has dubious support
# Skip precompiling since this fails when using the NGC base image.
# Need to verify that DS can use NCCL correctly for the comms, etc.
#export DS_BUILD_OPS=1
#Precompile supported deepspeed ops except sparse_attn
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_EVOFORMER_ATTN=0
export DS_BUILD_CUTLASS_OPS=0
export DS_BUILD_RAGGED_DEVICE_OPS=0

cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`

ARCH_TYPE=`uname -m`
if [ $ARCH_TYPE == "x86_64" ]; then
  export CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/x86_64-linux"
elif [ $ARCH_TYPE == "aarch64" ]; then
  export CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/sbsa-linux"
fi

#Remove 5.2 from TORCH_CUDA_ARCH_LIST, it is no longer supported by deepspeed
export TORCH_CUDA_ARCH_LIST=`echo $TORCH_CUDA_ARCH_LIST|sed 's/5.2 //'`
python -m pip install $DEEPSPEED_PIP --no-binary deepspeed
python -m deepspeed.env_report

