#!/bin/bash

set -x

WITH_AWS_TRACE=""
if [ $# -gt 1 ] ; then
    if [ "$2" = "1" ] ; then
	# Tell AWS to build with trace messages enabled
	WITH_AWS_TRACE="--enable-trace"
    fi
fi
OFI=$1

apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
				      --no-install-recommends tcsh

# Install AWS_OFI_NCCL
AWS_VER=v1.13.1
AWS_VER_NUM=1.13.1
AWS_NAME=aws-ofi-nccl
AWS_FILE="${AWS_NAME}-${AWS_VER_NUM}"
cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`
GDRCOPY_HOME="/usr"

# cuda install dir likely dependent on BaseOS (i.e. ubuntu 20.04)
# in case this changes in the future
# ARCH_TYPE=`uname -m`
# if [ $ARCH_TYPE == "x86_64" ]; then
#     CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/x86_64-linux"
# elif [ $ARCH_TYPE == "aarch64" ]; then
#     CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/sbsa-linux"
# fi
# Cuda path, including version. This should be sufficient for the build
CUDA_DIR=" --with-cuda=/usr/local/cuda-$cuda_ver_str "

AWS_SRC_DIR=/tmp/aws-ofi-nccl
mkdir -p ${AWS_SRC_DIR}
cd ${AWS_SRC_DIR}

if [ ! -d /opt/rocm ]
then
    echo "Building AWS for NVidia"
    AWS_CONFIG_OPTIONS="--prefix ${HPC_DIR}  \
      --with-libfabric=${HPC_DIR}            \
      --with-mpi=${HPC_DIR}                  \
      --with-cuda=${CUDA_DIR} ${WITH_AWS_TRACE}"

    AWS_BASE_URL="https://github.com/aws/aws-ofi-nccl/archive/refs/tags"
    AWS_URL="${AWS_BASE_URL}/${AWS_VER}.tar.gz"
    AWS_BASE_URL="https://github.com/aws/aws-ofi-nccl/releases/download"
    AWS_NAME="${AWS_NAME}-${AWS_VER_NUM}-aws"
    AWS_URL="${AWS_BASE_URL}/${AWS_VER}-aws/${AWS_NAME}.tar.gz"
    wget ${AWS_URL}
    tar -xzf ${AWS_NAME}.tar.gz --no-same-owner
    cd ${AWS_NAME}
else
    echo "Building AWS for AMD"
    AWS_CONFIG_OPTIONS="--prefix ${HPC_DIR}  \
      --with-libfabric=${HPC_DIR}            \
      --with-rccl=${HOROVOD_NCCL_HOME}       \
      --with-mpi=${HPC_DIR}                  \
      --with-hip=${ROCM_DIR} ${WITH_AWS_TRACE}"
    git clone https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
    cd aws-ofi-rccl
    export CC=hipcc
    export CFLAGS="-D__HIP_PLATFORM_AMD__"
    ###
    ### The following magic addresses https://github.com/ROCm/aws-ofi-rccl/pull/14
    ### until such time that the aws-ofi-rccl repo is updated.
    ###
    sed -i '39i\
/* Copied from libfabric:rdma/fabric.h@30ec628: "libfabric: Initial commit" */\
#ifndef container_of\
#define container_of(ptr, type, field) ((type *) ((char *)ptr - offsetof(type, field)))\
#endif\
/* end of copied libfabric macros */\
'   include/nccl_ofi.h
    head -50 include/nccl_ofi.h
    ###
    ###  End of magic
    ###
fi

./autogen.sh                                  && \
./configure ${AWS_CONFIG_OPTIONS}             && \
make                                          && \
make install                                  && \
cd /tmp                                       && \
rm -rf ${AWS_SRC_DIR}
