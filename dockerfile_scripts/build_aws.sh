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
CUDA_DIR="/usr/local/cuda-$cuda_ver_str"

AWS_CONFIG_OPTIONS="--prefix ${HPC_DIR}             \
	  --with-libfabric=${HPC_DIR}               \
	  --with-mpi=${HPC_DIR}                     \
	  --with-cuda=${CUDA_DIR} ${WITH_AWS_TRACE}"

# FI_MR_DMABUF:
# This flag indicates that the memory region to registered is
# a DMA-buf backed region. When set, the region is specified through
# the dmabuf field of the fi_mr_attr structure. This flag is only
# usable for domains opened with FI_HMEM capability support.
# This flag is introduced since Libfabric 1.20.
#
# When this flag was left default with 'yes' it seems to cause
# seg-fault on internal A100 sytem (pinoak)
ARCH_TYPE=`uname -m`
if [ $ARCH_TYPE == "x86_64" ]; then
    AWS_CONFIG_OPTIONS="$AWS_CONFIG_OPTIONS \
	    ac_cv_have_decl_FI_MR_DMABUF=no"
fi

AWS_SRC_DIR=/tmp/aws-ofi-nccl
AWS_BASE_URL="https://github.com/aws/aws-ofi-nccl/archive/refs/tags"
AWS_URL="${AWS_BASE_URL}/${AWS_VER}.tar.gz"
AWS_BASE_URL="https://github.com/aws/aws-ofi-nccl/releases/download"
AWS_NAME="${AWS_NAME}-${AWS_VER_NUM}-aws"
AWS_URL="${AWS_BASE_URL}/${AWS_VER}-aws/${AWS_NAME}.tar.gz"

mkdir -p ${AWS_SRC_DIR}                           && \
    cd ${AWS_SRC_DIR}                             && \
    wget ${AWS_URL}                               && \
    tar -xzf ${AWS_NAME}.tar.gz --no-same-owner   && \
    cd ${AWS_NAME}                                && \
    ./autogen.sh                                  && \
    ./configure ${AWS_CONFIG_OPTIONS}             && \
    make                                          && \
    make install                                  && \
    cd /tmp                                       && \
    rm -rf ${AWS_SRC_DIR}
