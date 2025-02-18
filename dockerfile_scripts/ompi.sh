#!/bin/bash

set -x

cuda_opt=""
if [ -n $CUDA_VERSION ] ; then
    cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`
    ARCH_TYPE=`uname -m`
    if [ $ARCH_TYPE == "x86_64" ]; then
	CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/x86_64-linux"
    elif [ $ARCH_TYPE == "aarch64" ]; then
	CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/sbsa-linux"
    fi
#    cuda_opt=" --with-cuda=${CUDA_DIR} "
    cuda_opt=" --with-cuda=/usr/local/cuda-$cuda_ver_str "
fi

OMPI_CONFIG_OPTIONS_VAR="--prefix ${HPC_DIR} --enable-prte-prefix-by-default --enable-shared --with-cma --with-pic --with-libfabric=${HPC_DIR} --without-ucx --with-pmix=internal ${cuda_opt} --with-cuda-libdir=/usr/local/cuda-$cuda_ver_str/lib64/stubs"

# Install OMPI
OMPI_VER=v5.0
OMPI_VER_NUM=5.0.6
OMPI_CONFIG_OPTIONS=${OMPI_CONFIG_OPTIONS_VAR}
OMPI_SRC_DIR=/tmp/openmpi-src
OMPI_BASE_URL="https://download.open-mpi.org/release/open-mpi"
OMPI_URL="${OMPI_BASE_URL}/${OMPI_VER}/openmpi-${OMPI_VER_NUM}.tar.gz"

mkdir -p ${OMPI_SRC_DIR}                        && \
  cd ${OMPI_SRC_DIR}                            && \
  wget ${OMPI_URL}                              && \
  tar -xzf openmpi-${OMPI_VER_NUM}.tar.gz       && \
  cd openmpi-${OMPI_VER_NUM}                    && \
  ./configure ${OMPI_CONFIG_OPTIONS}            && \
  make                                          && \
  make install                                  && \
  cd /tmp                                       && \
  rm -rf ${OMPI_SRC_DIR}
