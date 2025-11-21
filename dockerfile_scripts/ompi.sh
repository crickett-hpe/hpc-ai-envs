#!/bin/bash

set -x

OMPI_VER_NUM=$1

GPU_OPT=""
if [ ! -d /opt/rocm ]
then
    cuda_opt=""
    if [ -n $CUDA_VERSION ] ; then
        cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`
        ARCH_TYPE=`uname -m`
	CUDA_DIR="/usr/local/cuda-$cuda_ver_str"

	if [[ ! -e $CUDA_DIR && -e /opt/nvidia/hpc_sdk ]]; then
	    CUDA_DIR="/opt/nvidia/hpc_sdk/Linux_${ARCH_TYPE}/${HPCSDK_VERSION}/cuda"
	fi

        cuda_opt=" --with-cuda=${CUDA_DIR} "
        GPU_OPT="${cuda_opt} --with-cuda-libdir=${CUDA_DIR}/lib64/stubs"
    fi
else
    GPU_OPT="--with-rocm"
fi

OMPI_CONFIG_OPTIONS_VAR="--prefix ${HPC_DIR} --enable-prte-prefix-by-default \
   --enable-shared --with-cma --with-pic --with-libfabric=${HPC_DIR}         \
   --without-ucx --with-pmix=internal ${GPU_OPT}"

# Install OMPI
OMPI_VER=v$(echo $OMPI_VER_NUM | awk -F '.' '{print$1"."$2}')
#OMPI_VER_NUM=5.0.8
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
