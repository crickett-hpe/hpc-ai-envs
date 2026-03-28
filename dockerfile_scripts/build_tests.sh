#!/usr/bin/env bash

set -x
SCRIPT_DIR=$(dirname "$0")
TDIR="/tmp/tests"
mkdir -p ${TDIR}
cd ${TDIR}
if [ ! -d /opt/rocm ]
then
    INSTALL_DIR="${HPC_DIR}/tests/nccl-tests"
    mkdir -p ${INSTALL_DIR}
    if [ -n $CUDA_VERSION ] ; then
        cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`
        CUDA_DIR="/usr/local/cuda-$cuda_ver_str"
    fi

    NCCL_VER="v2.15.0"
    NCCL_REPO="https://github.com/NVIDIA/nccl-tests.git"
    git clone --depth 1 --branch ${NCCL_VER} ${NCCL_REPO}
    cd nccl-tests
    make -j8  MPI=1 MPI_HOME=${HPC_DIR} CUDA_HOME=${CUDA_DIR} NCCL_HOME=${HPC_DIR} BUILDDIR=${INSTALL_DIR}
    rm ${INSTALL_DIR}/*.o
    rm -rf ${INSTALL_DIR}/verifiable
    ## Build tests/nccl-sanity.c
    make -C ${SCRIPT_DIR}

    ## OSU Benchmark Configuration Option
    OSU_CONFIG="--enable-cuda --with-cuda-include=/usr/local/cuda/include --with-cuda-libpath=/usr/local/cuda/lib64"
else
    INSTALL_DIR="${HPC_DIR}/tests/rccl-tests"
    mkdir -p ${INSTALL_DIR}

    RCCL_REPO="https://github.com/ROCm/rccl-tests.git"
    git clone --depth 1 ${RCCL_REPO}
    cd rccl-tests
    if [[ ! -d /opt/rocm/rccl && -r /usr/local/lib/librccl.so ]]
    then
        export CUSTOM_RCCL_LIB=/usr/local/lib
    fi
    make -j8  MPI=1 MPI_HOME=${HPC_DIR} BUILDDIR=${INSTALL_DIR}
    rm ${INSTALL_DIR}/*.o
    rm -rf ${INSTALL_DIR}/verifiable
    rm -rf ${INSTALL_DIR}/src
    rm -rf ${INSTALL_DIR}/hipify

    ## OSU Benchmark Configuration Option
    OSU_CONFIG="--enable-rocm --with-rocm=/opt/rocm"
fi

## BUILD OSU Benchmark
OSU_VER=7.5.2
cd ${TDIR}
OSU_REPO="https://mvapich.cse.ohio-state.edu/download/mvapich"
wget ${OSU_REPO}/osu-micro-benchmarks-${OSU_VER}.tar.gz
tar -xzf osu-micro-benchmarks-${OSU_VER}.tar.gz --no-same-owner
cd osu-micro-benchmarks-${OSU_VER}
./configure CC=${HPC_DIR}/bin/mpicc CXX=${HPC_DIR}/bin/mpicxx \
        --prefix=${HPC_DIR}/tests \
	$OSU_CONFIG
make
make install
mv ${HPC_DIR}/tests/libexec/osu-micro-benchmarks ${HPC_DIR}/tests
rm -rf ${HPC_DIR}/tests/libexec

## Clean-up
cd /tmp
rm -rf ${TDIR}

