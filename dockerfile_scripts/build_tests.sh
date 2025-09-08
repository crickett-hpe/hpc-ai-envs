set -x
SCRIPT_DIR=$(dirname "$0")
TDIR="/tmp/tests"
SDIR="/tmp/nccl_src"
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
    make -C ${SCRIPT_DIR}

#    NCCL_MAJOR=$(grep "NCCL_MAJOR " /usr/include/nccl.h | awk '{print $NF}')
#    NCCL_MINOR=$(grep "NCCL_MINOR " /usr/include/nccl.h | awk '{print $NF}')
#    NCCL_PATCH=$(grep "NCCL_PATCH " /usr/include/nccl.h | awk '{print $NF}')
#
#    git clone https://github.com/nvidia/nccl.git $SDIR
#    #(cd $SDIR && git checkout v${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}-1)
#    (cd $SDIR && git checkout v2.27.7-1)
#
#    # Make the example nccl profiler
#    cd ${SDIR}/ext-profiler/example && \
#        make && \
#        cp libnccl-profiler*.so ${HPC_DIR}/lib

    set -x

else
    INSTALL_DIR="${HPC_DIR}/tests/rccl-tests"
    mkdir -p ${INSTALL_DIR}

    RCCL_REPO="https://github.com/ROCm/rccl-tests.git"
    git clone --depth 1 ${RCCL_REPO}
    cd rccl-tests
    make -j8  MPI=1 MPI_HOME=${HPC_DIR} BUILDDIR=${INSTALL_DIR}
    rm ${INSTALL_DIR}/*.o
    rm -rf ${INSTALL_DIR}/verifiable
    rm -rf ${INSTALL_DIR}/src
    rm -rf ${INSTALL_DIR}/hipify
fi
cd /tmp
rm -rf ${TDIR} ${SDIR}

