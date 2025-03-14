#!/bin/bash

set -x

# See if we should add ROCM to the OMPI build
OMPI_WITH_ROCM=""
if [ $# -gt 2 ] ; then
    if [ "$3" = "1" ] ; then
	# Tell OMPI to look for rocm in the default location
	OMPI_WITH_ROCM="--with-rocm"
    fi
fi

if [ "$4" = "1" ];then WITH_MPICH=1;fi

OS_VER=$1
OFI=$2
if [ "$OFI" = "1" ]; then
  # Install OFI
  OFI_VER=1.19.0
  OFI_CONFIG_OPTIONS="--prefix ${HPC_DIR}"
  OFI_SRC_DIR=/tmp/ofi-src
  OFI_BASE_URL="https://github.com/ofiwg/libfabric/releases/download"
  OFI_URL="${OFI_BASE_URL}/v${OFI_VER}/libfabric-${OFI_VER}.tar.bz2"

  mkdir -p ${OFI_SRC_DIR}                              && \
  mkdir -p ${OFI_INSTALL_DIR}                          && \
    cd ${OFI_SRC_DIR}                                  && \
    wget ${OFI_URL}                                    && \
    tar -xf libfabric-${OFI_VER}.tar.bz2              && \
    cd libfabric-${OFI_VER}                            && \
    ./configure ${OFI_CONFIG_OPTIONS}                  && \
    make install                                       && \
    cd /tmp                                            && \
    rm -rf ${OFI_SRC_DIR}

  #OMPI CONFIG ARGS FOR OFI
  # NCCL build OMPI_CONFIG_OPTIONS_VAR="--prefix ${HPC_DIR} --enable-prte-prefix-by-default --enable-shared --with-cma --with-pic --with-libfabric=${HPC_DIR} --without-ucx --with-pmix=internal --with-cuda=${cuda_opt}"

  OMPI_CONFIG_OPTIONS_VAR="--prefix ${HPC_DIR} --enable-prte-prefix-by-default --enable-shared --with-cma --with-pic --with-libfabric=${HPC_DIR} --without-ucx --with-pmix=internal ${OMPI_WITH_ROCM}"

#echo OMPI_CONFIG_OPTIONS_VAR $OMPI_CONFIG_OPTIONS_VAR= 

#exit 99

else
  # Install the Mellanox OFED stack.  Note that this is dependent on
  # what the base OS is (ie, Ubuntu 20.04) so if that changes then
  # this needs updated.  MOFED_VER=5.0-2.1.8.0 MOFED_VER=5.5-1.0.3.2
  MOFED_VER=5.4-3.4.0.0
  PLATFORM=x86_64
  MOFED_TAR_URL="http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}"
  MOFED_TAR="MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz"
  TMP_INSTALL_DIR=/tmp/ofed
  
  mkdir -p ${TMP_INSTALL_DIR}                                          && \
     cd ${TMP_INSTALL_DIR}                                             && \
     wget --quiet "${MOFED_TAR_URL}/${MOFED_TAR}"                      && \
     tar -xvf ${MOFED_TAR}                                             && \
     MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall   \
       --user-space-only --without-fw-update --all --force                \
       --skip-unsupported-devices-check                                && \
     rm -rf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz        \
            MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}            \
            MLNX_OFED_LINUX.*.logs                                     && \
     rm -rf ${TMP_INSTALL_DIR}

  # Install UCX
  UCX_VER=1.10.1
  UCX_CONFIG_OPTIONS="--prefix ${UCX_INSTALL_DIR} --enable-mt"
  UCX_SRC_DIR=/tmp/ucx-src
  UCX_BASE_URL="https://github.com/openucx/ucx/releases/download"
  UCX_URL="${UCX_BASE_URL}/v${UCX_VER}/ucx-${UCX_VER}.tar.gz"

  mkdir -p ${UCX_SRC_DIR}                              && \
    cd ${UCX_SRC_DIR}                                  && \
    wget ${UCX_URL}                                    && \
    tar -xzf ucx-${UCX_VER}.tar.gz                     && \
    cd ucx-${UCX_VER}                                  && \
    ./contrib/configure-release ${UCX_CONFIG_OPTIONS}  && \
    make -j8 install                                   && \
    cd /tmp                                            && \
    rm -rf ${UCX_SRC_DIR}

  #OMPI CONFIG ARGS FOR UCX
  OMPI_CONFIG_OPTIONS_VAR="--prefix ${OMPI_INSTALL_DIR} --enable-shared --with-verbs --with-cma --with-pic --enable-mpi-cxx --enable-mpi-thread-multiple --with-pmi --with-pmix=internal --with-platform=contrib/platform/mellanox/optimized --with-ucx=/container/ucx ${OMPI_WITH_ROCM}"

fi

if [ "$WITH_MPICH" != "1" ]; then
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
else
MPICH_VER=4.1.2
MPICH_CONFIG_OPTIONS="--with-device=ch4:ofi --with-libfabric=${HPC_DIR} --prefix=${HPC_DIR} --disable-fortran"
MPICH_URL="http://www.mpich.org/static/downloads/${MPICH_VER}/mpich-${MPICH_VER}.tar.gz"
MPICH_SRC_DIR=/tmp/mpich-src

mkdir -p ${MPICH_SRC_DIR}                                 &&\
  cd ${MPICH_SRC_DIR}                                     &&\
  curl -fSsL --retry 3 "${MPICH_URL}" | tar -xz --strip 1 &&\
  ./configure ${MPICH_CONFIG_OPTIONS}                     &&\
  make                                                    &&\
  make install                                            &&\
  rm -rf "${MPICH_SRC_DIR}"

fi
