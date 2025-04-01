#!/bin/bash

set -x

GPU_OPT=""
if [ ! -d /opt/rocm ]
then
    cuda_opt=""
    if [ -n $CUDA_VERSION ] ; then
        cuda_ver_str=`echo $CUDA_VERSION | awk -F "." '{print $1"."$2}'`
        ARCH_TYPE=`uname -m`
        if [ $ARCH_TYPE == "x86_64" ]; then
            CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/x86_64-linux"
        elif [ $ARCH_TYPE == "aarch64" ]; then
            CUDA_DIR="/usr/local/cuda-$cuda_ver_str/targets/sbsa-linux"
        fi
        cuda_opt=" --with-cuda=/usr/local/cuda-$cuda_ver_str "
        GPU_OPT="${cuda_opt} --with-cuda-libdir=/usr/local/cuda-$cuda_ver_str/lib64/stubs"
    fi
else
    GPU_OPT="--with-rocm"
fi

# Create patch for OMPI 5.0.7 for lnx support. Remove this once use newer
# OMPI that no longer requires this!
echo "--- ./ompi/mca/mtl/ofi/mtl_ofi_component.c.orig 2024-11-15 08:18:09.000000000 -0600
+++ ./ompi/mca/mtl/ofi/mtl_ofi_component.c      2025-01-23 09:31:04.000000000 -0600
@@ -832,7 +832,8 @@
          * have a problem here since it uses fi_mr_regattr only within the context of an rcache, and manages the
          * requested_key field in this way.
          */
-         if (!strncasecmp(prov->fabric_attr->prov_name, \"cxi\", 3)) {
+         if ((NULL != strstr(prov->fabric_attr->prov_name, \"cxi\")) ||
+             (NULL != strstr(prov->fabric_attr->prov_name, \"CXI\")) ) {
              ompi_mtl_ofi.hmem_needs_reg = false;
          }
" > ${SCRIPT_DIR}/mtl_ofi_component.patch

OMPI_CONFIG_OPTIONS_VAR="--prefix ${HPC_DIR} --enable-prte-prefix-by-default \
   --enable-shared --with-cma --with-pic --with-libfabric=${HPC_DIR}         \
   --without-ucx --with-pmix=internal ${GPU_OPT}"

# Install OMPI
OMPI_VER=v5.0
OMPI_VER_NUM=5.0.7
OMPI_CONFIG_OPTIONS=${OMPI_CONFIG_OPTIONS_VAR}
OMPI_SRC_DIR=/tmp/openmpi-src
OMPI_BASE_URL="https://download.open-mpi.org/release/open-mpi"
OMPI_URL="${OMPI_BASE_URL}/${OMPI_VER}/openmpi-${OMPI_VER_NUM}.tar.gz"

mkdir -p ${OMPI_SRC_DIR}                        && \
  cd ${OMPI_SRC_DIR}                            && \
  wget ${OMPI_URL}                              && \
  tar -xzf openmpi-${OMPI_VER_NUM}.tar.gz       && \
  cd openmpi-${OMPI_VER_NUM}                    && \
  patch ./ompi/mca/mtl/ofi/mtl_ofi_component.c ${SCRIPT_DIR}/mtl_ofi_component.patch && \
  ./configure ${OMPI_CONFIG_OPTIONS}            && \
  make                                          && \
  make install                                  && \
  cd /tmp                                       && \
  rm -rf ${OMPI_SRC_DIR}
