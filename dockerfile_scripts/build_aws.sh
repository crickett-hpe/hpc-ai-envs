#!/bin/bash

set -x

WITH_AWS_TRACE=""
if [ $# -gt 2 ] ; then
    if [ "$3" = "1" ] ; then
	# Tell AWS to build with trace messages enabled
	WITH_AWS_TRACE="--enable-trace"
    fi
fi
OFI=$1
WITH_XCCL=$2

apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
				      --no-install-recommends tcsh

# Install AWS_OFI_NCCL
AWS_VER=v1.16.3
AWS_VER_NUM=1.16.3
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
if [ ! -d /opt/rocm ]
then
    CUDA_DIR="/usr/local/cuda-$cuda_ver_str"
    ARCH_TYPE=`uname -m`
    if [[ ! -e $CUDA_DIR && -e /opt/nvidia/hpc_sdk ]]; then
        CUDA_DIR="/opt/nvidia/hpc_sdk/Linux_${ARCH_TYPE}/${HPCSDK_VERSION}/cuda"
    fi
fi

AWS_SRC_DIR=/tmp/aws-ofi-nccl
ROCM_DIR=/opt/rocm/
mkdir -p ${AWS_SRC_DIR}
cd ${AWS_SRC_DIR}

if [ ! -d ${ROCM_DIR} ]
then
    echo "Building AWS for NVidia"
    AWS_CONFIG_OPTIONS="--prefix ${HPC_DIR}  \
      --with-libfabric=${HPC_DIR}            \
      --with-mpi=${HPC_DIR}                  \
      --with-cuda=${CUDA_DIR} ${WITH_AWS_TRACE}"

    AWS_BASE_URL="https://github.com/aws/aws-ofi-nccl/archive/refs/tags"
    AWS_URL="${AWS_BASE_URL}/${AWS_VER}.tar.gz"
    AWS_BASE_URL="https://github.com/aws/aws-ofi-nccl/releases/download"
    AWS_NAME="${AWS_NAME}-${AWS_VER_NUM}"
    AWS_URL="${AWS_BASE_URL}/${AWS_VER}/${AWS_NAME}.tar.gz"

    echo "Building AWS for NVIDIA $AWS_CONFIG_OPTIONS WITH_XCCL=$WITH_XCCL"
    if [ "$WITH_XCCL" == "1" ]; then
        #### git clone https://github.com/HewlettPackard/open-ofi-xccl.git
        git clone https://github.com/ryanhankins/open-ofi-xccl.git
        cd open-ofi-xccl
        git checkout v1.14.x-xccl
    else
        ###export CC=g++
        wget ${AWS_URL}
        tar -xzf ${AWS_NAME}.tar.gz --no-same-owner
        cd ${AWS_NAME}

	CUDA_VERSION_NUM=`echo $CUDA_VERSION | awk -F "." '{print $1}'`
	if [ $CUDA_VERSION_NUM -gt 12 ]; then
	    patch -p1 -i /tmp/dockerfile_scripts/patches/cuda-${CUDA_VERSION_NUM}/aws-ofi-nccl.patch
	fi
    fi
else
    AWS_CONFIG_OPTIONS="--prefix ${HPC_DIR}  \
      --with-libfabric=${HPC_DIR}            \
      --with-rccl=${ROCM_DIR}/rccl           \
      --with-mpi=${HPC_DIR}                  \
      --with-rocm=${ROCM_DIR}                \
      --with-hip=${ROCM_DIR}   ${WITH_AWS_TRACE}"
    echo "Building AWS for AMD $AWS_CONFIG_OPTIONS WITH_XCCL=$WITH_XCCL"
    if [ "$WITH_XCCL" == "1" ]; then
        git clone https://github.com/ryanhankins/open-ofi-xccl.git
        cd open-ofi-xccl
        git checkout v1.14.x-xccl
        ###
        ### The following magic indicates during compile time that if
        ### HAVE_CUDA and HAVE_NEURON are not defined, that GPUDirect
        ### is still supported by libfabric provider.  This is because
        ### at this time, HAVE_ROCM is defined.
        ###
        cat > xccl.patch <<EOF
diff --git a/src/nccl_ofi_ofiutils.c b/src/nccl_ofi_ofiutils.c
index 4cf3305..8f4bb1a 100644
--- a/src/nccl_ofi_ofiutils.c
+++ b/src/nccl_ofi_ofiutils.c
@@ -364,9 +364,7 @@ int nccl_ofi_ofiutils_init_connection(struct fi_info *info, struct fid_domain *d
 		 */
 		support_gdr = GDR_SUPPORTED;
 #else
-		NCCL_OFI_WARN("Using Libfabric 1.18 API with GPUDirect RDMA support, and FI_OPT_CUDA_API_PERMITTED is not declared.");
-		ret = -EOPNOTSUPP;
-		goto error;
+		support_gdr = GDR_SUPPORTED;
 #endif
 	}
 	/* Run platform-specific endpoint configuration hook if declared */
EOF
        git apply --ignore-space-change --ignore-whitespace xccl.patch
    else
        git clone https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
        cd aws-ofi-rccl
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
    fi
    export CC=hipcc
    export CFLAGS="-D__HIP_PLATFORM_AMD__"
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
