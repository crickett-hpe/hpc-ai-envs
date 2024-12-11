#!/bin/bash

set -x

# The NGC base image from 24.11 and newer seems to include a build of
# libfabric and the AWS plugin. We need to remove it to prevent issues
# that could occur if that version is loaded instead of the libfabric/ofi
# libraries we build here.
if [ -d "/opt/amazon" ] ; then
    rm -rf /opt/amazon
fi

# Install Cray libcxi. This requires grabbing the cassini/cxi headers
# and installing them into ${HPC_DIR} so we can compile libcxi.
cray_src_dir=/tmp/cray-libs
mkdir -p $cray_src_dir && \
    cd $cray_src_dir && \
    git clone https://github.com/HewlettPackard/shs-cassini-headers.git && \
    git clone https://github.com/HewlettPackard/shs-cxi-driver.git && \
    git clone https://github.com/HewlettPackard/shs-libcxi.git && \
    git clone https://github.com/HewlettPackard/shs-libfabric.git

# Install the cassini headers
cd $cray_src_dir/shs-cassini-headers && \
    cp -r include ${HPC_DIR} && \
    cp -r share ${HPC_DIR} && \
    cp -r share/cassini-headers /usr/share && \
    cp -r share/cassini-headers ${HPC_DIR}/share && \
    cd ../

# Install the cxi-driver headers
cd $cray_src_dir/shs-cxi-driver && \
    cp -r include ${HPC_DIR} && \
    cp include/linux/cxi.h ${HPC_DIR}/include && \
    cd ../
    
# Build libcxi. Note that this will install into ${HPC_DIR} by default,
# which is what we want so that libfabric/ompi/aws can easily find it.
#cxi_cflags="-Wno-unused-variable -Wno-unused-but-set-variable -g -O0" 
#cxi_cppflags="-Wno-unused-variable -Wno-unused-but-set-variable -g -O0"
cxi_cflags="-Wno-unused-variable -Wno-unused-but-set-variable -I${HPC_DIR}/include -I${HPC_DIR}/linux -I${HPC_DIR}/uapi" 
cxi_cppflags="-Wno-unused-variable -Wno-unused-but-set-variable -I${HPC_DIR}/include -I${HPC_DIR}/linux -I${HPC_DIR}/uapi"
cd $cray_src_dir/shs-libcxi && \
    git checkout -b release/shs-12.0 && \
    ./autogen.sh && \
    ./configure --prefix=${HPC_DIR} \
		CFLAGS="${cxi_cflags}" CPPFLAGS="${cxi_cppflags}" && \
    make && \
    make install && \
    cd ../

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

# Build and install libfabric. Note that this should see the cxi bits
# and enable cxi support. It should also install into ${HPC_DIR} so that
# it is easier for ompi/aws to find it.
cray_ofi_config_opts="--prefix=${HPC_DIR} --with-cassini-headers=${HPC_DIR} --with-cxi-uapi-headers=${HPC_DIR} --enable-cxi=${HPC_DIR} $cuda_opt --enable-cuda-dlopen --enable-gdrcopy-dlopen --disable-sockets --disable-udp --disable-verbs --disable-mrail --disable-rxd --disable-shm --disable-usnic --disable-rstream --disable-efa --disable-psm2 --disable-psm3 --disable-opx"
#ofi_cflags="-Wno-unused-variable -Wno-unused-but-set-variable -g -O0" 
#ofi_cppflags="-Wno-unused-variable -Wno-unused-but-set-variable -g -O0"
ofi_cflags="-Wno-unused-variable -Wno-unused-but-set-variable -I${HPC_DIR}/include -I${HPC_DIR}/linux -I${HPC_DIR}/uapi" 
ofi_cppflags="-Wno-unused-variable -Wno-unused-but-set-variable -I${HPC_DIR}/include -I${HPC_DIR}/linux -I${HPC_DIR}/uapi"
cd $cray_src_dir && \
    git clone https://github.com/ofiwg/libfabric.git && \
    cd libfabric && \
    git checkout -b v2.0.x && \
    ./autogen.sh && \
    ./configure CFLAGS="${ofi_cflags}" CPPFLAGS="${ofi_cppflags}" \
		$cray_ofi_config_opts && \
    make && \
    make install && \
    cd ../

#cd $cray_src_dir/shs-libfabric && \
#    git checkout -b v2.0.x && \
#    ./autogen.sh && \
#    ./configure CFLAGS="${ofi_cflags}" CPPFLAGS="${ofi_cppflags}" \
#		$cray_ofi_config_opts && \
#    make && \
#    make install && \
#    cd ../

# Clean up our git repos used to build cxi/libfabric
rm -rf $cray_src_dir
