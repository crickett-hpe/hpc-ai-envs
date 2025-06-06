#!/bin/bash

WHOAMI=$(whoami)
# Only scrape host libs if AWS/NCCL/OFI was built for this image
if [ -d "${HPC_DIR}/lib" ]
then
   host_dir="/det_libfabric"
   if [ ! -d "$host_dir" ]; then
     host_dir="/det_host"
     if [ ! -d "$host_dir" ]; then
       host_dir="/host"
     fi
   fi
   if [ -d "$host_dir" ]; then
       libfabric=`find $host_dir -name libfabric.so 2>/dev/null`
       libfabric_dir="$(dirname "$libfabric")"
       if [[ ! -z "$libfabric" ]] ; then
           # Need libfabric to be first in the LD_LIBRARY_PATH to
           # override what is in the container. However, we likely
           # need/want the libs that it is dependent upon to show up
           # after the container ones so we append the rest of the
           # host libs.  tmp_dir="`pwd`/tmp"
           tmp_dir="/var/tmp"
           tmp_lib_dir="$tmp_dir/${WHOAMI}/detAI/lib"
           mkdir -p $tmp_lib_dir
           for lib in `/bin/ls $libfabric_dir/ | grep -elibfabric -ecxi` ; do
               ln -s $libfabric_dir/$lib $tmp_lib_dir 2>/dev/null
           done
           # Prepend the tmp dir for the host libfabric
           export LD_LIBRARY_PATH=$tmp_lib_dir:$LD_LIBRARY_PATH
           # Append the rest of the host lib dirs to try and avoid
           # problems with libs in the container. Note we might want
           # to set these paths based on where we find the libs that
           # libfabric is dependent upon.
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$libfabric_dir
       else 
	       echo "libfabric not found within $host_dir." >&2
       fi # end if found libfabric.so
   fi # end if /det_libfabric exists
   # See if we mounted in host libs in the expected location

   host_dir="/det_host"
   if [ ! -d "$host_dir" ]; then
   	host_dir="/host"
   fi
   if [ -d "$host_dir" ]; then
       # to set these paths based on where we find the libs that
       # libfabric is dependent upon.
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$host_dir/usr/lib64
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$host_dir/usr/local/lib64
   fi # end if /det_host exists

   if [ -r /usr/lib/x86_64-linux-gnu/libp11-kit.so.0 ]
   then
      export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libp11-kit.so.0:$LD_PRELOAD
   fi
   if [ -r /usr/lib/x86_64-linux-gnu/libffi.so.7 ]
   then
      export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7:$LD_PRELOAD
   fi

   # Settings specific to RCCL
   if [ "$WITH_RCCL" = "1" ]; then
      if [ "$WITH_NFS_WORKAROUND" = "1" ]; then
         export MIOPEN_USER_DB_PATH="/tmp/${WHOAMI}_${SLURM_LOCALID}"
         export MIOPEN_USER_CACHE_PATH="$MIOPEN_USER_DB_PATH/.cache"
         export MIOPEN_CACHE_DIR=${MIOPEN_USER_CACHE_PATH}
         mkdir -p $MIOPEN_USER_DB_PATH
         mkdir -p $MIOPEN_USER_CACHE_PATH/miopen
         export HOME=${MIOPEN_USER_DB_PATH}

      fi
   fi # end if [WITH_RCCL = 1]
fi # end if [ -d $HPC_DIR ]

# Env settings we want for both nccl/rccl
tmp_nvcache_dir=$(mktemp -d -p /var/tmp ${WHOAMI}-nvcache-XXXXXXXX)
# The following env variables are only set if they are currently
# unset, empty or null. To override one or more of these variables,
# simply set them prior to the enrtrypoint of this image. Note that
# the user could provide their own wrapper script entrypoint to
# override these as well.
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:=${tmp_nvcache_dir}}
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:=true}
# NOTE: Disable memory registration to workaround the current issues
#       between libfabric and cuda.  When those issus are resolved,
#       simply set the vaiable to 0 before launching the container.
export FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER:=1}
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:=userfaultfd}
export FI_CXI_RDZV_GET_MIN=${FI_CXI_RDZV_GET_MIN:=0}
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=${FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD:=16777216}
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=${FI_CXI_DISABLE_NON_INJECT_MSG_IDC:=1}
export FI_CXI_RDZV_THRESHOLD=${FI_CXI_RDZV_THRESHOLD:=0}
export FI_CXI_RDZV_EAGER_SIZE=${FI_CXI_RDZV_EAGER_SIZE:=0}
# Look for cxi devices and use that to build our NCCL list if there
nics=""
if [ -n "`ls /dev | grep cxi`" ] ; then
    nics=`ls /dev| grep cxi | sed s,cxi,hsn,g | tr '\n' ',' | sed -e 's/,$//g'`
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=${nics}}
fi
gpus=""
if command -v nvidia-smi >/dev/null 2>&1; then
    gpus=`nvidia-smi -L | awk '{print $2}' | sed -e 's,:,,g' | tr '\n' ',' | sed -e 's/,$//g'`
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=${gpus}}
fi
# Setting this to 0 showed considerably better performance for
# the nccl all_reduce test                  
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:=0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:=PHB}
export FI_HMEM_CUDA_USE_GDRCOPY=${FI_HMEM_CUDA_USE_GDRCOPY:=1}

# This seems required for newer NGC base images to avoid issues related to
# the OMPI that was installed in the base and replaced with our OMPI.
export OMPI_MCA_pml=${OMPI_MCA_pml:="^ucx"}

# Check if the driver in the container is newer than the one on the host.
# If this happens there can be bugs, such as incorrect answers due to
# invalid messages, due to a race condition at container startup. Some
# ranks may use the driver on the host where others may use what is in
# the container. Note that if the GPUs are in default mode vs exclusive
# then this race condition should not happen.
if command -v nvidia-smi >/dev/null 2>&1; then
    # Have a cuda driver in the container so check against the host
    host_driver=`nvidia-smi --query-gpu=driver_version --format=csv | tail -n 1 | awk -F "." '{print $1}'`
    if [ -n $CUDA_DRIVER_VERSION ] ; then
	container_driver=`echo $CUDA_DRIVER_VERSION | awk -F "." '{print $1}'`
	if [ -n $host_driver ] ; then
	    if [ -n $container_driver ] ; then
		if [ $container_driver -gt $host_driver ] ; then
		    # Make sure the cuda forward compatible path is
		    # prepended in case there is a driver mismatch between the
		    # host and container
		    compat_path=/usr/local/cuda/compat/lib.real
		    export LD_LIBRARY_PATH=$compat_path:$LD_LIBRARY_PATH
		fi # end if [ $container_driver -gt $host_driver ]
	    fi # end if [ -n $container_driver ]
	fi # end if [ -n $host_driver ]
    fi # end if [ -n $CUDA_DRIVER_VERSION ]
fi # end if nvidia-smi binary exists

# The base ngc container started including a build of the AWS OFI plugin.
# When running on SS11 systems trying to use the OFI plugin built inside
# this container, our tests could seg fault when trying to init nccl
# if we do not preload our version of the plugin. Note that our LD_LIBRARY_PATH
# points to our library first but we still need to preload our lib otherwise
# we can segfault if the one from the ngc base image exists.
if [ -r /container/hpc/lib/libnccl-net.so ]; then
    export LD_PRELOAD=/container/hpc/lib/libnccl-net.so:$LD_PRELOAD
fi

# Execute what we were told to execute
exec "${@}"
