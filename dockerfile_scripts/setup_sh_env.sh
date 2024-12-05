#!/bin/bash

set -x

# This script is used to alter the default env for Bourne shells.
# This is helpful since we install HPC related tools in HPC_DIR and
# it would be good to have the ${HPC_DIR}/include be in the default
# include path for gcc, etc.
env_file=/etc/hpc-ai-env.sh
echo "#!/bin/bash" >> $env_file

hpc_cpath="export CPATH=${HPC_DIR}/include:${HPC_DIR}/include/linux:${HPC_DIR}/include/uapi"
echo $hpc_cpath >> $env_file

hpc_ldpath="export LD_LIBRARY_PATH=${HPC_DIR}/lib:$LD_LIBRARY_PATH"
echo $hpc_ldpath >> $env_file

# Add the ${HPC_DIR}/lib to where ld searches
echo "${HPC_DIR}/lib" >> /etc/ld.so.conf.d/hpc-ai-env.conf

chmod 755 $env_file

# Add the command to source the env_file to /etc/bash.bashrc
echo "test -f $env_file && source $env_file" >> /etc/bash.bashrc


