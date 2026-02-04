#!/bin/bash

set -x

# This script is used to alter the default env for Bourne shells.
# This is helpful since we install HPC related tools in HPC_DIR and
# it would be good to have the ${HPC_DIR}/include be in the default
# include path for gcc, etc.
env_file=/etc/hpc-ai-env.sh
echo "#!/bin/bash" >> $env_file

hpc_cpath="export CPATH=${HPC_DIR}/include:${HPC_DIR}/include/linux:${HPC_DIR}/include/uapi:\$CPATH"
echo $hpc_cpath >> $env_file

hpc_ldpath="export LD_LIBRARY_PATH=${HPC_DIR}/lib:\$LD_LIBRARY_PATH"
echo $hpc_ldpath >> $env_file

# PMIx MCA defaults for container compatibility with host Slurm PMIx
# These disable problematic components that don't work in containers
echo "# PMIx/OMPI MCA defaults for Slurm compatibility" >> $env_file
echo "export PMIX_MCA_psec=^munge" >> $env_file
echo "export PMIX_MCA_gds=hash" >> $env_file
echo "export PMIX_MCA_ptl=^usock" >> $env_file
echo "export PMIX_SYSTEM_TMPDIR=/tmp" >> $env_file
echo "export OMPI_MCA_pml=^ucx" >> $env_file
echo "export OMPI_MCA_btl=^openib" >> $env_file

# Add the ${HPC_DIR}/lib to where ld searches
echo "${HPC_DIR}/lib" >> /etc/ld.so.conf.d/hpc-ai-env.conf

chmod 755 $env_file

# Add the command to source the env_file to /etc/bash.bashrc
echo "test -f $env_file && source $env_file" >> /etc/bash.bashrc


