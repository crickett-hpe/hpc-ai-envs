# HPC AI Environments

The following describes how to build Docker containers for PyTorch/TF
that target Cray EX HPC systems with NVIDIA or AMD GPUs and the
SlingShot (SS) interconnect and enable optimal use of the SS network
for NCCL/RCCL/MPI. The NCCL/RCCL support leverage the AWS OFI NCCL
plugin. It is intended that the Docker containers built using this
repository are converted to a Singularity/Apptainer container and
executed via a work load manager (WLM) such as Slurm on the HPC
system.

The images created by this repository use either the NGC or AMD InfinityHub
images as their base and then add the required bits for enabling SS support.
The SS support means installing a new version of OMPI/MPICH that can use
libfabric, Horovod targeting the given OMPI/MPICH, and the AWS OFI NCCL
plugin to enable NCCL/RCCL over libfabric.

## Prerequisites

* Docker or podman
    - Used to build the docker images
* Singularity/Apptainer
    - Used to convert the docker image to a Singularity/Apptainer sif and
      run the application

If Singularity/Apptainer are not available on the system, it can be
installed by a normal user using the following:

```
$> curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh |  bash -s - install-dir
```

After `apptainer` is installed it can be executed by adding
`install-dir/bin` to the `PATH`. For more information see the
following `apptainer` documentation where it discusses doing an
unprivileged install:

https://apptainer.org/docs/admin/main/installation.html


## Building Images

The build process expects to find `docker` in the default `PATH`. If
`docker` is not installed on the system, it is also sufficient to have
`podman` installed and simply make a symbolic link to `podman` that is
named `docker`. For example:

```
$> ln -s `which podman` $HOME/bin/docker
```

After cloning this repository and ensuring `docker` exists in the
default `PATH`, an image can be built by running `make` on for the
desired target. For example, to build the latest PyTorch image using
the NGC base image, a command similar to the following could be used:

```
$> make ngc >& ngc.txt
$> make rocm >& rocm.txt
```

If successful you should see the resulting docker image:

```
$> docker images | grep pytorch-ngc
localhost/cray/pytorch-ngc-hpc-dev-ss  053a634     47a5b67250f3  28 minutes ago     18.4 GB
localhost/cray/pytorch-ngc-hpc-dev     053a634     cef59f3db8a5  34 minutes ago     18.3 GB
localhost/cray/pytorch-ngc-dev         053a634     3d66151473fc  About an hour ago  18.2 GB
```

By default, the build will include MPI and OFI for targeting the Cray
HPC system. This can be disabled by specifying WITH_MPI=0 and
WITH_OFI=0 to `make`.

### Slingshot Support

By default, the build will clone the required libraries for enabling
support for the Slingshot network, including the Cray `libcxi` and
`libfabric` libraries. The GitHub repositories are cloned and built inside
the container to ensure proper version matching with required libraries.
Further, by building these libraries into the container, there should be
no need to pull in (ie, bind-mount) these or other libraries from the
host Cray system in order to optimally use the SS network.

Once the Docker image is built you can convert it to a Singularity/Apptainer
image using commands similar to the following:

```
$> docker save -o pytorch-ngc-hpc-dev-ss-053a634.tar cray/pytorch-ngc-hpc-dev-ss:053a634
$> singularity build pytorch-ngc-hpc-dev-ss-053a634.sif docker-archive:/path/to/docker/tarball/pytorch-ngc-hpc-dev-ss-053a634.tar
```

Alternatively, to generate a tar file of the image, you can add the `tar` target:

```
$> make ngc tar >& ngc.txt
$> make rocm tar >& rocm.txt
```

Or you can directly create the singularity/Apptainer image, and clean up the tar file
afterwards:

```
$> make ngc sif RM_SIF_TAR=1 >& ngc.txt
$> make rocm sif RM_SIF_TAR=1 >& rocm.txt
```

### User Specified Images

Users can specify which base image to use to simply have the required
SS related libs (ie, CXI, OFI and AWS OFI) added to their
containers. This should make it easier for users to update images that
have the desired packages to work well with NCCL over SS. To enable
this, the user can specify the build target `ngc` or `rocm` and
pass the base image name to use with the argument
USER_IMAGE. For example, the following could be used to build
the HPC version of the base image that will compile in the CXI,
libfabric and AWS OFI plugin:

```
make ngc USER_IMAGE=cray/ngc-24.11-py3-pt:5a96988
```

If successful, this will create the new `-hpc` version of the image:

```
Successfully tagged localhost/cray/ngc-24.11-py3-pt-hpc:5a96988-amd64
```

