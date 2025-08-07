### MPI OSU Benchmarks

The following is an example of how to test MPI to verify that the
MPI inside the container can correctly use the Cray libfabric/cxi. Note that
the container used in this example pulled in copies of the Cray libfabric/cxi
as part of the final build step (ie, Dockerfile-ss) so the command does
not bind-mount in the Cray libfabric/cxi at container runtime.

```
$> wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz
$> tar zxf osu-micro-benchmarks-7.4.tar.gz
$> cd osu-micro-benchmarks-7.4/
$> singularity shell --nv --bind $TMPDIR --bind `pwd` /projects/benchmarking/public/sif/cray-pytorch-ngc-hpc-dev.sif
Singularity>  ./configure CC=`which mpicc` CXX=`which mpicxx` --prefix=`pwd`
Singularity> make
Singularity> exit
$> srun --exclusive -c 72 --distribution=*:block --mpi=pmi2 -n 2 --ntasks-per-node=1 --cpu-bind=socket --ntasks-per-socket=1 --sockets-per-node=4 --gpus=8 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` /projects/benchmarking/public/sif/cray-pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/wrapper.sh ./c/mpi/one-sided/osu_get_bw
```

This should produce output similar to:

```
# OSU MPI_Get Bandwidth Test v7.4
# Window creation: MPI_Win_allocate
# Synchronization: MPI_Win_flush
# Datatype: MPI_CHAR.
# Size      Bandwidth (MB/s)
1                       1.03
2                       2.10
4                       4.20
8                       8.38
16                     16.78
32                     33.57
64                     65.82
128                   132.22
256                   254.89
512                   510.63
1024                 1019.58
2048                 2036.45
4096                 4045.65
8192                 7563.77
16384               11773.70
32768               15233.84
65536               18976.11
131072              21131.24
262144              22495.07
524288              23224.79
1048576             23590.77
2097152             23807.66
4194304             23903.40
```


### Notes

A common requirement for each of these tests that use these
Singularity containers is that the Cray libfabric/cxi need to be made
available to the container when running. This is done by the following
bind mount option to the singularity run commands:

```
--bind /opt/cray/libfabric/1.15.2.0/lib64:/host/lib,/usr:/host/usr
```

The destination mount points of `/host/lib` and `/host/usr` are
important because these are the locations where the container
entrypoint script will look for the Cray libfabric/cxi in order to
automatically swap them in place of the open-source libfabric built
into the container. This is what is needed to enable SS11 to be
utilized by NCCL inside of the container.


This repository is based off the following branch of the Determined-AI
task environments repository:

https://github.com/determined-ai/environments/tree/cleanup-hpc-build


