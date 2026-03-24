### MPI OSU Benchmarks

The OSU Micro-Benchmarks are included in the container image and located at:

```
/container/hpc/tests/osu-micro-benchmarks
```

To evaluate MPI performance, run the individual benchmark binaries across two
nodes using `srun`.

```
$> srun --exclusive -c 72 --distribution=*:block --mpi=pmix -n 2 --ntasks-per-node=1 --cpu-bind=socket --ntasks-per-socket=1 --sockets-per-node=4 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` /projects/benchmarking/public/sif/cray-pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/wrapper.sh /container/hpc/tests/osu-micro-benchmarks/mpi/one-sided/osu_get_bw
```

This should produce output similar to:

```
# OSU MPI_Get Bandwidth Test v7.5.2
# Window creation: MPI_Win_allocate
# Synchronization: MPI_Win_flush
# Datatype: MPI_CHAR.
# Size      Bandwidth (MB/s)
1                       0.24
2                       0.47
4                       0.93
8                       1.87
16                      3.76
32                      7.70
64                     15.95
128                    33.47
256                    73.39
512                   166.74
1024                  413.91
2048                 1286.75
4096                 3254.24
8192                 7944.83
16384               17655.37
32768               23105.17
65536               23683.79
131072              23989.79
262144              24138.02
524288              24211.46
1048576             23959.64
2097152             24194.21
4194304             24248.76
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


