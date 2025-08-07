### NCCL Tests

#### Source
```
git clone https://github.com/NVIDIA/nccl-tests.git
```

#### Steps

The nccl-tests need to be compiled using the MPI/nccl inside the container:

```
$> cd nccl-tests
$> singularity shell --nv --bind $TMPDIR --bind `pwd` /projects/benchmarking/public/sif/pytorch-ngc-hpc-dev.sif
Singularity> make MPI=1 MPI_HOME=/container/hpc
Singularity> exit
```

##### SS Containerized Libraries

An example command for running the all_reduce test on two nodes using
four GPUs per node and the SS libraries built into the container:

```
$> cd nccl-tests
$> srun --exclusive -c 72 '--distribution=*:block' --mpi=pmix_v4 -n 512 --ntasks-per-node=4 --cpu-bind=socket --ntasks-per-socket=1 --sockets-per-node=4 singularity run --nv --bind $TMPDIR --bind $HOME --bind `pwd` /path/to/sif/pytorch-ngc-hpc-dev.sif ./build/all_reduce_perf -b 1M -e 8G -f 2 -n 100 >& out-n$SLURM_NTASKS-ppn4.txt
```

##### Bind-mount SS Libraries

An example command for running the all_reduce test on two nodes using
four GPUs per node using SS libraries bind-mounted into the container at
runtime:

```
$> cd nccl-tests
$> env NCCL_ALGO=Tree srun --exclusive -c 72 --distribution=*:block --mpi=pmi2 -n 8 --ntasks-per-node=4 --cpu-bind=socket --ntasks-per-socket=1 --sockets-per-node=4 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` --bind /opt/cray/libfabric/1.15.2.0/lib64:/host/lib,/usr:/host/usr /projects/benchmarking/public/sif/pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/wrapper.sh ./build/all_reduce_perf -b 1M -e 8G -f 2 -n 100 >& out-n8-ppn4.txt
```


#### Example Output

```
# nThread 1 nGpus 1 minBytes 1048576 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
NCCL version 2.20.5+cuda12.4
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
     1048576        262144     float     sum      -1    105.9    9.90   17.33      0    105.7    9.92   17.37      0
     2097152        524288     float     sum      -1    142.1   14.76   25.83      0    140.4   14.94   26.14      0
     4194304       1048576     float     sum      -1    177.1   23.68   41.44      0    176.7   23.74   41.55      0
     8388608       2097152     float     sum      -1    280.4   29.92   52.36      0    273.4   30.69   53.70      0
    16777216       4194304     float     sum      -1    413.6   40.56   70.98      0    412.0   40.72   71.26      0
    33554432       8388608     float     sum      -1    631.1   53.17   93.04      0    629.5   53.30   93.28      0
    67108864      16777216     float     sum      -1   1074.3   62.47  109.32      0   1074.1   62.48  109.34      0
   134217728      33554432     float     sum      -1   1866.5   71.91  125.84      0   1861.1   72.12  126.20      0
   268435456      67108864     float     sum      -1   3331.2   80.58  141.02      0   3351.4   80.10  140.17      0
   536870912     134217728     float     sum      -1   6278.7   85.51  149.64      0   6287.4   85.39  149.43      0
  1073741824     268435456     float     sum      -1    12174   88.20  154.35      0    12167   88.25  154.44      0
  2147483648     536870912     float     sum      -1    23985   89.53  156.68      0    23959   89.63  156.86      0
  4294967296    1073741824     float     sum      -1    47553   90.32  158.06      0    47551   90.32  158.07      0
  8589934592    2147483648     float     sum      -1    94752   90.66  158.65      0    94730   90.68  158.69      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 103.965 
#

```

