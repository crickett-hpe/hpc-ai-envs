### Torch Distributed Resnet Benchmark

#### Source
```
https://github.com/pytorch/pytorch/tree/main/benchmarks/distributed/ddp
wget https://raw.githubusercontent.com/pytorch/pytorch/main/benchmarks/distributed/ddp/benchmark.py
```

#### Steps

##### SS Containerized Libraries

An example command for running the benchmark on two nodes with four GPUs
per node using SS libraries built into the container:

```
$> env MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1` srun --mpi=pmix_v4 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` /path/to/sif/pytorch-ngc-hpc-dev.sif python /projects/benchmarking/public/examples/ddp/benchmark.py >& benchmark-out-n8-ppn4.txt
```

##### Bind-mount SS Libraries

An example command for running the benchmark on two nodes with four GPUs
per node using bind-mounted SS libraries:

```
$> env MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1` srun --mpi=pmi2 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` --bind /projects/benchmarking/public --bind /opt/cray/libfabric/1.15.2.0/lib64:/host/lib,/usr:/host/usr /projects/benchmarking/public/sif/pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/wrapper.sh python /projects/benchmarking/public/examples/ddp/benchmark.py >& benchmark-out-n8-ppn4.txt
```

#### Example Output

The following is example output from running the Torch distributed Resnet benchmark on two nodes with four ranks per node:

```
-----------------------------------
PyTorch distributed benchmark suite
-----------------------------------

* PyTorch version: 2.3.0a0+40ec155e58.nv24.03
* CUDA version: 12.4
* Distributed backend: nccl
* Maximum bucket size: 25MB

--- nvidia-smi topo -m ---

        GPU0   GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV6     NV6     NV6     0-71           0,5-11          4
GPU1    NV6      X      NV6     NV6     72-143         1,13-19         12
GPU2    NV6     NV6      X      NV6     144-215        2,21-27         20
GPU3    NV6     NV6     NV6      X      216-287        3,29-35         28

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

--------------------------


Benchmark: resnet50 with batch size 32

                            sec/iter    ex/sec      sec/iter    ex/sec      sec/iter    ex/sec      sec/iter    ex/sec
   1 GPUs --   no ddp:  p50:  0.023s    1392/s  p75:  0.023s    1387/s  p90:  0.023s    1386/s  p95:  0.023s    1381/s
   1 GPUs --    1M/1G:  p50:  0.024s    1322/s  p75:  0.024s    1320/s  p90:  0.024s    1312/s  p95:  0.024s    1308/s
   2 GPUs --    1M/2G:  p50:  0.025s    1256/s  p75:  0.026s    1253/s  p90:  0.026s    1250/s  p95:  0.026s    1244/s
   4 GPUs --    1M/4G:  p50:  0.025s    1255/s  p75:  0.026s    1245/s  p90:  0.026s    1237/s  p95:  0.026s    1234/s
   4 GPUs --    1M/8G:  p50:  0.025s    1264/s  p75:  0.025s    1258/s  p90:  0.026s    1242/s  p95:  0.026s    1241/s
   8 GPUs --    2M/8G:  p50:  0.026s    1233/s  p75:  0.026s    1211/s  p90:  0.027s    1191/s  p95:  0.027s    1183/s

Benchmark: resnet101 with batch size 32

                            sec/iter    ex/sec      sec/iter    ex/sec      sec/iter    ex/sec      sec/iter    ex/sec
   1 GPUs --   no ddp:  p50:  0.040s     790/s  p75:  0.041s     789/s  p90:  0.041s     786/s  p95:  0.041s     783/s
   1 GPUs --    1M/1G:  p50:  0.044s     727/s  p75:  0.044s     726/s  p90:  0.044s     724/s  p95:  0.044s     723/s
   2 GPUs --    1M/2G:  p50:  0.046s     691/s  p75:  0.047s     686/s  p90:  0.047s     680/s  p95:  0.047s     675/s
   4 GPUs --    1M/4G:  p50:  0.046s     703/s  p75:  0.046s     701/s  p90:  0.046s     694/s  p95:  0.046s     691/s
   4 GPUs --    1M/8G:  p50:  0.045s     710/s  p75:  0.045s     709/s  p90:  0.045s     708/s  p95:  0.045s     708/s
   8 GPUs --    2M/8G:  p50:  0.045s     706/s  p75:  0.046s     697/s  p90:  0.046s     696/s  p95:  0.046s     692/s

```
