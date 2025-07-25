### Horovod Torch Synthetic Benchmark

#### Source

```
$> git clone https://github.com/horovod/horovod.git
```

#### Steps

The following is an example command for running the PyTorch Horovod benchmark
on two nodes with four GPUs per node. Note that the TF benchmark would be
nearly identical and should just require changing the name of the image to
that of the TF container and updating the path to the TF2 benchmark.

##### SS Containerized Libraries

```
$> cd horovod
$> srun --mpi=pmix_v4 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` /path/to/sif/pytorch-ngc-hpc-dev.sif python ./examples/pytorch/pytorch_synthetic_benchmark.py --batch-size=96 --fp16-allreduce >& pt-syn-n8-ppn4.txt
```

##### Bind-mount SS Libraries

```
$> cd horovod
$> srun --mpi=pmi2 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` --bind /opt/cray/libfabric/1.15.2.0/lib64:/host/lib,/usr:/host/usr /projects/benchmarking/public/sif/pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/wrapper.sh python ./examples/pytorch/pytorch_synthetic_benchmark.py --batch-size=96 --fp16-allreduce >& pt-syn-n8-ppn4.txt
```

#### Example Output

Running the above command should produce output similar to:

```
NCCL version 2.20.5+cuda12.4
Model: resnet50
Batch size: 96
Number of GPUs: 8
Running warmup...
Running benchmark...
Iter #0: 1785.8 img/sec per GPU
Iter #1: 1787.9 img/sec per GPU
Iter #2: 1787.6 img/sec per GPU
Iter #3: 1784.7 img/sec per GPU
Iter #4: 1783.5 img/sec per GPU
Iter #5: 1785.2 img/sec per GPU
Iter #6: 1785.0 img/sec per GPU
Iter #7: 1782.4 img/sec per GPU
Iter #8: 1785.3 img/sec per GPU
Iter #9: 1784.1 img/sec per GPU
Img/sec per GPU: 1785.2 +-3.1
Total img/sec on 8 GPU(s): 14281.2 +-25.0
```
