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

Optionally, the build command can be given a pointer to a local
directory that contains copies of the Cray lib{fabric*|cxi*} required to
optimally use the Cray SS network. These libraries can be built into
the docker image to make it easier for users to run their applications
that leverage the SS network. Note that this is optional. If the user
does not specify this directory for the build step then the proper
directories containing the Cray lib{fabric*|cxi*} can be provided at
container runtime. In this case, the libraries will be bind-mounted
into the container at runtime at a known location and the entrypoint
script included in the container will update the LD_LIBRARY_PATH to
utilize these libraries and enable optimal SS performance. Examples of how
to specify the Cray lib{fabric*|cxi*} at runtime are provided below.

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
$> make build-pytorch-ngc >& build-pytorch-ngc.txt
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

Alternatively, you can specify the Cray libraries to use, including
those that are on your host system. If you specify the option
`WITH_SS11=1`, the build will search for the Cray libfabric/cxi
libraries and, if they are found, will include them into the built
container and modify the `LD_LIBRARY_PATH` to point to them.

If the Cray libfabric/cxi libraries are not installed in an expected location,
or if you want to use a specific version, you can pass the locations to
`make` using the variables `CRAY_LIBFABRIC_DIR` and `CRAY_LIBCXI_DIR`.
For example, you could specify the locations to the build with a command
similar to:

```
$> make build-pytorch-ngc CRAY_LIBFABRIC_DIR=/lus/scratch/username/ss11-libs/libfabric-1.18.2 CRAY_LIBCXI_DIR=/lus/scratch/username/ss11-libs/libcxi-1.5
```

Note that if you want to copy the Cray libraries to a single directory
and provide the name of that directory to the build you can do that
with a command similar to:

```
$> make build-pytorch-ngc HPC_LIBS_DIR="ss11-libs" >& build-pytorch-ngc-ss.txt
```

Inside the `ss11-libs` directory you would have the required Cray libraries,
such as:

```
$> ls ss11-libs
libcxi.la  libcxi.so.1      libcxiutils.la  libcxiutils.so.0      libfabric.a   libfabric.so.1
libcxi.so  libcxi.so.1.5.0  libcxiutils.so  libcxiutils.so.0.0.0  libfabric.so  libfabric.so.1.18.2
```

If successful, that build will result in a Docker image ending with the
`-ss` to signify that the SS libraries were built into the
container. For example, in the Docker images listed above the image
`cray/pytorch-ngc-hpc-dev-ss:abcdef` will have the SS images copied
into it so that the user does not need to specify the locations at
container runtime. See the `Dockerfile-ss` for more information on how
this works.

Once the Docker image is built you can convert it to a Singularity/Apptainer
image using commands similar to the following:

```
$> docker save -o pytorch-ngc-hpc-dev-ss-053a634.tar cray/pytorch-ngc-hpc-dev-ss:053a634
$> singularity build pytorch-ngc-hpc-dev-ss-053a634.sif docker-archive:/path/to/docker/tarball/pytorch-ngc-hpc-dev-ss-053a634.tar
```

### User Specified Images

Users can specify which base image to use to simply have the required
SS related libs (ie, CXI, OFI and AWS OFI) added to their
containers. This should make it easier for users to update images that
have the desired packages to work well with NCCL over SS. To enable
this, the user can specify the build target `build-user-spec-ngc` and
pass the base image name to use with the argument
USER_NGC_BASE_IMAGE. For example, the following could be used to build
the HPC version of the base image that will compile in the CXI,
libfabric and AWS OFI plugin:

```
make build-user-spec-ngc USER_NGC_BASE_IMAGE=cray/ngc-24.11-py3-pt:5a96988
```

If successful, this will create the new `-hpc` version of the image:

```
Successfully tagged localhost/cray/ngc-24.11-py3-pt-hpc:5a96988
```

## Examples

The following examples illustrate how to run various AI benchmarks
using the PyTorch/TF containers on a SS system. These tests show both
methods of running on SS: using the container that build the Cray
`libcxi` and `libfabric` into the container and alternatively,
bind-mount in the necessary Cray lib{fabric|cxi} at container runtime
to the expected mount points for the container entrypoint script.

The bind-mount option is done by using binds similar to:

```
--bind /opt/cray/libfabric/1.15.2.0/lib64:/host/lib,/usr:/host/usr
```

The first mount:

```
/opt/cray/libfabric/1.15.2.0/lib64:/host/lib
```

will bind-mount in the directory containing the Cray libfabric* `lib64`
directory to the `/host/lib` directory expected by the container
entrypoint script. Note that the source directory may vary on the host
system but the `/host/lib` is the expected destination.

The second mount:

```
/usr:/host/usr
```

Is necessary for the Cray libcxi* as well as other libraries that the
libcxi* may be dependent upon (e.g., libjson-c.so.3).

Note: if the container is built with the `HPC_LIBS_DIR` option and a
directory containing the Cray lib{fabric*|cxi*} is provided then these
bind mounts should not be necessary.

TODO: Provide versions of these examples that do not include the Cray
bind-mounts but instead use the `-ss` version of the image.

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


### llama-2 7b HuggingFace

#### Source
```
git clone https://github.com/huggingface/transformers.git
```

#### Steps

The cloned version of `run_clm.py` needs to have a few modifications
before it can run to remove references to packages not installed in
the container, etc. Other changes are also done to enable using
FlashAttention2 following instructions from:

https://huggingface.co/docs/transformers/perf_infer_gpu_one

The following diff command shows the changes made
to the original version to create the modified version used in this
example.

```
diff --git a/examples/pytorch/language-modeling/run_clm.py b/examples/pytorch/language-modeling/run_clm.py
index 794bb5f1c..82118f64b 100755
--- a/examples/pytorch/language-modeling/run_clm.py
+++ b/examples/pytorch/language-modeling/run_clm.py
@@ -45,17 +45,18 @@ from transformers import (
     Trainer,
     TrainingArguments,
     default_data_collator,
-    is_torch_xla_available,
+    is_torch_tpu_available,
+#    is_torch_xla_available,
     set_seed,
 )
-from transformers.testing_utils import CaptureLogger
+#from transformers.testing_utils import CaptureLogger
 from transformers.trainer_utils import get_last_checkpoint
 from transformers.utils import check_min_version, send_example_telemetry
 from transformers.utils.versions import require_version
 
 
 # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
-check_min_version("4.45.0.dev0")
+#check_min_version("4.45.0.dev0")
 
 require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
 
@@ -149,6 +150,12 @@ class ModelArguments:
             )
         },
     )
+    use_flash_attention_2: bool = field(
+        default=True
+    )
+    device_map: Optional[str] = field(
+        default='auto'
+    )
 
     def __post_init__(self):
         if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
@@ -312,7 +319,7 @@ def main():
             cache_dir=model_args.cache_dir,
             token=model_args.token,
             streaming=data_args.streaming,
-            trust_remote_code=model_args.trust_remote_code,
+#            trust_remote_code=model_args.trust_remote_code,
         )
         if "validation" not in raw_datasets.keys():
             raw_datasets["validation"] = load_dataset(
@@ -434,6 +441,7 @@ def main():
             trust_remote_code=model_args.trust_remote_code,
             torch_dtype=torch_dtype,
             low_cpu_mem_usage=model_args.low_cpu_mem_usage,
+            use_flash_attention_2=True,
         )
     else:
         model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
@@ -458,14 +466,15 @@ def main():
     tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
 
     def tokenize_function(examples):
-        with CaptureLogger(tok_logger) as cl:
-            output = tokenizer(examples[text_column_name])
-        # clm input could be much much longer than block_size
-        if "Token indices sequence length is longer than the" in cl.out:
-            tok_logger.warning(
-                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
-                " before being passed to the model."
-            )
+        # with CaptureLogger(tok_logger) as cl:
+        #     output = tokenizer(examples[text_column_name])
+        # # clm input could be much much longer than block_size
+        # if "Token indices sequence length is longer than the" in cl.out:
+        #     tok_logger.warning(
+        #         "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
+        #         " before being passed to the model."
+        #     )
+        output = tokenizer(examples[text_column_name])
         return output
 
     with training_args.main_process_first(desc="dataset map tokenization"):
@@ -589,10 +598,14 @@ def main():
         tokenizer=tokenizer,
         # Data collator will default to DataCollatorWithPadding, so we change it.
         data_collator=default_data_collator,
-        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
+        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
         preprocess_logits_for_metrics=preprocess_logits_for_metrics
-        if training_args.do_eval and not is_torch_xla_available()
+        if training_args.do_eval and not is_torch_tpu_available()
         else None,
+        # compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
+        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
+        # if training_args.do_eval and not is_torch_xla_available()
+        # else None,
     )
 
     # Training
@@ -648,9 +661,9 @@ def main():
         trainer.create_model_card(**kwargs)
 
 
-def _mp_fn(index):
-    # For xla_spawn (TPUs)
-    main()
+# def _mp_fn(index):
+#     # For xla_spawn (TPUs)
+#     main()
 
 
 if __name__ == "__main__":
```

A command similar to the following can be used to run the modified
`run_clm.py`. Note that you must provide your own HuggingFace token to
enable pulling the model, etc, in the command below in place of
`--token=<YOUR_HF_TOKEN_HERE>`. Further, the `ds-wrapper.sh` mentioned below
is a simple wrapper that defines a few relevant `HF` environment variables
and then executes the program passed to it. Some of the variables the
example `ds-wrapper.sh` set are:

```
export HF_DATASETS_CACHE=/lustre/scratch/username/hfcache
export TRANSFORMERS_CACHE=/lustre/scratch/username/hfcache
export HF_MODULES_CACHE=/lustre/scratch/username/hfcache
export HF_DATASETS_IN_MEMORY_MAX_SIZE=200000000
```

The example below runs on two nodes and uses four GPUs per node.

##### SS Containerized Libraries

```
$> env MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1` srun --mpi=pmix_v4 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind $TMPDIR --bind $HOME --bind `pwd` /path/to/sif/pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/llama-2/ds-wrapper.sh python /projects/benchmarking/public/examples/llama-2/transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext --dataset_config_name wikitext-2-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir `pwd`/output --overwrite_output_dir --token=<YOUR_HF_TOKEN_HERE> --block_size 4096 --torch_dtype=bfloat16 --bf16=True --deepspeed=/projects/benchmarking/public/examples/llama-2/ds_config.json --gradient_checkpointing=True >& out-n8-ppn4.txt
```

##### Bind-mount SS Libraries

```
$> env MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1` srun --mpi=pmi2 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` --bind /opt/cray/libfabric/1.15.2.0/lib64:/host/lib,/usr:/host/usr /projects/benchmarking/public/sif/pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/llama-2/ds-wrapper.sh python /projects/benchmarking/public/examples/llama-2/transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext --dataset_config_name wikitext-2-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir `pwd`/output --overwrite_output_dir --token=<YOUR_HF_TOKEN_HERE> --block_size 4096 --torch_dtype=bfloat16 --bf16=True --deepspeed=/projects/benchmarking/public/examples/llama-2/ds_config.json --gradient_checkpointing=True >& out-n8-ppn4.txt
```

#### Example Output

Using the command above should give output similar to:

```
***** train metrics *****
  epoch                    =        3.0
  train_loss               =     1.6149
  train_runtime            = 0:02:57.89
  train_samples            =        679
  train_samples_per_second =      11.45
  train_steps_per_second   =      0.185
```

An estimated TFLOPs can be calculated using the following formula:

```
(train_samples_per_second * 6 * model_size * sequence_length)
```

In the case of llama-2-7b used above the model size is 7b, the sequence_length from above was 4096, and the 6
represents number of operations per weight, which gives:

```
11.45 * 6 * 7b * 4096 / 1t ==> (11.45 * 6 * 7 * 4096) / 1000 = 1969.7664 TFLOPs
```

For the TFLOPs per GPU simply divide the total by the number of GPUs used. For this run there were 8 total GPUs on
the 2 nodes:

```
((11.45 * 6 * 7 * 4096) / 1000) / 8) = 246.2 TFOPs per GPU
```

Note that increasing the `--per_device_train_batch_size` to 16 got closer to 262 TFLOPs per GPU and scaled well up to
8 nodes. The following are sample TFLOPs seen running with a batch size of 16 on 1 -> 8 nodes with 4 GPUs per node:

```
out-n4-ppn4.txt:  train_samples_per_second =      5.864
out-n8-ppn4.txt:  train_samples_per_second =     12.229
out-n16-ppn4.txt:  train_samples_per_second =     24.423
out-n32-ppn4.txt:  train_samples_per_second =     48.651
```

which equates to:
```
N1xppn4: 252.2 TFLOP/s per GPU
N2xppn4: 263.0 TFLOP/s per GPU
N4xppn4: 262.6 TFLOP/s per GPU
N8xppn4: 261.5 TFLOP/s per GPU
```

The runs on a single node had memory pressure causing more torch
memory flushes which impacted performance.


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


