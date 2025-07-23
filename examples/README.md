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

Here is the list of all our examples:
- [**OSU Benchmark**] (//https://github.com/crickett-hpe/hpc-ai-envs/blob/soohoon/examples-doc/examples/osu-benchmark)
- [**NCCL Tests**] (https://github.com/crickett-hpe/hpc-ai-envs/tree/soohoon/examples-doc/examples/nccl-tests)
- [**Torch Distributed Tests**] (https://github.com/crickett-hpe/hpc-ai-envs/tree/soohoon/examples-doc/examples/torch-distributed)
- [**Horovod Tests**] (https://github.com/crickett-hpe/hpc-ai-envs/tree/soohoon/examples-doc/examples/horovod)
- [**Language Modeling**] (https://github.com/crickett-hpe/hpc-ai-envs/tree/soohoon/examples-doc/examples/language-modeling)

