source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=56
export TENSOR_MODEL_PARALLEL=2
export PIPELINE_MODEL_PARALLEL=8
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export MICRO_BATCH_SIZE=1
export NVTE_DPA_FP8_RECIPE="F16"

export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True
export USE_TE_OPS=False

export MAX_STEPS=700

export DGXNNODES=128
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=120
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# disable coredump due to hang at teardown
export ATTEMPT_CUDA_GDB_CORE_DUMP=0
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
export CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=0
export CUDA_COREDUMP_SHOW_PROGRESS=0
export SEGMENT=16
