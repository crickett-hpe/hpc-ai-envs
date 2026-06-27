#config_EX4000_4x4x1xtp1pp1cp1_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh
source $(dirname ${BASH_SOURCE[0]})/config_net.sh

export MLPERF_CLUSTER_NAME="HPE Cray EX4000"
export GPU_ARCH="gh200"
export BIND="-c 71 --cpu_bind=verbose"

export MINIBS=1
export MICRO_BATCH_SIZE=1

export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export LR=0.001
export WARMUP_STEPS=80
export VAL_CHECK_INTERVAL=256

export OPT_LR_DECAY_STEPS=$MAX_STEPS

export DGXNNODES=4
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=240
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export FP8_DPA=False
