source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh

export MINIBS=1
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False

export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=1

export WARMUP_STEPS=64
export VAL_CHECK_INTERVAL=192

export LR=0.0008

export DGXNNODES=8
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=30
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
