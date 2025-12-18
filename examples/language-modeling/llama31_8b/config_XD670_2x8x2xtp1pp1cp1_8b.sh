source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

export MINIBS=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False

export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=2 

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=1

export WARMUP_STEPS=80
#export WARMUP_VALIDATION_STEPS=0
export OPT_LR_DECAY_STEPS=$((MAX_STEPS-WARMUP_STEPS))

export LR=0.0004
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768


export DGXNNODES=2
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=400 #140
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
