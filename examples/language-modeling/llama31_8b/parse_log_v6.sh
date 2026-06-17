#!/bin/bash

LOG=$1
REV=$2

# Vars without defautls
: "${LOG:?LOG is not set}"

# Vars with defaults
: "${REV:=1}"

#LOGFILE=$1
#ENVFILE=$2
##LOGFILE=llama31_8b-$JOBID.log
#if [ -z $LOGFILE ]; then
#  LOGFILE=slurm-$JOBID.out
#fi
#
#ENVFILE="./results/container-env-${JOBID}.log"
#IMAGE=$(grep "\.sif\$" $LOGFILE | awk '{print $NF}')

if [ ! -f $LOG ]; then
    echo "LOG $LOG does not exist"
    exit 1
fi

LOGBASE=$(grep "LOGBASE=" $LOG | awk -F "=" '{print $NF}')
LOGFILE="results/${LOGBASE}_${REV}.log"
SLURM_JOBID=$(grep "DLPAL" $LOGFILE | tail -1 | awk '{print $3}')
ENVFILE="results/${SLURM_JOBID}_${REV}.env"
IMAGE=$(grep "CONT=" $ENVFILE | awk -F '=' '{print $NF}')

NCCL_TEST=$(grep "Avg bus bandwidth" $LOG)

NODES=$(grep "DGXNNODES" $ENVFILE | sed 's/"//g' | awk -F '=' '{print $NF}')
NGPUS=$(grep "DGXNGPU" $ENVFILE | sed 's/"//g' | awk -F '=' '{print $NF}')
MINIBS=$(grep "MINIBS" $ENVFILE | sed 's/"//g' | awk -F '=' '{print $NF}')

## Hyper-Parameter
GBS=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "global_batch_size") | .value')
GAS=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "gradient_accumulation_steps") | .value')
MSL=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "max_sequence_length") | .value')
EVS=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "eval_samples") | .value')
TRS=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "train_samples") | .value')
MXS=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "max_steps") | .value')
TGA=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "target_accuracy") | .value')
LNR=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "opt_base_learning_rate") | .value')

TP=$(grep -m 1 "tensor_model_parallel_size=" $LOGFILE | awk -F '=' '{print $NF}')
PP=$(grep -m 1 "pipeline_model_parallel_size=" $LOGFILE | awk -F '=' '{print $NF}')
CP=$(grep -m 1 "context_parallel_size=" $LOGFILE | awk -F '=' '{print $NF}')
VP=$(grep -m 1 "virtual_pipeline_model_parallel_size=" $LOGFILE | awk -F '=' '{print $NF}')
SP=$(grep " sequence_parallel_size" $LOGFILE | awk -F '=' '{print $NF}')

TP_COMM_OVERLAP=$(grep 'tp_comm_overlap=' $LOGFILE | awk '{print $NF}')
FP8=$(grep ' fp8=' $LOGFILE | awk '{print $NF}')
FP8_DPA=$(grep ' fp8_dot_product_attention=' $LOGFILE | awk '{print $NF}')

OPT_LR_RATE_DECAY_STEPS=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "opt_learning_rate_decay_steps") | .value')
EVAL_ACCURACY=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "eval_accuracy") | .value' | tail -1)
STEP=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "eval_stop") | .metadata.step' | tail -1)

HOSTS=$(grep -m1 :::DLPAL $LOGFILE | awk '{ print $7; }' | tr -d '\n')
RUN_TIME=$(grep -e run_stop -e run_start $LOGFILE | awk '{ print $6; }' | awk -F '[, ]' '{print $1}' | awk 'NR==1{s=-$1;next}{s+=$1}END{print s}' | awk '{ print $1/1000/60; }')

AVG_TRAIN_STEP_TIME=$(grep ':::MLLOG' $LOGFILE | sed 's/^.*:::MLLOG //' | jq -r 'select(.key == "tracked_stats" and (.value.train_step_time != null)) | .value.train_step_time' | awk ' { sum += $1; count +=1; } END { if (count > 0) { printf(sum/count)} else { print "NaN"}}')

echo "LOGFILE: $LOGFILE"
echo "LOGBASE: $LOGBASE"
echo "ENV_FILE: $ENVFILE"
echo "SLURM_JOB: $SLURM_JOBID"
echo "IMAGE: $IMAGE"

echo ""
echo "DGXNNODES: $NODES"
echo "DGXNGPU: $NGPUS"
echo "MINIBS: $MINIBS"
echo "TP: $TP PP: $PP CP: $CP VP: $VP"

echo ""
echo "Hyper-Parameter:"
echo "global_batch_size: $GBS"
echo "gradient_accumulation_steps: $GAS"
echo "max_sequence_length: $MSL"
echo "eval_samples: $EVS"
echo "train_samples: $TRS"
echo "max_steps: $MXS"
echo "target_accuracy: $TGA"
echo "learning_rate: $LNR"

echo ""
echo $TP_COMM_OVERLAP
echo $FP8
echo $FP8_DPA

echo ""
echo "opt_learning_rate_decay_steps: $OPT_LR_RATE_DECAY_STEPS"
echo "eval_accuracy: $EVAL_ACCURACY"
echo "step: $STEP"

echo ""
echo "AVERAGE_TRAIN_STEP_TIME: $AVG_TRAIN_STEP_TIME"
echo "$MSL * $GBS / $AVG_TRAIN_STEP_TIME / ($NODES * 4)"
TSG=$(echo "scale=2; $MSL * $GBS / $AVG_TRAIN_STEP_TIME / ($NODES * $NGPUS)" | bc)
echo "$TSG tokens/sec/GPU"

echo ""
echo "${HOSTS}: $RUN_TIME min"
echo ""


#echo "0: gradient_accumulation_steps: $GAS"
#echo "0: total number of epochs: $EPOCHS"
#echo "scale=2; $MINIBS * $GAS * $MAX_STEPS / $EPOCHS"
#EXAMPLES=$(echo "scale=2; $MINIBS * $GAS * $MAX_STEPS / $EPOCHS" | bc)
#echo "0: total number of examples: $EXAMPLES"
