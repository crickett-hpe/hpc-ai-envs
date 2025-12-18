#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
set -e

###########################################################################
# This script is invoked inside the container, and a copy is launched on every
# rank.
#
# This script MUST NOT have any SLURM dependences (no use of SLURM envvars)
#
# If this is a pytorch benchmark this script assumes that it is invoked with
# something (torchrun, srun+enroot, srun+slurm2pytorch, or
# mpirun+slurm2pytorch) that correctly sets the variables described at
# https://pytorch.org/docs/stable/elastic/run.html#environment-variables RANK,
# LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT
#
# To run this script interactively you must invoke it with torchrun
#
# if you need NODE_RANK you can derive it by
# NODE_RANK=$(( RANK / LOCAL_WORLD_SIZE ))
#
# If this script is for a framework that assumes mpirun (or srun --mpi), the
# envvars the envvars described at
# https://docs.open-mpi.org/en/v5.0.x/tuning-apps/environment-var.html
# OMPI_COMM_WORLD_SIZE, OMPI_COMM_WORLD_RANK, OMPI_COMM_WORLD_LOCAL_SIZE,
# OMPI_COMM_WORLD_LOCAL_RANK, OMPI_COMM_WORLD_NODE_RANK
###########################################################################

# vars that should be set by the launcher (Pytorch)
: "${RANK:?RANK not set}"
: "${LOCAL_RANK:?LOCAL_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD_SIZE:?LOCAL_WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"

[ "${DEBUG}" = "0" ] && set -x

# Vars without defaults
: "${SEED:?SEED not set}"
: "${WALLTIME:=?WALLTIME not set}"

# Vars with defaults
: "${MULTI_NODE:=''}"
: "${UNITTEST:=0}"

: "${NVTX_FLAG:=0}"
: "${NSYS_METRICS:=0}"
: "${TIME_TAGS:=0}"

: "${LOAD_CHECKPOINT:=""}"  # if set, training starts from this checkpoint (see SHARE_RERUNS effects below). Otherwise starts from scratch.
: "${SHARE_RERUNS:=0}"  # uses `shared_logs` results directory for all runs so that the checkpoints can be shared between different runs

echo "LOAD_CHECKPOINT=${LOAD_CHECKPOINT}"
# In order to share checkpoints between different runs (e.g. of a dryrun):
# 1) set ENABLE_RERUNS=1
# 2) set SHARE_RERUNS=1 so that the checkpoints subdirectory is the same for all runs
# 3) set the same LOGDIR in all runs
# 4) run training with `run.sub` or set NEMO_RESULTS_SUBDIR manually to a fixed value
# 5) run dependent slurm job
# 6) set the same SEED in all runs
# NOTE: a dryrun already meets 3) and 4) criteria.
# NOTE: if SHARE_RERUNS is set and an existing checkpoints directory is detected, LOAD_CHECKPOINT has no effect.
#       This is a convenience to avoid unsetting LOAD_CHECKPOINT when resuming from a further checkpoint
# NOTE: ENABLE_RERUNS and LOAD_CHECKPOINT variables are orthogonal, i.e. it makes sense to turn on or off ENABLE_RERUNS
# either when LOAD_CHECKPOINT is set (starting from iteration 4000) or not (starting from scratch).

: "${USE_DIST_OPTIMIZER:=True}"
: "${CKPT_EVERY_VALIDATION:=False}"
: "${WALLTIME_EXIT_MINUTES:=0}"
: "${EXTRA_ARGS:=$@}"

[[ "${DEBUG}" ]] && echo RANK="${RANK}", LOCAL_RANK="${LOCAL_RANK}", MASTER_ADDR="${MASTER_ADDR}", MASTER_PORT="${MASTER_PORT}", WORLD_SIZE="${WORLD_SIZE}", UCX_NET_DEVICES="${UCX_NET_DEVICES}", NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}", NCCL_IB_HCA="${NCCL_IB_HCA}", NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}", NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING}", SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING="${SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING}", UCX_VFS_ENABLE="${UCX_VFS_ENABLE}"

if [[ "$RANK" -eq 0 ]]; then
    env > /results/container-env-"$SLURM_JOB_ID".log
fi

if [ "${NEMO_RESULTS_IN_TMP:-0}" -eq 1 ]; then
  readonly _explicit_log_dir=/tmp/${NEMO_RESULTS_SUBDIR:-""}
else
  readonly _explicit_log_dir=/results/${NEMO_RESULTS_SUBDIR:-""}
fi

if [ -n "${LOAD_CHECKPOINT}" ]; then
  if [ ${SHARE_RERUNS:-0} -eq 1 ] && [ -d "${_explicit_log_dir}/checkpoints" ] && [ -n "$(ls -A "${_explicit_log_dir}/checkpoints")" ]
  then
    [[ "$RANK" -eq 0 ]] && echo \
      "Detected a shared rerun." \
      "Resuming from previous run checkpoint stored in ${_explicit_log_dir}/checkpoints" \
      "instead of the initial checkpoint ${LOAD_CHECKPOINT}"
      unset LOAD_CHECKPOINT
  fi
else
    unset LOAD_CHECKPOINT
fi

if [ -n "${NEMO_RESULTS_SUBDIR}" ]; then
  EXTRA_ARGS+=" exp_manager.explicit_log_dir=\"${_explicit_log_dir}\""
fi

if [ "${TRAIN_ONLY:-0}" -eq 1 ]; then
  EXTRA_ARGS+=" data_prefix@model.data.data_prefix=train_only_c4"
elif [ "${USE_SYNTHETIC_DATA:-0}" -eq 1 ]; then
  EXTRA_ARGS+=" data_prefix@model.data.data_prefix=synthetic"
fi

[ "$INTERLEAVED_PIPELINE" == "0" ] && export INTERLEAVED_PIPELINE=null

# Get rank to hostname mapping
if [ "$LOCAL_RANK" -eq 0 ]; then
  echo "Hello from: $(hostname)"
fi

if [ "$CKPT_EVERY_VALIDATION" = True ]; then
  EXTRA_ARGS+=" exp_manager.checkpoint_callback_params.every_n_epochs=1"
  EXTRA_ARGS+=" exp_manager.checkpoint_callback_params.save_last=False"
fi

if [ "${WALLTIME_EXIT_MINUTES:-0}" -gt 0 ]; then
  [ "${NEXP:-1}" -gt 1 ] && echo "Warning: NEXP>1 and WALLTIME_EXIT_MINUTES>0 makes little sense (max_time for each run is set based on total WALLTIME)."

  max_time_minutes=$(( WALLTIME - WALLTIME_EXIT_MINUTES))
  [[ "$RANK" -eq 0 ]] && echo "Setting max_time to $max_time_minutes minutes"

  EXTRA_ARGS+=" +trainer.max_time=00:00:${max_time_minutes}:00"
fi

if [ "${PRINT_CONFIG_ONLY:-False}" = True ]; then
  EXTRA_ARGS+=" -c job --resolve"
fi

if [ ${NVTX_FLAG} -gt 0 ]; then
  NSYSCMD=" nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --output ${NSYS_PREFIX}${SLURM_PROCID}${NSYS_SUFFIX}"
  if [ ${NSYS_METRICS} -gt 0 ]; then
    NSYSCMD="${NSYSCMD} --gpu-metrics-devices=${LOCAL_RANK} --gpu-metrics-set=${NSYS_METRICS_SET} --gpu-metrics-frequency=50000"
  fi
fi

# run benchmark
[[ "$RANK" -eq 0 ]] && echo "running LLM benchmark"

declare -a CMD

if [[ ${LOCAL_WORLD_SIZE} -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( ${NSYSCMD} 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [[ "$RANK" -eq 0 ]]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

if [ "${HANG_MONITOR_TIMEOUT-0}" -gt 0 ] && [ "${ATTEMPT_CUDA_GDB_CORE_DUMP-0}" == "1" ]; then
      mkdir -p "${CUDA_COREDUMP_PIPE_DIR}"
fi

[[ "$RANK" -eq 0 ]] && echo "Extra args: $EXTRA_ARGS"

CUDA_COREDUMP_FILE=/results/llama31.%h.%p ${LOGGER:-} ${BINDCMD:-} ${CMD[@]} /workspace/llm/pretrain.py \
	$EXTRA_ARGS \
	; ret_code=$?

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi
