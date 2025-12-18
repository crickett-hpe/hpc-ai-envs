# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

: "${LOGDIR:=./results}"
: "${RUN_ONLY_NCCL:=0}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${NPY_INDEX_DIR:=${LOGDIR}/${DATESTAMP}_npy_index}"
: "${MEM_DUMP_DIR:=${LOGDIR}/mem_dump}"
: "${EXTRA_ARGS:=""}"


# Setup directories
( umask 0002; mkdir -p "${LOGDIR}"; mkdir -p "${LOGDIR}/${NEMO_RESULTS_SUBDIR:-${DATESTAMP}}"; mkdir -p $NPY_INDEX_DIR; mkdir -p $MEM_DUMP_DIR )

_cont_mounts="${LOGDIR}:/results,${NPY_INDEX_DIR}:/npy_index,${MEM_DUMP_DIR}:/mem_dump"


if [[ "${USE_SYNTHETIC_DATA:-0}" -eq 0 ]]; then
    # this is the default case, with real data
    TOKENIZER="${DATADIR}/${MODEL_SIZE}/tokenizer"
    PREPROC_DATA="${DATADIR}/${MODEL_SIZE}"
    _cont_mounts="${_cont_mounts},${TOKENIZER}:/workspace/llm/nemo_tokenizer:ro,${PREPROC_DATA}:/preproc_data:ro"
    mounts_to_verify="PREPROC_DATA:/preproc_data"
    if [[ -n "${LOAD_CHECKPOINTS_PATH:-}" ]]; then
        _cont_mounts="${_cont_mounts},${LOAD_CHECKPOINTS_PATH}:/load_checkpoints:ro"
        mounts_to_verify="${mounts_to_verify} LOAD_CHECKPOINT_405B:/load_checkpoints/405b"
    fi
    if [[ -n "${CHECKPOINTS_DIR:-}" ]] && [[ ${RUN_ONLY_NCCL} -eq 0 ]]; then
        _cont_mounts="${_cont_mounts},${CHECKPOINTS_DIR}:/results/${NEMO_RESULTS_SUBDIR}/checkpoints"
    fi
else
    export MOCK_DATASET="True"
    export EXTRA_ARGS+=" model.tokenizer.model="
    if [[ "${VERIFY_MOUNTS:-0}" -eq 1 ]]; then
        echo "Overriding VERIFY_MOUNTS to 0 because USE_SYNTHETIC_DATA is 1"
        export VERIFY_MOUNTS=0
    fi
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    _cont_mounts="${EXTRA_MOUNTS},${_cont_mounts}"
fi

if [[ -n "${APILOG_DIR:-}" ]]; then
    APILOG_DIR=${APILOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
    mkdir -p ${APILOG_DIR}
    _cont_mounts="${_cont_mounts},${APILOG_DIR}:/logs"

    # Create JSON file for cuDNN
    JSON_MODEL_NAME="MLPERF_${MODEL_NAME}_${MODEL_FRAMEWORK}_train"
    JSON_README_LINK="${README_PREFIX}/${MODEL_NAME}/${MODEL_FRAMEWORK}/README.md"
    JSON_FMT='{model_name: $mn, readme_link: $rl, configs: {($dt): [$bs]}, sweep: {($dt): [$bs]}}'
    JSON_OUTPUT="${JSON_MODEL_NAME}.cudnn.json"
    jq -n --indent 4 --arg mn "${JSON_MODEL_NAME}" --arg rl "${JSON_README_LINK}" --arg dt "${APILOG_PRECISION}" --arg bs "${BATCHSIZE}" "$JSON_FMT" > "${APILOG_DIR}/${JSON_OUTPUT}"
fi

if [[ "${JET:-0}" -eq 1 ]]; then
    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
fi

if [[ -n "${REMOUNT_WORKDIR:-}" ]]; then
    echo 'Remounting workdir'
    _cont_mounts="${REMOUNT_WORKDIR}:/workspace/llm,${_cont_mounts}"
fi
if [[ -n "${REMOUNT_NEMO_PATH:-}" ]]; then
    echo "Remounting Nemo from ${REMOUNT_NEMO_PATH}"
    _cont_mounts="${REMOUNT_NEMO_PATH}:/opt/bignlp/NeMo,${_cont_mounts},${REMOUNT_NEMO_PATH}:/workspace/NeMo"
fi
if [ "${BREAKDOWN:-0}" -eq 1 ] && [[ -f "${DLSIM_REPORT_PATH:-}" ]]; then
    # Mount DLSim report from host filesystem if it exists
    export DLSIM_REPORT_CONT="/workspace/breakdown/dlsim_report.xlsx"
    _cont_mounts="${_cont_mounts},${DLSIM_REPORT_PATH}:${DLSIM_REPORT_CONT}"
fi
