# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${FROM_IMAGE_NAME}

# Document build setup
ARG FROM_IMAGE_NAME
ENV CUSTOM_FROM_IMAGE_NAME ${FROM_IMAGE_NAME}

# Custom libraries version
WORKDIR /workspace/

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

## 0. Pytorch Checkpoint size patch
WORKDIR /workspace/pytorch
COPY ./pytorch_ckpt.patch /workspace/pytorch/pytorch_ckpt.patch
# torch path obtained by:
# python3 -c "import torch; print(torch.__file__.replace('torch/__init__.py', ''))"
RUN patch --directory=/usr/local/lib/python3.12/dist-packages -p1 < /workspace/pytorch/pytorch_ckpt.patch
WORKDIR /workspace/

RUN pip install numcodecs==0.13.1


## 1. Apex
ARG APEX_REVISION=SKIP
ENV CUSTOM_APEX_REVISION ${APEX_REVISION}
ARG APEX_MAX_JOBS=4

RUN if [ "${APEX_REVISION}" != SKIP ]; then \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    echo APEX_REVISION=${APEX_REVISION} && \
    git checkout ${APEX_REVISION} && \
    echo APEX_COMMIT_HASH=$(git rev-parse HEAD) && \
    MAX_JOBS=${APEX_MAX_JOBS} NVCC_APPEND_FLAGS="--threads 8" pip install -v --no-build-isolation --no-cache-dir --disable-pip-version-check --config-settings "--build-option=--cpp_ext --cuda_ext --bnp --xentropy --deprecated_fused_adam --deprecated_fused_lamb --fast_multihead_attn --distributed_lamb --fast_layer_norm --transducer --distributed_adam --fmha --fast_bottleneck --nccl_p2p --peer_memory --permutation_search --focal_loss --fused_conv_bias_relu --index_mul_2d --cudnn_gbn --group_norm" . \
    ; fi

## 2. Transformer Engine
ARG TE_REVISION=c47f329b2084406093124851a3aeecb935183def
ENV CUSTOM_TE_REVISION ${TE_REVISION}


RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    git remote add vasu https://github.com/vasunvidia/TransformerEngine.git && \
    git fetch vasu vrengasamy/te_seq_prehook && \
    git cherry-pick 6d7b438b9b9ea105b99c5dc7c95963569bb3a897 && \
    NVTE_CUDA_ARCHS="90;100" NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

## 3. NeMo
ARG NEMO_REVISION=eddf23fcba178504ada2b4b4a3eb29aa71520ffa
ENV CUSTOM_NEMO_REVISION ${NEMO_REVISION}

RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    git checkout ${NEMO_REVISION} && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /NEMO_COMMIT_HASH.env && \
    pip uninstall -y nemo-toolkit sacrebleu && \
    pip install "cython<3.0.0" && \
    pip install -e ".[llm]"

## 3.1 NeMo-Run
ARG NEMORUN_REVISION=main
ENV CUSTOM_NEMORUN_REVISION ${NEMORUN_REVISION}

RUN git clone https://github.com/NVIDIA/NeMo-Run.git && \
    cd NeMo-Run && \
    git checkout ${NEMORUN_REVISION} && \
    echo NEMORUN_COMMIT_HASH=$(git rev-parse HEAD) && \
    pip install -e .

# 4. Megatron-core
ARG MCORE_REVISION=4cd81e85d2a24a727060b660bd05ca1e1a1994d7
ENV CUSTOM_MCORE_REVISION ${MCORE_REVISION}

RUN if [ "${MCORE_REVISION}" != SKIP ]; then \
    pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM Megatron-LM && \
    cd Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install . && \
    cd megatron/core/datasets && \
    make \
    ; fi

ENV PYTHONPATH "${PYTHONPATH}:/workspace/Megatron-LM"

## 5. Benchmark dependencies
RUN pip uninstall transformers -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Benchmark code
WORKDIR /workspace/llm

COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps .
ENV PYTHONPATH "/workspace/llm:/workspace/NeMo:${PYTHONPATH}"

# Install NCCL v2.28.3-1:
ARG NCCL_REVISION=v2.28.3-1
RUN export DEBIAN_FRONTEND=noninteractive &&        \
    apt-get update &&                               \
    apt-get remove -y libnccl2 &&                   \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl &&                                      \
    echo "NCCL_REVISION=${NCCL_REVISION}" &&      \
    git checkout "${NCCL_REVISION}" &&             \
    make -j24 install
ENV NCCL_VERSION "${NCCL_VERSION}-fix-${NCCL_REVISION}"

# To get rid of bitsandbytes logs
RUN pip uninstall -y bitsandbytes
