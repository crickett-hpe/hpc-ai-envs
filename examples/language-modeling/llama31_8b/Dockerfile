# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${FROM_IMAGE_NAME}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

## CUDNN
RUN NVARCH=$(uname -m |grep x86_64 2>/dev/null || echo sbsa) && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y libcudnn9-cuda-13=9.21.1.3-1 && rm -rf /var/lib/apt/lists/*
ENV CUDNN_VERSION=9.21.1.3

WORKDIR /workspace

## NCCL
ARG NCCL_REVISION=v2.30.4-1
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends fakeroot devscripts debhelper && \
    apt-get purge -y libnccl-dev libnccl2 && \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    git checkout ${NCCL_REVISION} && \
    echo "NCCL_REVISION=${NCCL_REVISION}" && \
    make -j24 pkg.debian.build && \
    dpkg -i build/pkg/deb/*.deb && \
    cd .. && rm -rf nccl && \
    apt-get purge -y fakeroot devscripts debhelper
ENV NCCL_VERSION=${NCCL_REVISION}

## 0. Pytorch Checkpoint size patch
WORKDIR /workspace/pytorch
COPY ./pytorch_ckpt.patch /workspace/pytorch/pytorch_ckpt.patch
# torch path obtained by:
# python3 -c "import torch; print(torch.__file__.replace('torch/__init__.py', ''))"
RUN patch --directory=/usr/local/lib/python3.12/dist-packages -p1 < /workspace/pytorch/pytorch_ckpt.patch

WORKDIR /workspace/
RUN pip install numcodecs==0.16.3
RUN pip install nvidia-mathdx==25.1.1

## 1. Megatron-Bridge and Megatron-Core
ARG MBRIDGE_REVISION=26.04-alpha.rc1
ARG MCORE_REVISION=26.04-alpha.rc1
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    cd /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

## 2. Transformer Engine
ARG TE_REVISION=SKIP
ENV CUSTOM_TE_REVISION ${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer-engine click && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    NVTE_CUDA_ARCHS="90;100a;103a" NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

## 3. Benchmark dependencies
COPY requirements.txt .
RUN pip uninstall -y transformers torchao && pip install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH "/usr/local/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}"

## 4. Benchmark code
WORKDIR /workspace/llm
COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps --no-build-isolation .

