SHELL := /bin/bash -o pipefail
VERSION := $(shell cat VERSION)
VERSION_DASHES := $(subst .,-,$(VERSION))
#SHORT_GIT_HASH := $(shell git rev-parse --short HEAD)
SHORT_GIT_HASH := abcdef

export DOCKERHUB_REGISTRY := cray
export REGISTRY_REPO := hpc-ai-envs
CPU_PREFIX_39 := $(REGISTRY_REPO):py-3.9-
CPU_PREFIX_310 := $(REGISTRY_REPO):py-3.10-
ROCM_56_PREFIX := $(REGISTRY_REPO):rocm-5.6-
ROCM_57_PREFIX := $(REGISTRY_REPO):rocm-5.7-
ROCM_60_PREFIX := $(REGISTRY_REPO):rocm-6.0-

CPU_SUFFIX := -cpu
CUDA_SUFFIX := -cuda
PLATFORM_LINUX_ARM_64 := linux/arm64
PLATFORM_LINUX_AMD_64 := linux/amd64
HOROVOD_GPU_OPERATIONS := NCCL

ifeq "$(WITH_MPI)" "1"
	HPC_SUFFIX := -hpc
	PLATFORMS := $(PLATFORM_LINUX_AMD_64),$(PLATFORM_LINUX_ARM_64)
	HOROVOD_WITH_MPI := 1
	HOROVOD_WITHOUT_MPI := 0
	HOROVOD_CPU_OPERATIONS := MPI
	CUDA_SUFFIX := -cuda
	WITH_AWS_TRACE := 0
	NCCL_BUILD_ARG := WITH_NCCL
        ifeq "$(WITH_NCCL)" "1"
		NCCL_BUILD_ARG := WITH_NCCL=1
		ifeq "$(WITH_AWS_TRACE)" "1"
			WITH_AWS_TRACE := 1
		endif
        endif
	MPI_BUILD_ARG := WITH_MPI=1

	ifeq "$(WITH_OFI)" "1"
	        CUDA_SUFFIX := -cuda
		CPU_SUFFIX := -cpu
		OFI_BUILD_ARG := WITH_OFI=1
	else
		CPU_SUFFIX := -cpu
		OFI_BUILD_ARG := WITH_OFI
	endif
else
	PLATFORMS := $(PLATFORM_LINUX_AMD_64),$(PLATFORM_LINUX_ARM_64)
	WITH_MPI := 0
	OFI_BUILD_ARG := WITH_OFI
	NCCL_BUILD_ARG := WITH_NCCL
	HOROVOD_WITH_MPI := 0
	HOROVOD_WITHOUT_MPI := 1
	HOROVOD_CPU_OPERATIONS := GLOO
	MPI_BUILD_ARG := USE_GLOO=1
endif

NGC_PYTORCH_PREFIX := nvcr.io/nvidia/pytorch
NGC_TENSORFLOW_PREFIX := nvcr.io/nvidia/tensorflow
NGC_PYTORCH_VERSION := 24.03-py3
NGC_TENSORFLOW_VERSION := 24.03-tf2-py3
NGC_PYTORCH_REPO := pytorch-ngc-dev
NGC_PYTORCH_HPC_REPO := pytorch-ngc-hpc-dev
NGC_TF_REPO := tensorflow-ngc-dev
NGC_TF_HPC_REPO := tensorflow-ngc-hpc-dev

# build hpc together since hpc is dependent on the normal build
.PHONY: build-pytorch-ngc
build-pytorch-ngc:
	docker build -f Dockerfile-pytorch-ngc \
		--build-arg BASE_IMAGE="$(NGC_PYTORCH_PREFIX):$(NGC_PYTORCH_VERSION)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_REPO):$(SHORT_GIT_HASH) \
		.
	docker build -f Dockerfile-ngc-hpc \
		--build-arg "$(MPI_BUILD_ARG)" \
		--build-arg "$(OFI_BUILD_ARG)" \
		--build-arg "$(NCCL_BUILD_ARG)" \
		--build-arg "WITH_PT=1" \
		--build-arg "WITH_TF=0" \
		--build-arg BASE_IMAGE="$(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_REPO):$(SHORT_GIT_HASH)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH) \
		.
ifneq ($(HPC_LIBS_DIR),)
	@echo "HPC_LIBS_DIR: $(HPC_LIBS_DIR)"
	docker build -f Dockerfile-ss \
		--build-arg BASE_IMAGE=$(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH) \
		--build-arg "HPC_LIBS_DIR=$(HPC_LIBS_DIR)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO)-ss:$(SHORT_GIT_HASH) \
		.
endif	

.PHONY: build-tensorflow-ngc
build-tensorflow-ngc:
	docker build -f Dockerfile-tensorflow-ngc \
		--build-arg BASE_IMAGE="$(NGC_TENSORFLOW_PREFIX):$(NGC_TENSORFLOW_VERSION)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_TF_REPO):$(SHORT_GIT_HASH) \
		.
	docker build -f Dockerfile-ngc-hpc \
		--build-arg "$(MPI_BUILD_ARG)" \
		--build-arg "$(OFI_BUILD_ARG)" \
		--build-arg "$(NCCL_BUILD_ARG)" \
		--build-arg "WITH_PT=0" \
		--build-arg "WITH_TF=1" \
		--build-arg BASE_IMAGE="$(DOCKERHUB_REGISTRY)/$(NGC_TF_REPO):$(SHORT_GIT_HASH)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_TF_HPC_REPO):$(SHORT_GIT_HASH) \
		.
ifneq ($(HPC_LIBS_DIR),)
	@echo "HPC_LIBS_DIR: $(HPC_LIBS_DIR)"
	docker build -f Dockerfile-ss \
		--build-arg BASE_IMAGE=$(DOCKERHUB_REGISTRY)/$(NGC_TF_HPC_REPO):$(SHORT_GIT_HASH) \
		--build-arg "HPC_LIBS_DIR=$(HPC_LIBS_DIR)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_TF_HPC_REPO)-ss:$(SHORT_GIT_HASH) \
		.
endif	

ifeq ($(WITH_MPICH),1)
ROCM57_TORCH13_MPI :=pytorch-1.3-tf-2.10-rocm-mpich
else
ROCM57_TORCH13_MPI :=pytorch-1.3-tf-2.10-rocm-ompi
endif
export ROCM57_TORCH13_TF_ENVIRONMENT_NAME := $(ROCM_57_PREFIX)$(ROCM57_TORCH13_MPI)
.PHONY: build-pytorch13-tf210-rocm57
build-pytorch13-tf210-rocm57:
	docker build -f Dockerfile-default-rocm \
		--build-arg BASE_IMAGE="rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_1.13.1"\
		--build-arg TENSORFLOW_PIP="tensorflow-rocm==2.10.1.540" \
		--build-arg HOROVOD_PIP="horovod==0.28.1" \
		--build-arg WITH_MPICH=$(WITH_MPICH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM57_TORCH13_TF_ENVIRONMENT_NAME)-$(SHORT_GIT_HASH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM57_TORCH13_TF_ENVIRONMENT_NAME)-$(VERSION) \
		.

ifeq ($(WITH_MPICH),1)
ROCM57_TORCH_MPI :=pytorch-2.0-tf-2.10-rocm-mpich
else
ROCM57_TORCH_MPI :=pytorch-2.0-tf-2.10-rocm-ompi
endif
export ROCM57_TORCH_TF_ENVIRONMENT_NAME := $(ROCM_57_PREFIX)$(ROCM57_TORCH_MPI)
.PHONY: build-pytorch20-tf210-rocm57
build-pytorch20-tf210-rocm57:
	docker build -f Dockerfile-default-rocm \
		--build-arg BASE_IMAGE="rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1" \
		--build-arg TENSORFLOW_PIP="tensorflow-rocm==2.10.1.540" \
		--build-arg HOROVOD_PIP="horovod==0.28.1" \
		--build-arg WITH_MPICH=$(WITH_MPICH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM57_TORCH_TF_ENVIRONMENT_NAME)-$(SHORT_GIT_HASH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM57_TORCH_TF_ENVIRONMENT_NAME)-$(VERSION) \
		.

ifeq ($(WITH_MPICH),1)
ROCM60_TORCH13_MPI :=pytorch-1.3-tf-2.10-rocm-mpich
else
ROCM60_TORCH13_MPI :=pytorch-1.3-tf-2.10-rocm-ompi
endif
export ROCM60_TORCH13_TF_ENVIRONMENT_NAME := $(ROCM_60_PREFIX)$(ROCM60_TORCH13_MPI)
.PHONY: build-pytorch13-tf210-rocm60
build-pytorch13-tf210-rocm60:
	docker build -f Dockerfile-default-rocm \
		--build-arg BASE_IMAGE="rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_1.13.1" \
		--build-arg TENSORFLOW_PIP="tensorflow-rocm==2.10.1.540" \
		--build-arg HOROVOD_PIP="horovod==0.28.1" \
		--build-arg WITH_MPICH=$(WITH_MPICH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM60_TORCH13_TF_ENVIRONMENT_NAME)-$(SHORT_GIT_HASH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM60_TORCH13_TF_ENVIRONMENT_NAME)-$(VERSION) \
		.

ifeq ($(WITH_MPICH),1)
ROCM60_TORCH_MPI :=pytorch-2.0-tf-2.10-rocm-mpich
else
ROCM60_TORCH_MPI :=pytorch-2.0-tf-2.10-rocm-ompi
endif
export ROCM60_TORCH_TF_ENVIRONMENT_NAME := $(ROCM_60_PREFIX)$(ROCM60_TORCH_MPI)
.PHONY: build-pytorch20-tf210-rocm60
build-pytorch20-tf210-rocm60:
	docker build -f Dockerfile-default-rocm \
		--build-arg BASE_IMAGE="rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1" \
		--build-arg TENSORFLOW_PIP="tensorflow-rocm==2.10.1.540" \
		--build-arg HOROVOD_PIP="horovod==0.28.1" \
		--build-arg WITH_MPICH=$(WITH_MPICH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM60_TORCH_TF_ENVIRONMENT_NAME)-$(SHORT_GIT_HASH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM60_TORCH_TF_ENVIRONMENT_NAME)-$(VERSION) \
		.

DEEPSPEED_VERSION := 0.8.3
export TORCH_TB_PROFILER_PIP := torch-tb-profiler==0.4.1
export GPU_DEEPSPEED_ENVIRONMENT_NAME := $(CUDA_113_PREFIX)pytorch-1.10-deepspeed-$(DEEPSPEED_VERSION)$(GPU_SUFFIX)
export GPU_GPT_NEOX_DEEPSPEED_ENVIRONMENT_NAME := $(CUDA_113_PREFIX)pytorch-1.10-gpt-neox-deepspeed$(GPU_SUFFIX)
export TORCH_PIP_DEEPSPEED_GPU := torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

export ROCM57_TORCH_TF_ENVIRONMENT_NAME_DEEPSPEED := $(ROCM_57_PREFIX)pytorch-2.0-tf-2.10-rocm-deepspeed

.PHONY: build-pytorch20-tf210-rocm57-deepspeed
build-pytorch20-tf210-rocm57-deepspeed:
	DOCKER_BUILDKIT=0 docker build --shm-size='1gb' -f Dockerfile-default-rocm \
		--build-arg BASE_IMAGE="rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.1.1" \
		--build-arg TENSORFLOW_PIP="tensorflow-rocm==2.10.1.540" \
		--build-arg HOROVOD_PIP="horovod==0.28.1" \
		--build-arg TORCH_PIP="$(TORCH_PIP_DEEPSPEED_GPU)" \
		--build-arg TORCH_TB_PROFILER_PIP="$(TORCH_TB_PROFILER_PIP)" \
		--build-arg TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0" \
		--build-arg APEX_GIT="https://github.com/determined-ai/apex.git@3caf0f40c92e92b40051d3afff8568a24b8be28d" \
		--build-arg DEEPSPEED_PIP="deepspeed==$(DEEPSPEED_VERSION)" \
		--build-arg WITH_MPICH=$(WITH_MPICH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM57_TORCH_TF_ENVIRONMENT_NAME_DEEPSPEED)-$(SHORT_GIT_HASH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM57_TORCH_TF_ENVIRONMENT_NAME_DEEPSPEED)-$(VERSION) \
    .

export ROCM60_TORCH_TF_ENVIRONMENT_NAME_DEEPSPEED := $(ROCM_60_PREFIX)pytorch-2.0-tf-2.10-rocm-deepspeed
.PHONY: build-pytorch20-tf210-rocm60-deepspeed
build-pytorch20-tf210-rocm60-deepspeed:
	DOCKER_BUILDKIT=0 docker build --shm-size='1gb' -f Dockerfile-default-rocm \
		--build-arg BASE_IMAGE="rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1" \
		--build-arg TENSORFLOW_PIP="tensorflow-rocm==2.10.1.540" \
		--build-arg HOROVOD_PIP="horovod==0.28.1" \
		--build-arg TORCH_PIP="$(TORCH_PIP_DEEPSPEED_GPU)" \
		--build-arg TORCH_TB_PROFILER_PIP="$(TORCH_TB_PROFILER_PIP)" \
		--build-arg TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0" \
		--build-arg APEX_GIT="https://github.com/determined-ai/apex.git@3caf0f40c92e92b40051d3afff8568a24b8be28d" \
		--build-arg DEEPSPEED_PIP="deepspeed==$(DEEPSPEED_VERSION)" \
		--build-arg WITH_MPICH=$(WITH_MPICH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM60_TORCH_TF_ENVIRONMENT_NAME_DEEPSPEED)-$(SHORT_GIT_HASH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM60_TORCH_TF_ENVIRONMENT_NAME_DEEPSPEED)-$(VERSION) \
    .

