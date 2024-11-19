SHELL := /bin/bash -o pipefail
VERSION := $(shell cat VERSION)
VERSION_DASHES := $(subst .,-,$(VERSION))
SHORT_GIT_HASH := $(shell git rev-parse --short HEAD)

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

# Default to enabling MPI, OFI and SS11. Note that if we cannot
# find the SS11 libs automatically and the user did not provide
# a location we will not end up building the -ss version of the image.
# This just means the user would need to bind-mount the SS11 libs
# at runtime.
WITH_MPI ?= 1
WITH_OFI ?= 1
WITH_SS11 ?= 1
BUILD_SIF ?= 1
CRAY_LIBFABRIC_DIR ?= "/opt/cray/libfabric/1.15.2.0"
CRAY_LIBCXI_DIR ?= "/usr"

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

ifeq "$(WITH_SS11)" "1"
	ifeq ($(HPC_LIBS_DIR),)
           LIBFAB_SO=$(shell find $(CRAY_LIBFABRIC_DIR) -name libfabric\*so)
           LIBCXI_SO=$(shell find $(CRAY_LIBCXI_DIR) -name libcxi\*so)
           # Make sure we found the libs
           ifneq ($(and $(LIBFAB_SO),$(LIBCXI_SO)),)
              LIBFAB_DIR=$(shell dirname $(LIBFAB_SO))
              LIBCXI_DIR=$(shell dirname $(LIBCXI_SO))
              # Copy the libfabric/cxi to a tmp dir for the HPC_LIBS_DIR
              TMP_FILE:=$(shell mktemp -d -t ss11-libs.XXXXXX)
              TMP_FILE_BASE=$(shell basename $(TMP_FILE))
              # Make a tmp dir in the cwd using the tmp_file name.
              # We do this to distinguish if we made the dir vs the user
              # putting it there so we know to clean it up after the build.
              $(shell mkdir $(TMP_FILE_BASE))
              HPC_LIBS_DIR=$(TMP_FILE_BASE)
              cp_out:=$(shell cp $(LIBFAB_DIR)/libfabric* $(HPC_LIBS_DIR))
              cp_out:=$(shell cp $(LIBCXI_DIR)/libcxi* $(HPC_LIBS_DIR))
              # Signal that the libs were copied so we clean them up after.
              HPC_TMP_LIBS_DIR := 1
           endif
        endif
endif


NGC_PYTORCH_PREFIX := nvcr.io/nvidia/pytorch
NGC_TENSORFLOW_PREFIX := nvcr.io/nvidia/tensorflow
NGC_PYTORCH_VERSION := 24.03-py3
NGC_TENSORFLOW_VERSION := 24.03-tf2-py3
NGC_PYTORCH_REPO := pytorch-ngc-dev
NGC_PYTORCH_HPC_REPO := pytorch-ngc-hpc-dev
NGC_TF_REPO := tensorflow-ngc-dev
NGC_TF_HPC_REPO := tensorflow-ngc-hpc-dev

# build pytorch sif
TMP_SIF := $(shell mktemp -d -t sif-reg.XXXXXX)
TMP_SIF_BASE := "$(PWD)/$(shell basename $(TMP_SIF))"

# Use the user's SINGULARITY_TMPDIR environment variable if it is set
SINGULARITY_TMPDIR ?= $(TMP_SIF_BASE)

# Use the user's SINGULARITY_CACHEDIR environment variable if it is set
SINGULARITY_CACHEDIR ?= $(TMP_SIF_BASE)

.PHONY: build-sif
build-sif:
	# Make a tmp dir in the cwd using the tmp_file name.
	mkdir $(TMP_SIF_BASE)
	docker save -o "$(TARGET_NAME).tar" $(DOCKERHUB_REGISTRY)/$(TARGET_TAG)
	env SINGULARITY_TMPDIR=$(SINGULARITY_TMPDIR) SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR) \
            SINGULARITY_NOHTTPS=true NAMESPACE="" \
            singularity -vvv build $(TARGET_NAME).sif \
                             "docker-archive://$(TARGET_NAME).tar"
	rm -rf $(TMP_SIF_BASE) "$(TARGET_NAME).tar"

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
        ifneq ($(HPC_TMP_LIBS_DIR),)
	    rm -rf $(HPC_LIBS_DIR)
        endif
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(NGC_PYTORCH_HPC_REPO)-ss:$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(NGC_PYTORCH_HPC_REPO)-ss:$(SHORT_GIT_HASH)" TARGET_NAME="$(NGC_PYTORCH_HPC_REPO)-$(SHORT_GIT_HASH)"
        endif
else
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH)" TARGET_NAME="$(NGC_PYTORCH_HPC_REPO)-$(SHORT_GIT_HASH)"
        endif
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
	ifneq ($(HPC_TMP_LIBS_DIR),)
		rm -rf $(HPC_LIBS_DIR)
	endif
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(NGC_TF_HPC_REPO)-ss:$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(NGC_TF_HPC_REPO)-ss:$(SHORT_GIT_HASH)" TARGET_NAME="$(NGC_TF_HPC_REPO)-$(SHORT_GIT_HASH)"
        endif
else
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(NGC_TF_HPC_REPO):$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(NGC_TF_HPC_REPO):$(SHORT_GIT_HASH)" TARGET_NAME="$(NGC_TF_HPC_REPO)-$(SHORT_GIT_HASH)"
        endif
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

