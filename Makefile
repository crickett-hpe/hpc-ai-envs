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
ROCM_63_PREFIX := $(REGISTRY_REPO):rocm-6.3-

CPU_SUFFIX := -cpu
CUDA_SUFFIX := -cuda
PLATFORM_LINUX_ARM_64 := linux/arm64
PLATFORM_LINUX_AMD_64 := linux/amd64
HOROVOD_GPU_OPERATIONS := NCCL
BUILD_OPTS ?=

# Default to enabling MPI, OFI and SS11. Note that if we cannot
# find the SS11 libs automatically and the user did not provide
# a location we will not end up building the -ss version of the image.
# This just means the user would need to bind-mount the SS11 libs
# at runtime.
WITH_MPI ?= 1
WITH_OFI ?= 1
WITH_SS11 ?= 0
CRAY_LIBFABRIC_DIR ?= "/opt/cray/libfabric/1.15.2.0"
CRAY_LIBCXI_DIR ?= "/usr"

# If the user doesn't explicitly pass in a value for BUILD_SIF, then
# default it to 1 if singularity is in the PATH
BUILD_SIF ?= $(shell singularity -h 2>/dev/null|head -1c 2>/dev/null|wc -l)

# If the user specifies USE_CWD_SIF=1 on the command line, singularity
# will use the current working directory for temp and cache space, this
# is useful if there's not enough space in /tmp for example.
# If not specified (or if USE_CWD_SIF=0 is set) then singularity will
# use its default tmp and cache dir locations.
USE_CWD_SIF ?= 0

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
           LIBFAB_SO=$(shell find $(CRAY_LIBFABRIC_DIR) -name libfabric\*so.\*)
           LIBCXI_SO=$(shell find $(CRAY_LIBCXI_DIR) -name libcxi\*so.\*)
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


# Separate out NGC vs ROCM base images only because it could impact which
# tools we add, such as nccl vs rccl, in the HPC Dockerfile. Note we could
# likely modify this to work for both and have a separate flag to specify
# nccl/rccl, etc, to keep things cleaner.
ifneq ($(USER_NGC_BASE_IMAGE),)
        USER_NGC_IMAGE_REPO=$(shell echo "$(USER_NGC_BASE_IMAGE)" | awk 'BEGIN{FS=OFS="/"}{NF--; print}')
        USER_NGC_IMAGE_NAME=$(shell echo "$(USER_NGC_BASE_IMAGE)" | awk -F "/" '{print $$NF}' | awk -F ":" '{print $$1}')
        USER_NGC_IMAGE_VER=$(shell echo "$(USER_NGC_BASE_IMAGE)" | awk -F "/" '{print $$NF}' | awk -F ":" '{print $$NF}')
        USER_NGC_IMAGE_HPC=$(USER_NGC_IMAGE_REPO)/$(USER_NGC_IMAGE_NAME)-hpc:$(USER_NGC_IMAGE_VER)
        USER_NGC_IMAGE_SS=$(USER_NGC_IMAGE_REPO)/$(USER_NGC_IMAGE_NAME)-hpc-ss:$(USER_NGC_IMAGE_VER)
        USER_NGC_IMAGE_SIF=$(shell echo "$(USER_NGC_BASE_IMAGE)" | sed s,'/','-',g | sed s,':','-',g)
endif


NGC_PYTORCH_PREFIX := nvcr.io/nvidia/pytorch
NGC_TENSORFLOW_PREFIX := nvcr.io/nvidia/tensorflow
NGC_PYTORCH_VERSION := 24.11-py3
NGC_TENSORFLOW_VERSION := 24.03-tf2-py3
NGC_PYTORCH_REPO := ngc-$(NGC_PYTORCH_VERSION)-pt
NGC_PYTORCH_HPC_REPO := ngc-$(NGC_PYTORCH_VERSION)-pt-hpc
NGC_TF_REPO := tensorflow-ngc-dev
NGC_TF_HPC_REPO := tensorflow-ngc-hpc-dev

# build pytorch sif
TMP_SIF := $(shell mktemp -d -t sif-reg.XXXXXX)
TMP_SIF_BASE := "$(PWD)/$(shell basename $(TMP_SIF))"

SING_DIRS :=
ifeq "$(USE_CWD_SIF)" "1"
     SING_DIRS := SINGULARITY_TMPDIR=$(TMP_SIF_BASE) SINGULARITY_CACHEDIR=$(TMP_SIF_BASE)
endif

.PHONY: build-sif
build-sif:
	# Make a tmp dir in the cwd using the tmp_file name.
	mkdir $(TMP_SIF_BASE)
	docker save -o "$(TARGET_NAME).tar" $(TARGET_TAG)
	env $(SING_DIRS) \
            SINGULARITY_NOHTTPS=true NAMESPACE="" \
            singularity -vvv build $(TARGET_NAME).sif \
                             "docker-archive://$(TARGET_NAME).tar"
	rm -rf $(TMP_SIF_BASE) "$(TARGET_NAME).tar"

# build hpc together since hpc is dependent on the normal build
.PHONY: build-pytorch-ngc
build-pytorch-ngc:
	docker build -f Dockerfile-pytorch-ngc $(BUILD_OPTS) \
		--build-arg BASE_IMAGE="$(NGC_PYTORCH_PREFIX):$(NGC_PYTORCH_VERSION)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_REPO):$(SHORT_GIT_HASH) \
		.
	docker build -f Dockerfile-ngc-hpc $(BUILD_OPTS) \
		--build-arg "$(NCCL_BUILD_ARG)" \
		--build-arg "$(MPI_BUILD_ARG)" \
		--build-arg "$(OFI_BUILD_ARG)" \
		--build-arg "WITH_PT=1" \
		--build-arg "WITH_TF=0" \
		--build-arg BASE_IMAGE="$(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_REPO):$(SHORT_GIT_HASH)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH) \
		.
	@echo "HPC_LIBS_DIR: $(HPC_LIBS_DIR)"
	@echo "WITH_SS11: $(WITH_SS11)"
	@echo "LIBFAB_DIR: $(LIBFAB_DIR)"
	@echo "LIBCXI_DIR: $(LIBCXI_DIR)"
ifneq ($(HPC_LIBS_DIR),)
	@echo "HPC_LIBS_DIR: $(HPC_LIBS_DIR)"
	docker build -f Dockerfile-ss $(BUILD_OPTS) \
		--build-arg BASE_IMAGE=$(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH) \
		--build-arg "HPC_LIBS_DIR=$(HPC_LIBS_DIR)" \
		-t $(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO)-ss:$(SHORT_GIT_HASH) \
		.
        ifneq ($(HPC_TMP_LIBS_DIR),)
	    rm -rf $(HPC_LIBS_DIR)
        endif
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(NGC_PYTORCH_HPC_REPO)-ss:$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO)-ss:$(SHORT_GIT_HASH)" \
                          TARGET_NAME="$(NGC_PYTORCH_HPC_REPO)-$(SHORT_GIT_HASH)"
        endif
else
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(DOCKERHUB_REGISTRY)/$(NGC_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH)" \
                          TARGET_NAME="$(NGC_PYTORCH_HPC_REPO)-$(SHORT_GIT_HASH)"
        endif
endif


# Build an HPC container using the base image provided by the user. 
# This enables us to append the SS11 bits to an otherwise working
# user image to make it easier for users to deploy their containers on SS11.
.PHONY: build-user-spec-ngc
build-user-spec-ngc:
	@echo "USER_NGC_BASE_IMAGE: $(USER_NGC_BASE_IMAGE)"
	@echo "USER_NGC_IMAGE_REPO: $(USER_NGC_IMAGE_REPO)"
	@echo "USER_NGC_IMAGE_NAME: $(USER_NGC_IMAGE_NAME)"
	@echo "USER_NGC_IMAGE_VER: $(USER_NGC_IMAGE_VER)"
	@echo "USER_NGC_IMAGE_HPC: $(USER_NGC_IMAGE_HPC)"
	@echo "USER_NGC_IMAGE_SS: $(USER_NGC_IMAGE_SS)"
	@echo "USER_NGC_IMAGE_SIF: $(USER_NGC_IMAGE_SIF)"
	docker build -f Dockerfile-ngc-hpc \
		--build-arg "$(NCCL_BUILD_ARG)" \
		--build-arg "$(MPI_BUILD_ARG)" \
		--build-arg "$(OFI_BUILD_ARG)" \
		--build-arg "WITH_PT=1" \
		--build-arg "WITH_TF=0" \
		--build-arg BASE_IMAGE="$(USER_NGC_BASE_IMAGE)" \
		-t $(USER_NGC_IMAGE_HPC)\
		.
ifneq ($(HPC_LIBS_DIR),)
	docker build -f Dockerfile-ss \
		--build-arg BASE_IMAGE=$(USER_NGC_IMAGE_HPC) \
		--build-arg "HPC_LIBS_DIR=$(HPC_LIBS_DIR)" \
		-t $(USER_NGC_IMAGE_SS) \
		.
        ifneq ($(HPC_TMP_LIBS_DIR),)
	    rm -rf $(HPC_LIBS_DIR)
        endif
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(USER_NGC_IMAGE_SS)"
	    make build-sif TARGET_TAG="$(USER_NGC_IMAGE_SS)" TARGET_NAME="$(USER_NGC_IMAGE_SIF)"
        endif
else
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(USER_NGC_IMAGE_HPC)"
	    make build-sif TARGET_TAG="$(USER_NGC_IMAGE_HPC)" TARGET_NAME="$(USER_NGC_IMAGE_SIF)"
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
	    make build-sif TARGET_TAG="$(DOCKERHUB_REGISTRY)/$(NGC_TF_HPC_REPO)-ss:$(SHORT_GIT_HASH)" \
                          TARGET_NAME="$(NGC_TF_HPC_REPO)-$(SHORT_GIT_HASH)"
        endif
else
        ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(NGC_TF_HPC_REPO):$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(DOCKERHUB_REGISTRY)/$(NGC_TF_HPC_REPO):$(SHORT_GIT_HASH)" \
                          TARGET_NAME="$(NGC_TF_HPC_REPO)-$(SHORT_GIT_HASH)"
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
ifeq ($(WITH_MPICH),1)
ROCM63_TORCH_MPI :=pytorch-2.4-tf-2.10-rocm-mpich
else
ROCM63_TORCH_MPI :=pytorch-2.4-tf-2.10-rocm-ompi
endif
ROCM_PYTORCH_VERSION := 24.11-py3
ROCM_PYTORCH_REPO := rocm-$(ROCM_PYTORCH_VERSION)-pt
ROCM_PYTORCH_HPC_REPO := rocm-$(ROCM_PYTORCH_VERSION)-pt-hpc
export ROCM63_TORCH_TF_ENVIRONMENT_NAME := $(ROCM_60_PREFIX)$(ROCM63_TORCH_MPI)
.PHONY: build-pytorch-rocm63
build-pytorch-rocm:
	docker build -f Dockerfile-pytorch-rocm $(BUILD_OPTS) \
		--build-arg BASE_IMAGE="rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0" \
		--build-arg WITH_MPICH=$(WITH_MPICH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM_PYTORCH_REPO):$(SHORT_GIT_HASH) \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM63_TORCH_TF_ENVIRONMENT_NAME)-$(VERSION) \
		.
	@echo "ROCM63_TORCH_TF_ENVIRONMENT_NAME: $(DOCKERHUB_REGISTRY)/$(ROCM_PYTORCH_REPO):$(SHORT_GIT_HASH)"
	docker build -f Dockerfile-rocm-hpc $(BUILD_OPTS) \
		--build-arg TENSORFLOW_PIP="tensorflow-rocm==2.10.1.540" \
		--build-arg HOROVOD_PIP="horovod==0.28.1" \
		--build-arg "$(NCCL_BUILD_ARG)" \
		--build-arg "$(MPI_BUILD_ARG)" \
		--build-arg "$(OFI_BUILD_ARG)" \
		--build-arg "WITH_PT=1" \
		--build-arg "WITH_TF=0" \
		--build-arg BASE_IMAGE="$(DOCKERHUB_REGISTRY)/$(ROCM_PYTORCH_REPO):$(SHORT_GIT_HASH)" \
		-t $(DOCKERHUB_REGISTRY)/$(ROCM_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH) \
		.
ifeq "$(BUILD_SIF)" "1"
	    @echo "BUILD_SIF: $(ROCM_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH)"
	    make build-sif TARGET_TAG="$(DOCKERHUB_REGISTRY)/$(ROCM_PYTORCH_HPC_REPO):$(SHORT_GIT_HASH)" \
                          TARGET_NAME="$(ROCM_PYTORCH_HPC_REPO)-$(SHORT_GIT_HASH)"
endif


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

