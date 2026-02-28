SHELL := /bin/bash -o pipefail
VERSION := $(shell cat VERSION)
VERSION_DASHES := $(subst .,-,$(VERSION))
SHORT_GIT_HASH := $(shell git rev-parse --short HEAD)

export DOCKERHUB_REGISTRY := cray
export REGISTRY_REPO := hpc-ai-envs

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
WITH_HOROVOD ?= 0
WITH_AWS_TRACE ?= 0
CRAY_LIBFABRIC_DIR ?= "/opt/cray/libfabric/1.15.2.0"
CRAY_LIBCXI_DIR ?= "/usr"
NGC_VERSION ?= 25.06
LIBFABRIC_VERSION ?= "2.2.0"

# If the user doesn't explicitly pass in a value for BUILD_SIF, then
# default it to 1 if singularity is in the PATH
BUILD_SIF ?= $(shell singularity -h 2>/dev/null|head -1c 2>/dev/null|wc -l)

# If the user specifies USE_CWD_SIF=1 on the command line, singularity
# will use the current working directory for temp and cache space, this
# is useful if there's not enough space in /tmp for example.
# If not specified (or if USE_CWD_SIF=0 is set) then singularity will
# use its default tmp and cache dir locations.
USE_CWD_SIF ?= 0

# If not specified (or if RM_SIF_TAR=0 is set) then the docker saved
# tarfile will not be removed
RM_SIF_TAR ?= 0

ifeq ($(BUILD_SIF),1)
    BUILD_TAR ?= 1
else
    BUILD_TAR ?= 0
endif

ifeq "$(WITH_MPI)" "1"
	HPC_SUFFIX := -hpc
	PLATFORMS := $(PLATFORM_LINUX_AMD_64),$(PLATFORM_LINUX_ARM_64)
	HOROVOD_WITH_MPI := 1
	HOROVOD_WITHOUT_MPI := 0
	HOROVOD_CPU_OPERATIONS := MPI
	CUDA_SUFFIX := -cuda
	NCCL_BUILD_ARG := WITH_NCCL
        ifeq "$(WITH_NCCL)" "1"
		NCCL_BUILD_ARG := WITH_NCCL=1
        endif
	MPI_BUILD_ARG := WITH_MPI=1

	ifeq "$(WITH_AWS_TRACE)" "1"
		AWS_TRACE_ARG := WITH_AWS_TRACE=1
	else
		AWS_TRACE_ARG := WITH_AWS_TRACE=0
	endif

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
	AWS_TRACE_ARG := WITH_AWS_TRACE=0
endif

XCCL_BUILD_ARG := WITH_XCCL=0
ifeq "$(WITH_XCCL)" "1"
	XCCL_BUILD_ARG := WITH_XCCL=1
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

NGC_PYTORCH_PREFIX := nvcr.io/nvidia/pytorch
NGC_PYTORCH_VERSION := $(NGC_VERSION)-py3
NGC_PYTORCH_REPO := ngc-$(NGC_PYTORCH_VERSION)-pt
NGC_PYTORCH_HPC_REPO := ngc-$(NGC_PYTORCH_VERSION)-pt-hpc

# From https://hub.docker.com/r/rocm/pytorch/tags
# rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0
ROCM_PT_PREFIX  := rocm/pytorch
ROCM_VERSION    := rocm6.3.4
ROCM_UBUNTU     := ubuntu22.04
PYTHON_VERSION  := py3.10
ROCM_PT_RELEASE := pytorch_release_2.4.0
ROCM_PT_VERSION := $(ROCM_VERSION)_$(ROCM_UBUNTU)_$(PYTHON_VERSION)_$(ROCM_PT_RELEASE)
ROCM_PYTORCH_REPO := $(ROCM_VERSION)-$(PYTHON_VERSION)-pt
ROCM_PYTORCH_HPC_REPO := $(ROCM_VERSION)-$(PYTHON_VERSION)-pt-hpc

# Separate out NGC vs ROCM base images only because it could impact which
# tools we add, such as nccl vs rccl, in the HPC Dockerfile. Note we could
# likely modify this to work for both and have a separate flag to specify
# nccl/rccl, etc, to keep things cleaner.
USER_NGC_BASE_IMAGE ?= $(NGC_PYTORCH_PREFIX):$(NGC_PYTORCH_VERSION)
ifneq ($(USER_NGC_BASE_IMAGE),)
        USER_NGC_IMAGE_REPO=$(shell echo "$(USER_NGC_BASE_IMAGE)" | awk 'BEGIN{FS=OFS="/"}{NF--; print}')
        USER_NGC_IMAGE_NAME=$(shell echo "$(USER_NGC_BASE_IMAGE)" | awk -F "/" '{print $$NF}' | awk -F ":" '{print $$1}')
        USER_NGC_IMAGE_VER=$(shell echo "$(USER_NGC_BASE_IMAGE)" | awk -F "/" '{print $$NF}' | awk -F ":" '{print $$NF}')
        USER_NGC_IMAGE_HPC=$(USER_NGC_IMAGE_REPO)/$(USER_NGC_IMAGE_NAME)-hpc:$(USER_NGC_IMAGE_VER)
        USER_NGC_IMAGE_SS=$(USER_NGC_IMAGE_REPO)/$(USER_NGC_IMAGE_NAME)-hpc-ss:$(USER_NGC_IMAGE_VER)
        USER_NGC_IMAGE_SIF=$(shell echo "$(USER_NGC_BASE_IMAGE)" | sed s,'/','-',g | sed s,':','-',g)
endif

USER_ROCM_BASE_IMAGE ?= $(ROCM_PT_PREFIX):$(ROCM_PT_VERSION)
ifneq ($(USER_ROCM_BASE_IMAGE),)
        USER_ROCM_IMAGE_REPO=$(shell echo "$(USER_ROCM_BASE_IMAGE)" | awk 'BEGIN{FS=OFS="/"}{NF--; print}')
        USER_ROCM_IMAGE_NAME=$(shell echo "$(USER_ROCM_BASE_IMAGE)" | awk -F "/" '{print $$NF}' | awk -F ":" '{print $$1}')
        USER_ROCM_IMAGE_VER=$(shell echo "$(USER_ROCM_BASE_IMAGE)" | awk -F "/" '{print $$NF}' | awk -F ":" '{print $$NF}')
        USER_ROCM_IMAGE_HPC=$(USER_ROCM_IMAGE_REPO)/$(USER_ROCM_IMAGE_NAME)-hpc:$(USER_ROCM_IMAGE_VER)
        USER_ROCM_IMAGE_SS=$(USER_ROCM_IMAGE_REPO)/$(USER_ROCM_IMAGE_NAME)-hpc-ss:$(USER_ROCM_IMAGE_VER)
        USER_ROCM_IMAGE_SIF=$(shell echo "$(USER_ROCM_BASE_IMAGE)" | sed s,'/','-',g | sed s,':','-',g)
endif

# build pytorch sif
TMP_SIF := $(shell mktemp -d -t sif-reg.XXXXXX)
TMP_SIF_BASE := "$(PWD)/$(shell basename $(TMP_SIF))"

SING_DIRS :=
ifeq "$(USE_CWD_SIF)" "1"
     SING_DIRS := SINGULARITY_TMPDIR=$(TMP_SIF_BASE) SINGULARITY_CACHEDIR=$(TMP_SIF_BASE)
endif

.PHONY: build-sif
build-sif:
	# Either BUILD_TAR = 1 or BUILD_TAR and BUILD_SIF = 1
	docker save -o "$(TARGET_NAME).tar" $(TARGET_TAG)
ifeq ($(BUILD_SIF),1)
		# Make a tmp dir in the cwd using the tmp_file name.
		mkdir $(TMP_SIF_BASE)
		env $(SING_DIRS) \
	            SINGULARITY_NOHTTPS=true NAMESPACE="" \
	            singularity -vvv build $(TARGET_NAME).sif \
	                             "docker-archive://$(TARGET_NAME).tar"
                ifeq ($(RM_SIF_TAR),1)
			rm -f "$(TARGET_NAME).tar"
                endif
		rm -rf $(TMP_SIF_BASE)
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
	docker build -f Dockerfile-ngc-hpc $(BUILD_OPTS) \
		--build-arg "$(NCCL_BUILD_ARG)" \
		--build-arg "$(XCCL_BUILD_ARG)" \
		--build-arg "$(MPI_BUILD_ARG)" \
		--build-arg "$(OFI_BUILD_ARG)" \
		--build-arg "$(AWS_TRACE_ARG)" \
		--build-arg "WITH_PT=1" \
		--build-arg "WITH_TF=0" \
		--build-arg BASE_IMAGE="$(USER_NGC_BASE_IMAGE)" \
		--build-arg "LIBFABRIC_VERSION=$(LIBFABRIC_VERSION)" \
		-t $(USER_NGC_IMAGE_HPC)\
		.
ifeq "$(BUILD_TAR)" "1"
	    @echo "BUILD_TAR: $(USER_NGC_IMAGE_HPC)"
	    make build-sif TARGET_TAG="$(USER_NGC_IMAGE_HPC)" \
	        TARGET_NAME="$(shell echo "$(USER_NGC_BASE_IMAGE)" | sed s,'/','-',g | sed s,':','-hpc-',g)"
endif


# Build an HPC container using the base image provided by the user.
# This enables us to append the SS11 bits to an otherwise working
# user image to make it easier for users to deploy their containers on SS11.
.PHONY: build-user-spec-rocm
build-user-spec-rocm:
	@echo "USER_ROCM_BASE_IMAGE: $(USER_ROCM_BASE_IMAGE)"
	@echo "USER_ROCM_IMAGE_REPO: $(USER_ROCM_IMAGE_REPO)"
	@echo "USER_ROCM_IMAGE_NAME: $(USER_ROCM_IMAGE_NAME)"
	@echo "USER_ROCM_IMAGE_VER: $(USER_ROCM_IMAGE_VER)"
	@echo "USER_ROCM_IMAGE_HPC: $(USER_ROCM_IMAGE_HPC)"
	@echo "USER_ROCM_IMAGE_SS: $(USER_ROCM_IMAGE_SS)"
	@echo "USER_ROCM_IMAGE_SIF: $(USER_ROCM_IMAGE_SIF)"
	docker build -f Dockerfile-rocm-hpc $(BUILD_OPTS) \
		--build-arg "$(NCCL_BUILD_ARG)" \
		--build-arg "$(XCCL_BUILD_ARG)" \
		--build-arg "$(MPI_BUILD_ARG)" \
		--build-arg "$(OFI_BUILD_ARG)" \
		--build-arg "$(AWS_TRACE_ARG)" \
		--build-arg "WITH_PT=1" \
		--build-arg "WITH_TF=0" \
		--build-arg BASE_IMAGE="$(USER_ROCM_BASE_IMAGE)" \
		--build-arg "LIBFABRIC_VERSION=$(LIBFABRIC_VERSION)" \
		-t $(USER_ROCM_IMAGE_HPC)\
		.
ifeq "$(BUILD_TAR)" "1"
	    @echo "BUILD_TAR: $(USER_ROCM_IMAGE_HPC)"
	    make build-sif TARGET_TAG="$(USER_ROCM_IMAGE_HPC)" \
                TARGET_NAME="$(shell echo "$(USER_ROCM_BASE_IMAGE)" | sed s,'/','-',g | sed s,':','-hpc-',g)"
endif
