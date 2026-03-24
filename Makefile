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
WITH_HOROVOD ?= 0
WITH_AWS_TRACE ?= 0
LIBFABRIC_VERSION ?= 2.2.0
DOCKER ?= docker


NGC_VERSION ?= 25.06
NGC_PYTORCH_PREFIX   := nvcr.io/nvidia/pytorch
NGC_PYTORCH_VERSION  := $(NGC_VERSION)-py3
NGC_PYTORCH_REPO     := ngc-$(NGC_PYTORCH_VERSION)-pt
NGC_PYTORCH_HPC_REPO := ngc-$(NGC_PYTORCH_VERSION)-pt-hpc

ROCM_VERSION ?= rocm6.3.4
# From https://hub.docker.com/r/rocm/pytorch/tags
# rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0
ROCM_PT_PREFIX  := rocm/pytorch
ROCM_UBUNTU     := ubuntu22.04
PYTHON_VERSION  := py3.10
ROCM_PT_RELEASE := pytorch_release_2.4.0
ROCM_PT_VERSION := $(ROCM_VERSION)_$(ROCM_UBUNTU)_$(PYTHON_VERSION)_$(ROCM_PT_RELEASE)

# If the user specifies USE_CWD_SIF=1 on the command line, singularity
# will use the current working directory for temp and cache space, this
# is useful if there's not enough space in /tmp for example.
# If not specified (or if USE_CWD_SIF=0 is set) then singularity will
# use its default tmp and cache dir locations.
USE_CWD_SIF ?= 0

# If not specified (or if RM_SIF_TAR=0 is set) then the docker saved
# tarfile will not be removed
RM_SIF_TAR ?= 0

ifeq ($(WITH_MPI),1)
	HPC_SUFFIX := -hpc
	PLATFORMS := $(PLATFORM_LINUX_AMD_64),$(PLATFORM_LINUX_ARM_64)
	HOROVOD_WITH_MPI := 1
	HOROVOD_WITHOUT_MPI := 0
	HOROVOD_CPU_OPERATIONS := MPI
	CUDA_SUFFIX := -cuda
	NCCL_BUILD_ARG := WITH_NCCL
        ifeq ($(WITH_NCCL),1)
		NCCL_BUILD_ARG := WITH_NCCL=1
        endif
	MPI_BUILD_ARG := WITH_MPI=1

	ifeq ($(WITH_AWS_TRACE),1)
		AWS_TRACE_ARG := WITH_AWS_TRACE=1
	else
		AWS_TRACE_ARG := WITH_AWS_TRACE=0
	endif

	ifeq ($(WITH_OFI),1)
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
ifeq ($(WITH_XCCL),1)
	XCCL_BUILD_ARG := WITH_XCCL=1
endif

# Get raw architecture from the host
RAW_ARCH := $(shell uname -m)

# Map to Docker/OCI standard names
ifeq ($(RAW_ARCH),x86_64)
    ARCH := amd64
else ifeq ($(RAW_ARCH),aarch64)
    ARCH := arm64
else ifeq ($(RAW_ARCH),arm64)
    ARCH := arm64
else
    ARCH := $(RAW_ARCH)
endif

# The following function dynamically builds these variables, and
# verifies it was able to correctly parse the provided image names:
#
#   USER_NGC_BASE_IMAGE        # Everything passed in
#   USER_NGC_IMAGE_FULL_NAME   # Everything after last /
#   USER_NGC_IMAGE_REPO        # Everything before last /
#   USER_NGC_IMAGE_NAME        # Everything left before :
#   USER_NGC_IMAGE_VER         # Everything left after :
#   USER_NGC_IMAGE_HPC         # <repo>/<name>-hpc:<ver>-$(ARCH)
#
#   USER_ROCM_BASE_IMAGE       # Everything passed in
#   USER_ROCM_IMAGE_FULL_NAME  # Everything after last /
#   USER_ROCM_IMAGE_REPO       # Everything before last /
#   USER_ROCM_IMAGE_NAME       # Everything left before :
#   USER_ROCM_IMAGE_VER        # Everything left after :
#   USER_ROCM_IMAGE_HPC        # <repo>/<name>-hpc:<ver>-$(ARCH)

define PARSE_IMAGE_VARS
    $(1)_BASE_IMAGE := $(2)
    
    # 1. Get everything AFTER the last slash (e.g., pytorch:25.06-py3)
    # We grab the last word after splitting the path by slashes
    $(1)_IMAGE_FULL_NAME := $$(lastword $$(subst /, ,$(2)))
    
    # 2. Get everything BEFORE the last slash
    # We replace the filename part with nothing in the original string
    $(1)_IMAGE_REPO := $$(subst /$$($(1)_IMAGE_FULL_NAME),,$(2))
    
    # 3. Split the full name into Name and Version using the colon
    $(1)_IMAGE_NAME := $$(word 1,$$(subst :, ,$$($(1)_IMAGE_FULL_NAME)))
    $(1)_IMAGE_VER  := $$(word 2,$$(subst :, ,$$($(1)_IMAGE_FULL_NAME)))
    
    # 4. Build final name tag
    $(1)_IMAGE_HPC  := $$($(1)_IMAGE_REPO)/$$($(1)_IMAGE_NAME)-hpc:$$($(1)_IMAGE_VER)-$(ARCH)

    # --- VALIDATION CHECKS ---
    $$(if $$($(1)_IMAGE_REPO),, $$(error ERROR: Could not parse Repository from $(2)))
    $$(if $$($(1)_IMAGE_NAME),, $$(error ERROR: Could not parse Image Name from $(2)))
    $$(if $$($(1)_IMAGE_VER),,  $$(error ERROR: Could not parse Image Version/Tag from $(2)))
    # -------------------------
endef

# Set defaults
NGC_DEFAULT  := $(NGC_PYTORCH_PREFIX):$(NGC_PYTORCH_VERSION)
ROCM_DEFAULT := $(ROCM_PT_PREFIX):$(ROCM_PT_VERSION)

# Use 'eval' to instantiate the variables globally
# Check if USER_IMAGE was passed in; if not, use the defaults.
ifeq ($(USER_IMAGE),)
    $(eval $(call PARSE_IMAGE_VARS,USER_NGC,$(NGC_DEFAULT)))
    $(eval $(call PARSE_IMAGE_VARS,USER_ROCM,$(ROCM_DEFAULT)))
else
    # If USER_IMAGE is provided, we map it to both or just the one being built
    $(eval $(call PARSE_IMAGE_VARS,USER_NGC,$(USER_IMAGE)))
    $(eval $(call PARSE_IMAGE_VARS,USER_ROCM,$(USER_IMAGE)))
endif

# Determine which platform was requested based on the goals
# This checks if the user typed 'nvidia' or 'amd' on the command line
ifeq ($(filter ngc,$(MAKECMDGOALS)),ngc)
    PLATFORM := ngc
endif
ifeq ($(filter rocm,$(MAKECMDGOALS)),rocm)
    PLATFORM := rocm
endif

# build docker tar file
.PHONY: tar
tar sif: TARGET_NAME := $(strip $(if $(filter ngc,$(MAKECMDGOALS)),$(subst :,-,$(subst /,-,$(USER_NGC_IMAGE_HPC))),\
                        $(if $(filter rocm,$(MAKECMDGOALS)),$(subst :,-,$(subst /,-,$(USER_ROCM_IMAGE_HPC))))))
tar sif: TARGET_TAG := $(strip $(if $(filter ngc,$(MAKECMDGOALS)),$(USER_NGC_IMAGE_HPC),\
                       $(if $(filter rocm,$(MAKECMDGOALS)),$(USER_ROCM_IMAGE_HPC))))

tar: $(PLATFORM)
	@echo "BUILD_TAR: $(PLATFORM) \"$(TARGET_NAME).tar\" from tag \"$(TARGET_TAG)\""
	$(DOCKER) save -o "$(TARGET_NAME).tar" $(TARGET_TAG)

# build pytorch sif
.PHONY: sif
sif: tar
	@echo "BUILD_SIF: $(TARGET_NAME).sif"
	@set -euo pipefail;                                                                     \
	TMP_SIF=$$(mktemp -d -p "$$(pwd)" -t sif-reg.XXXXXX);                                   \
	trap 'rm -rf "$$TMP_SIF" >/dev/null 2>&1 || true' EXIT;                                 \
	mkdir -p "$$TMP_SIF";                                                                   \
	if [ "$(USE_CWD_SIF)" = "1" ]; then                                                     \
	    echo "Using CWD for singularity tmp/cache: $$TMP_SIF";                              \
	    SING_ENV="SINGULARITY_TMPDIR=$$TMP_SIF SINGULARITY_CACHEDIR=$$TMP_SIF";             \
	else                                                                                    \
	    SING_ENV="";                                                                        \
	fi;                                                                                     \
	echo eval $$SING_ENV SINGULARITY_NOHTTPS=true NAMESPACE=""                                   \
	    singularity -vvv build $(TARGET_NAME).sif "docker-archive://$(TARGET_NAME).tar";    \
	eval $$SING_ENV SINGULARITY_NOHTTPS=true NAMESPACE=""                                   \
	    singularity -vvv build $(TARGET_NAME).sif "docker-archive://$(TARGET_NAME).tar";    \
	if [ "$(RM_SIF_TAR)" = "1" ]; then rm -f "$(TARGET_NAME).tar"; fi

# Build an HPC container using the base image provided by the user.
# This enables us to append the SS11 bits to an otherwise working
# user image to make it easier for users to deploy their containers on SS11.
.PHONY: ngc
ngc:
	@echo "USER_NGC_BASE_IMAGE: $(USER_NGC_BASE_IMAGE)"
	@echo "USER_NGC_IMAGE_REPO: $(USER_NGC_IMAGE_REPO)"
	@echo "USER_NGC_IMAGE_NAME: $(USER_NGC_IMAGE_NAME)"
	@echo "USER_NGC_IMAGE_VER: $(USER_NGC_IMAGE_VER)"
	@echo "USER_NGC_IMAGE_HPC: $(USER_NGC_IMAGE_HPC)"
	$(DOCKER) build -f Dockerfile-ngc-hpc $(BUILD_OPTS) \
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


# Build an HPC container using the base image provided by the user.
# This enables us to append the SS11 bits to an otherwise working
# user image to make it easier for users to deploy their containers on SS11.
.PHONY: rocm
rocm:
	@echo "USER_ROCM_BASE_IMAGE: $(USER_ROCM_BASE_IMAGE)"
	@echo "USER_ROCM_IMAGE_REPO: $(USER_ROCM_IMAGE_REPO)"
	@echo "USER_ROCM_IMAGE_NAME: $(USER_ROCM_IMAGE_NAME)"
	@echo "USER_ROCM_IMAGE_VER: $(USER_ROCM_IMAGE_VER)"
	@echo "USER_ROCM_IMAGE_HPC: $(USER_ROCM_IMAGE_HPC)"
	$(DOCKER) build -f Dockerfile-rocm-hpc $(BUILD_OPTS) \
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
