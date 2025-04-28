# Makefile for nccl-sanity

HPC_DIR = /container/hpc
BUILD_DIR ?= $(HPC_DIR)/tests/nccl-tests

# --- Compiler ---
NVCC = nvcc

# --- Executable Name ---
TARGET = $(BUILD_DIR)/nccl-sanity

# --- Source Files ---
SOURCES = nccl-sanity.c

# --- Configurable Paths (Change these or override on command line) ---
# Default path for MPI include directory (REQUIRED)
MPI_INCLUDE_DIR ?= /container/hpc/include
# Default path for MPI library directory (REQUIRED)
MPI_LIB_DIR     ?= /container/hpc/lib

# --- Compiler and Linker Flags ---
# Include directories
INCLUDES = -I$(MPI_INCLUDE_DIR)

# NVCC Flags (e.g., optimization, architecture)
# Add -gencode flags for specific architectures if needed, e.g.:
# NVCCFLAGS = -O2 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80
NVCCFLAGS = -O2

# Linker Flags (Library Paths)
LDFLAGS = -L$(MPI_LIB_DIR)

# Libraries to Link
LDLIBS = -lmpi -lcudart -lnccl

# --- Build Rules ---

# Default target
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(SOURCES)
	@echo "Compiling and Linking: $(SOURCES)"
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)
	@echo "Build complete: $(TARGET)"

# Rule to clean generated files
clean:
	@echo "Cleaning..."
	rm -f $(TARGET)

# Phony targets are not files
.PHONY: all clean
