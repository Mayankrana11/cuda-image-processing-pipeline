
# =============================================================================
# Makefile - GPU Image Processing Pipeline (Windows)
# Compatible with GNU Make, CUDA 13.4, and MSVC Build Tools
# =============================================================================

# CUDA Toolkit installation path
CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4
NVCC := "$(CUDA_PATH)/bin/nvcc.exe"

# Compiler options
NVCC_FLAGS := -std=c++17 -O2 -Xcompiler /W3

# GPU architecture: NVIDIA RTX 4060 (compute capability 8.9)
ARCH := -gencode arch=compute_89,code=sm_89

# Project directories
SRC_DIR := src
BIN_DIR := bin
TARGET := $(BIN_DIR)/image_pipeline.exe

# Compile ONLY the main CUDA source file
SOURCE := $(SRC_DIR)/image_pipeline.cu

# Default input/output paths
INPUT_DIR ?= data/input
OUTPUT_DIR ?= output
MAX_IMAGES ?=

.PHONY: all clean dirs run

# Default target
all: dirs $(TARGET)

# Create required directories
dirs:
	@if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"
	@if not exist "$(OUTPUT_DIR)" mkdir "$(OUTPUT_DIR)"

# Compile the CUDA application
$(TARGET): $(SOURCE) | dirs
	$(NVCC) $(NVCC_FLAGS) $(ARCH) "$(SOURCE)" -o "$@" -lcudart
	@if errorlevel 1 exit /b 1
	@echo Build successful: $@

# Remove the executable and build directory
clean:
	@if exist "$(BIN_DIR)" rmdir /s /q "$(BIN_DIR)"

# Build and run the pipeline
run: all
	@if not exist "$(INPUT_DIR)" (echo Error: input directory "$(INPUT_DIR)" not found. & exit /b 1)
	"$(TARGET)" --input "$(INPUT_DIR)" --output "$(OUTPUT_DIR)" $(if $(MAX_IMAGES),--max $(MAX_IMAGES),)
