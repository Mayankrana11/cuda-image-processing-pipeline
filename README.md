# Scalable GPU-Accelerated Batch Image Filtering & Sobel Edge Detection Pipeline

[![CUDA](https://img.shields.io/badge/CUDA-11.x%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)](#building)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A GPU-accelerated batch image-processing pipeline built with **CUDA C++**, featuring Gaussian blur, Sobel edge detection, pitched device memory, and shared-memory tiling.

Developed as a practical CUDA programming project inspired by the *CUDA at Scale for the Enterprise* Coursera course.

## Overview

Given a directory of grayscale PGM images, the pipeline performs two GPU-accelerated operations:

1. **Gaussian Blur** — applies a 5×5 Gaussian filter to smooth the input image and reduce high-frequency noise.
2. **Sobel Edge Detection** — applies a custom CUDA kernel to calculate image gradient magnitude and highlight edges.

The project has platform-specific implementations:

- **Windows:** Uses custom CUDA kernels for both Gaussian blur and Sobel edge detection.
- **Linux:** The original implementation uses NVIDIA Performance Primitives (NPP) for Gaussian blur and a custom CUDA Sobel kernel.

Both implementations use CUDA device memory and process images in batches. Processed images are saved as PGM files.

For each successfully processed input image, the pipeline generates:

- `blurred_<name>.pgm` — Gaussian-smoothed image.
- `edges_<name>.pgm` — Sobel edge map.

## Features

- Batch processing of grayscale PGM images.
- GPU-accelerated 5×5 Gaussian blur.
- Custom CUDA Sobel edge-detection kernel.
- Shared-memory tiling with halo pixels for Sobel processing.
- Pitched device memory using `cudaMallocPitch()` and `cudaMemcpy2D()`.
- Configurable input and output directories.
- Optional limit on the number of images processed.
- Windows and Linux build workflows.

## Repository Layout

```text
cuda-image-pipeline/
├── src/
│   ├── image_pipeline.cu          # Windows-compatible implementation
│   └── image_pipeline_linux.cu    # Original Linux/NPP implementation
├── scripts/
│   └── gen_pgm.py                 # Synthetic PGM test-image generator
├── data/
│   └── input/                     # Input PGM images
├── output/                        # Generated output images
├── bin/                           # Compiled executable
├── Makefile                       # Windows GNU Make build
├── Makefile.linux                 # Optional Linux-specific Makefile
├── run.sh                         # Linux helper script, if configured
└── README.md
```

The Linux source and Makefile names should match the files included in your checkout. If your repository uses a different layout, adjust the commands below accordingly.

## Prerequisites

| Requirement | Windows | Linux |
|---|---|---|
| NVIDIA GPU | CUDA-compatible GPU | CUDA-compatible GPU |
| NVIDIA driver | Compatible with installed CUDA Toolkit | Compatible with installed CUDA Toolkit |
| CUDA Toolkit | CUDA 13.4 in the tested setup | CUDA 11.x or later, subject to source/API compatibility |
| C++ compiler | MSVC Build Tools | GCC/G++ |
| Build system | GNU Make | GNU Make |
| Python 3 | Optional, for test-image generation | Optional, for test-image generation |

**Windows implementation:** Uses C++17 and does not require NPP for Gaussian blur.

**Linux implementation:** Uses NVIDIA NPP for Gaussian blur, so the corresponding NPP headers and libraries must be installed with the CUDA Toolkit.

The Windows Makefile targets `compute_89` / `sm_89`, suitable for NVIDIA RTX 40-series GPUs, including the RTX 4060. Change the architecture flags if building for a different GPU.

## Building

### Windows

The Windows build uses GNU Make with NVIDIA `nvcc` and the MSVC compiler toolchain.

#### 1. Open the correct terminal

Open an **x64 Native Tools Command Prompt for Visual Studio** or a Developer Command Prompt where MSVC and CUDA are available.

#### 2. Navigate to the repository

```bat
cd /d C:\Users\Mayank\Desktop\cuda-image-pipeline
```

#### 3. Build the project

```bat
make clean
make
```

The executable is generated at:

```text
bin\image_pipeline.exe
```

The Windows Makefile compiles only `src/image_pipeline.cu` and links against the CUDA runtime.

### Linux

The Linux implementation uses NVIDIA NPP for Gaussian blur and a custom CUDA Sobel kernel.

#### 1. Install the required tools

Ensure that the NVIDIA driver, CUDA Toolkit, CUDA compiler, C++ compiler, and GNU Make are installed.

Verify the toolchain:

```bash
nvidia-smi
nvcc --version
g++ --version
make --version
```

#### 2. Navigate to the repository

```bash
cd cuda-image-pipeline
```

#### 3. Compile the Linux source

If the Linux implementation is located at `src/image_pipeline_linux.cu`, build it with:

```bash
mkdir -p bin output

nvcc -std=c++14 -O2 \
    -gencode arch=compute_89,code=sm_89 \
    src/image_pipeline_linux.cu \
    -o bin/image_pipeline \
    -lnppif -lnppc -lcudart
```

Adjust the architecture flags for your GPU.

If you maintain a Linux-specific Makefile, you can instead use:

```bash
make -f Makefile.linux clean
make -f Makefile.linux
```

The executable is generated at:

```text
bin/image_pipeline
```

The Linux build command above assumes the original NPP function and headers are available in your installed CUDA version. If your CUDA version no longer provides the original function, use a compatible NPP implementation or port the Linux build to the custom CUDA Gaussian kernel.

## Generating Test Data

The optional Python helper can generate synthetic grayscale PGM images for benchmarking and functional testing.

For example, generate 200 images at 512×512 pixels:

```bash
python3 scripts/gen_pgm.py \
    --count 200 \
    --width 512 \
    --height 512 \
    --output data/input
```

On Windows, use:

```bat
python scripts\gen_pgm.py --count 200 --width 512 --height 512 --output data/input
```

Ensure the input directory contains valid binary PGM (`P5`) images with an 8-bit maximum pixel value of 255.

## Running the Pipeline

The executable accepts the same command-line arguments on both platforms.

### Windows

Run the executable directly:

```bat
bin\image_pipeline.exe --input data\input --output output
```

Process at most 50 images:

```bat
bin\image_pipeline.exe --input data\input --output output --max 50
```

Or use the Makefile's run target:

```bat
make run INPUT_DIR=data/input OUTPUT_DIR=output
```

To limit processing to 50 images:

```bat
make run INPUT_DIR=data/input OUTPUT_DIR=output MAX_IMAGES=50
```

### Linux

Run the executable directly:

```bash
./bin/image_pipeline \
    --input data/input \
    --output output
```

Process at most 50 images:

```bash
./bin/image_pipeline \
    --input data/input \
    --output output \
    --max 50
```

If your Linux `run.sh` script is configured for your environment, you can also run:

```bash
chmod +x run.sh
./run.sh
```

Check the script before running it to confirm which source file it compiles, where it expects input images, and whether it generates test data automatically.

## Command-Line Options

| Argument | Description |
|---|---|
| `--input <dir>` | Directory containing input `.pgm` images. |
| `--output <dir>` | Directory for generated blurred and edge images. |
| `--max <N>` | Process at most `N` images. |
| `--help` | Display usage information. |

Example:

```bash
./bin/image_pipeline --input data/input --output output --max 100
```

The executable creates the output directory if it does not exist.

## Algorithm Details

### Stage 1 — Gaussian Blur

The Gaussian filter smooths the input image before edge detection, reducing sensitivity to high-frequency noise.

The 5×5 Gaussian weights are constructed from the one-dimensional vector:

```text
[1, 4, 6, 4, 1]
```

The two-dimensional kernel is its outer product:

```text
 1   4   6   4   1
 4  16  24  16   4
 6  24  36  24   6
 4  16  24  16   4
 1   4   6   4   1
```

The weights sum to 256, so the filtered pixel value is normalized by 256.

On Windows, a custom CUDA kernel computes the Gaussian filter. The original Linux implementation uses NVIDIA NPP's Gaussian filtering functionality.

### Stage 2 — Sobel Edge Detection

The Sobel operator estimates horizontal and vertical image gradients using two 3×3 convolution masks.

Horizontal gradient:

```text
Gx = -1  0  1
     -2  0  2
     -1  0  1
```

Vertical gradient:

```text
Gy = -1 -2 -1
      0  0  0
      1  2  1
```

The gradient magnitude is computed as:

```text
magnitude = sqrt(Gx² + Gy²)
```

The output is clamped to the range `[0, 255]` and stored as an 8-bit grayscale image.

### Shared-Memory Tiling

The Sobel kernel uses 16×16 CUDA thread blocks. Each block processes a 16×16 output tile while loading an 18×18 shared-memory tile, including a one-pixel halo around the output region.

The halo provides the neighboring pixels required for the Sobel convolution. Shared-memory reuse reduces repeated global-memory accesses compared with a straightforward implementation that independently loads all neighboring pixels for every output.

### Pitched Device Memory

The pipeline allocates device images using `cudaMallocPitch()` and transfers them with `cudaMemcpy2D()`.

Pitched allocation allows CUDA to align row strides for efficient memory access. The implementation tracks the pitch of each allocation when addressing pixels and transferring image data.

## Example Output

After successful execution, the output directory contains files similar to:

```text
output/
├── blurred_image001.pgm
├── edges_image001.pgm
├── blurred_image002.pgm
├── edges_image002.pgm
└── ...
```

The blurred image contains the smoothed grayscale input. The edge image highlights regions of strong intensity change.

## Benchmarking

The pipeline reports the number of images processed, total measured processing time, and average time per image.

An example result from a Windows batch run:

```text
=== Summary ===
Images processed : 200
Total GPU time   : 229.66 ms
Avg  per image   : 1.15 ms
```

This corresponds to approximately 870 images per second for that particular workload.

**Benchmarking note:** Results depend on image dimensions, GPU model, driver and CUDA versions, data-transfer overhead, and system configuration. The reported timings should not be treated as a universal performance guarantee.

A meaningful CPU-versus-GPU comparison should use the same images, equivalent filtering operations, repeated trials, and clearly defined timing boundaries. GPU initialization, warm-up, memory transfers, and disk I/O should be accounted for consistently.

## Troubleshooting

### Windows: `dirent.h` not found

The original Linux implementation uses POSIX directory APIs that are unavailable in standard MSVC.

Use the Windows-compatible `src/image_pipeline.cu` implementation, which uses C++17 `std::filesystem`.

### Windows: `nppiFilterGauss_8u_C1R` is undefined

The Windows implementation uses a custom CUDA Gaussian blur kernel and does not require this NPP function.

Ensure the Windows Makefile compiles `src/image_pipeline.cu` and does not compile the original Linux source alongside it.

### Linux: NPP headers or symbols are missing

Verify that the NPP headers and libraries are installed with your CUDA Toolkit. Check the installed API documentation if the original Gaussian filtering function is unavailable in your version.

### No `.pgm` files found

Check that the input directory exists and contains valid `.pgm` files:

```bash
ls data/input
```

On Windows:

```bat
dir data\input
```

### CUDA compilation or runtime errors

Check the installed CUDA compiler, GPU driver compatibility, and target architecture. Ensure the executable is built for a supported GPU architecture.

## Lessons Learned

- **Pitched memory:** Correctly handling row pitch is essential when allocating and transferring two-dimensional GPU images.
- **Shared-memory tiling:** Halo-based tiling allows neighboring pixels to be reused across threads in the Sobel computation.
- **Custom CUDA kernels:** Implementing Gaussian blur directly in CUDA avoids dependency on a particular NPP Gaussian function and enables control over the filtering implementation.
- **Batch processing:** Processing independent images in a loop provides a straightforward foundation for future multi-stream or multi-GPU extensions.
- **Cross-platform development:** Linux-specific directory APIs and platform-dependent build assumptions need to be addressed when porting CUDA projects to Windows.

## Future Improvements

- Add a CPU baseline and report speedup against the GPU implementation.
- Experiment with CUDA streams to overlap memory transfers and computation.
- Benchmark multiple image dimensions and batch sizes.
- Explore separable Gaussian filtering to reduce arithmetic operations.
- Add automated correctness tests for blur and edge-detection outputs.
- Integrate reproducible benchmark reporting.

## License

MIT License © 2026 [Mayank Rana](https://github.com/Mayankrana11)

See [LICENSE](LICENSE) for details.
