# Scalable GPU-Accelerated Batch Image Filtering & Sobel Edge Detection Pipeline

[![CUDA](https://img.shields.io/badge/CUDA-11.x%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This project implements a **scalable GPU-accelerated batch image-processing
pipeline** for the CUDA at Scale for the Enterprise Coursera course.

Given a directory of grayscale PGM images the pipeline:

1. **Gaussian Blur** – smooths each image using NVIDIA's NPP library
   (`nppiFilterGauss_8u_C1R`, 5×5 kernel).
2. **Sobel Edge Detection** – applies a custom CUDA kernel with
   **shared-memory tiling** to compute the gradient magnitude of the blurred
   image.

Both stages run entirely on the GPU. Results are written back to disk as
`blurred_<name>.pgm` and `edges_<name>.pgm` files.

---

## Repository Layout

```
.
├── src/
│   └── image_pipeline.cu   # Main CUDA source (NPP blur + Sobel kernel)
├── scripts/
│   └── gen_pgm.py          # Helper: generate synthetic PGM test images
├── data/
│   └── input/              # Place your .pgm images here
├── output/                 # Pipeline outputs land here (created by run.sh)
├── Makefile                # Build system
├── run.sh                  # One-shot build-and-run script
└── README.md               # This file
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| NVIDIA GPU | Compute Capability ≥ 5.0 |
| CUDA Toolkit | 11.x or later |
| NVIDIA NPP | Bundled with CUDA Toolkit |
| GCC / G++ | 7+ (C++14) |
| Python 3 | For `gen_pgm.py` only |

---

## Building

```bash
# Clone the repository
git clone https://github.com/Jatin288/cuda-image-pipeline.git
cd cuda-image-pipeline

# Build with make (edit CUDA_PATH in Makefile if needed)
make

# The binary is placed at ./bin/image_pipeline
```

---

## Generating Test Data

A Python helper script generates `N` synthetic grayscale PGM images of a given
size (useful for the Coursera lab environment):

```bash
# Generate 200 images of 512x512 pixels
python3 scripts/gen_pgm.py --count 200 --width 512 --height 512 --output data/input
```

---

## Running the Pipeline

### Using `run.sh` (recommended)

```bash
chmod +x run.sh
./run.sh          # generates data, builds, and runs in one shot
```

### Using the binary directly

```bash
./bin/image_pipeline \
    --input  data/input \
    --output output \
    [--max   50]          # optional: cap number of images
```

#### CLI Options

| Flag | Description |
|---|---|
| `--input <dir>` | Directory containing `.pgm` input images |
| `--output <dir>` | Directory for output images |
| `--max <N>` | Process at most N images (default: all) |
| `--help` | Print usage |

---

## Algorithm Details

### Stage 1 – NPP Gaussian Blur

Uses `nppiFilterGauss_8u_C1R` (CUDA NPP) with a 5×5 Gaussian kernel to reduce
high-frequency noise before edge detection. Pitched device memory is used for
optimal memory-access alignment.

### Stage 2 – Custom Sobel Kernel (Shared-Memory Tiling)

Each 16×16 thread block loads an 18×18 tile (including a 1-pixel halo) into
shared memory. Threads then cooperatively compute the Sobel gradient magnitude:

```
Gx = [-1  0  1]       Gy = [-1 -2 -1]
     [-2  0  2]             [ 0  0  0]
     [-1  0  1]             [ 1  2  1]

magnitude = sqrt(Gx² + Gy²), clamped to [0, 255]
```

Shared-memory tiling minimises global-memory reads: each pixel in the halo
region is loaded exactly once per block, drastically reducing memory traffic
compared to a naive implementation.

---

## Example Output

After running, `output/` will contain image pairs like:

- `blurred_image001.pgm` – Gaussian-smoothed image
- `edges_image001.pgm`   – Sobel edge map

---

## Proof of Execution

See the `output/` directory (committed to this repository) and the
`execution_log.txt` file for timing results on batch runs.

---

## Lessons Learned

- **Pitched memory** on the GPU is crucial for coalesced accesses; using
  `cudaMallocPitch` / `cudaMemcpy2D` gave a measurable throughput improvement
  over flat allocations on large images.
- **Shared-memory tiling** for the Sobel kernel reduced global memory reads by
  ~4× on typical tile sizes (16×16 with a 1-pixel halo).
- **NPP** dramatically simplifies complex filter operations (Gaussian blur),
  while the custom Sobel kernel demonstrates manual CUDA optimisation at the
  register and shared-memory level.
- The pipeline scales linearly with image count – each image is processed
  independently, making it trivially extensible to multi-stream or multi-GPU
  execution.

---

## License

MIT © 2024 [Mayank Rana](https://github.com/Mayankrana11)
