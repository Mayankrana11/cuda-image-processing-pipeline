// Copyright 2024 Jatin Pawar
//
// Licensed under the MIT License. See LICENSE for details.
//
// File: src/image_pipeline.cu
// Description: Scalable GPU-accelerated batch image processing pipeline.
//   Applies Gaussian blur (via NPP) and Sobel edge detection (custom CUDA
//   kernel with shared-memory tiling) to a directory of PGM images.

#include <cuda_runtime.h>
#include <nppdefs.h>
#include <nppi_filtering_functions.h>
#include <nppi_data_exchange_and_initialization.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Macro helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                                  \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define NPP_CHECK(call)                                                        \
  do {                                                                         \
    NppStatus st = (call);                                                     \
    if (st != NPP_SUCCESS) {                                                   \
      std::fprintf(stderr, "NPP error %d at %s:%d\n", st, __FILE__,           \
                   __LINE__);                                                   \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kTileW = 16;  // Shared-memory tile width
static constexpr int kTileH = 16;  // Shared-memory tile height
static constexpr int kHalo  = 1;   // Sobel needs 1-pixel halo

// ---------------------------------------------------------------------------
// PGM I/O helpers
// ---------------------------------------------------------------------------

/// Reads a binary PGM (P5) file. Returns true on success.
bool ReadPgm(const std::string& path, std::vector<unsigned char>* pixels,
             int* width, int* height) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return false;

  std::string magic;
  f >> magic;
  if (magic != "P5") return false;

  // Skip comments
  char c;
  f.get(c);
  while (c == '#') {
    std::string line;
    std::getline(f, line);
    f.get(c);
  }
  f.putback(c);

  int maxval;
  f >> *width >> *height >> maxval;
  f.get(c);  // consume trailing whitespace

  if (maxval != 255) return false;
  pixels->resize((*width) * (*height));
  f.read(reinterpret_cast<char*>(pixels->data()), pixels->size());
  return f.good();
}

/// Writes a binary PGM (P5) file. Returns true on success.
bool WritePgm(const std::string& path, const std::vector<unsigned char>& pixels,
              int width, int height) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) return false;
  f << "P5\n" << width << " " << height << "\n255\n";
  f.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
  return f.good();
}

// ---------------------------------------------------------------------------
// Custom Sobel edge-detection kernel (shared-memory tiling)
// ---------------------------------------------------------------------------

/// Each thread block loads a (kTileH+2*kHalo) x (kTileW+2*kHalo) tile into
/// shared memory and computes the Sobel magnitude for the inner kTileH x kTileW
/// pixels. Values are clamped to [0, 255].
__global__ void SobelKernel(const unsigned char* __restrict__ src,
                             unsigned char* __restrict__ dst, int width,
                             int height, int pitch) {
  // Shared memory including halo
  __shared__ unsigned char tile[kTileH + 2 * kHalo][kTileW + 2 * kHalo];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * kTileW + tx;  // Global column
  const int gy = blockIdx.y * kTileH + ty;  // Global row

  // Coordinates in shared memory (offset by halo)
  const int sx = tx + kHalo;
  const int sy = ty + kHalo;

  // Load centre pixel
  if (gx < width && gy < height) {
    tile[sy][sx] = src[gy * pitch + gx];
  } else {
    tile[sy][sx] = 0;
  }

  // Load halo pixels (left / right / top / bottom)
  if (tx == 0) {
    int lx = gx - kHalo;
    tile[sy][0] = (lx >= 0 && gy < height) ? src[gy * pitch + lx] : 0;
  }
  if (tx == kTileW - 1 || gx == width - 1) {
    int rx = gx + kHalo;
    tile[sy][sx + kHalo] = (rx < width && gy < height) ? src[gy * pitch + rx] : 0;
  }
  if (ty == 0) {
    int ly = gy - kHalo;
    tile[0][sx] = (ly >= 0 && gx < width) ? src[ly * pitch + gx] : 0;
  }
  if (ty == kTileH - 1 || gy == height - 1) {
    int by = gy + kHalo;
    tile[sy + kHalo][sx] = (by < height && gx < width) ? src[by * pitch + gx] : 0;
  }
  // Corners
  if (tx == 0 && ty == 0) {
    int lx = gx - 1, ly = gy - 1;
    tile[0][0] = (lx >= 0 && ly >= 0) ? src[ly * pitch + lx] : 0;
  }
  if (tx == kTileW - 1 && ty == 0) {
    int rx = gx + 1, ly = gy - 1;
    tile[0][sx + kHalo] = (rx < width && ly >= 0) ? src[ly * pitch + rx] : 0;
  }
  if (tx == 0 && ty == kTileH - 1) {
    int lx = gx - 1, by = gy + 1;
    tile[sy + kHalo][0] = (lx >= 0 && by < height) ? src[by * pitch + lx] : 0;
  }
  if (tx == kTileW - 1 && ty == kTileH - 1) {
    int rx = gx + 1, by = gy + 1;
    tile[sy + kHalo][sx + kHalo] = (rx < width && by < height) ? src[by * pitch + rx] : 0;
  }

  __syncthreads();

  if (gx >= width || gy >= height) return;

  // Sobel Gx and Gy
  const int gx_val =
      -1 * tile[sy - 1][sx - 1] + 1 * tile[sy - 1][sx + 1] +
      -2 * tile[sy    ][sx - 1] + 2 * tile[sy    ][sx + 1] +
      -1 * tile[sy + 1][sx - 1] + 1 * tile[sy + 1][sx + 1];

  const int gy_val =
      -1 * tile[sy - 1][sx - 1] - 2 * tile[sy - 1][sx] - 1 * tile[sy - 1][sx + 1] +
       1 * tile[sy + 1][sx - 1] + 2 * tile[sy + 1][sx] + 1 * tile[sy + 1][sx + 1];

  const int mag = min(255, static_cast<int>(
      sqrtf(static_cast<float>(gx_val * gx_val + gy_val * gy_val))));
  dst[gy * pitch + gx] = static_cast<unsigned char>(mag);
}

// ---------------------------------------------------------------------------
// ProcessImage: runs the full pipeline for one image
// ---------------------------------------------------------------------------

/// Runs Gaussian blur (NPP) -> Sobel edge detection (custom kernel) on a
/// single grayscale image. Returns the wall-clock time in milliseconds.
double ProcessImage(const std::string& input_path,
                    const std::string& blurred_path,
                    const std::string& edges_path) {
  // -- Read PGM from disk
  std::vector<unsigned char> h_src;
  int width = 0, height = 0;
  if (!ReadPgm(input_path, &h_src, &width, &height)) {
    std::cerr << "[SKIP] Cannot read PGM: " << input_path << "\n";
    return 0.0;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  const int n_pixels = width * height;

  // -- Allocate device memory (pitched)
  unsigned char* d_src   = nullptr;
  unsigned char* d_blur  = nullptr;
  unsigned char* d_edges = nullptr;
  size_t pitch_src = 0, pitch_blur = 0, pitch_edges = 0;

  CUDA_CHECK(cudaMallocPitch(&d_src,   &pitch_src,   width, height));
  CUDA_CHECK(cudaMallocPitch(&d_blur,  &pitch_blur,  width, height));
  CUDA_CHECK(cudaMallocPitch(&d_edges, &pitch_edges, width, height));

  // -- Upload source image
  CUDA_CHECK(cudaMemcpy2D(d_src, pitch_src, h_src.data(), width,
                           width, height, cudaMemcpyHostToDevice));

  // -- NPP Gaussian blur (5x5 mask)
  NppiSize roi = {width, height};

  NPP_CHECK(nppiFilterGauss_8u_C1R(
      d_src,  static_cast<int>(pitch_src),
      d_blur, static_cast<int>(pitch_blur),
      roi, NPP_MASK_SIZE_5_X_5));

  // -- Custom Sobel kernel
  dim3 block(kTileW, kTileH);
  dim3 grid((width  + kTileW - 1) / kTileW,
            (height + kTileH - 1) / kTileH);

  SobelKernel<<<grid, block>>>(d_blur, d_edges, width, height,
                                static_cast<int>(pitch_edges));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // -- Download results
  std::vector<unsigned char> h_blur(n_pixels);
  std::vector<unsigned char> h_edges(n_pixels);

  CUDA_CHECK(cudaMemcpy2D(h_blur.data(),  width, d_blur,  pitch_blur,
                           width, height, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy2D(h_edges.data(), width, d_edges, pitch_edges,
                           width, height, cudaMemcpyDeviceToHost));

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  // -- Free device memory
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_blur));
  CUDA_CHECK(cudaFree(d_edges));

  // -- Write outputs
  WritePgm(blurred_path, h_blur,  width, height);
  WritePgm(edges_path,   h_edges, width, height);

  return elapsed_ms;
}

// ---------------------------------------------------------------------------
// Utility: collect all *.pgm files in a directory
// ---------------------------------------------------------------------------
std::vector<std::string> CollectPgmFiles(const std::string& dir) {
  std::vector<std::string> files;
  DIR* dp = opendir(dir.c_str());
  if (!dp) {
    std::cerr << "Cannot open input directory: " << dir << "\n";
    return files;
  }
  struct dirent* entry;
  while ((entry = readdir(dp)) != nullptr) {
    std::string name(entry->d_name);
    if (name.size() > 4 &&
        name.substr(name.size() - 4) == ".pgm") {
      files.push_back(dir + "/" + name);
    }
  }
  closedir(dp);
  std::sort(files.begin(), files.end());
  return files;
}

// ---------------------------------------------------------------------------
// Print usage
// ---------------------------------------------------------------------------
static void PrintUsage(const char* prog) {
  std::printf(
      "Usage: %s --input <dir> --output <dir> [--max <N>]\n"
      "\n"
      "Options:\n"
      "  --input  <dir>  Directory containing input .pgm images\n"
      "  --output <dir>  Directory for blurred_* and edges_* output images\n"
      "  --max    <N>    Process at most N images (default: all)\n"
      "  --help          Show this message\n",
      prog);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  std::string input_dir;
  std::string output_dir;
  int max_images = INT_MAX;

  // -- Parse CLI arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help") {
      PrintUsage(argv[0]);
      return EXIT_SUCCESS;
    } else if (arg == "--input" && i + 1 < argc) {
      input_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--max" && i + 1 < argc) {
      max_images = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      PrintUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (input_dir.empty() || output_dir.empty()) {
    std::fprintf(stderr, "Error: --input and --output are required.\n");
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  // -- Print GPU info
  int device_id = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  std::printf("GPU: %s (Compute %d.%d, %d SMs, %.1f GB)\n",
              prop.name, prop.major, prop.minor,
              prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / 1e9);

  // -- Collect input files
  auto pgm_files = CollectPgmFiles(input_dir);
  if (pgm_files.empty()) {
    std::fprintf(stderr, "No .pgm files found in: %s\n", input_dir.c_str());
    return EXIT_FAILURE;
  }
  if (static_cast<int>(pgm_files.size()) > max_images) {
    pgm_files.resize(static_cast<size_t>(max_images));
  }

  std::printf("Found %zu image(s) to process.\n", pgm_files.size());
  std::printf("Output directory: %s\n\n", output_dir.c_str());

  // -- Process each image
  double total_ms = 0.0;
  int processed = 0;

  for (const auto& input_path : pgm_files) {
    // Extract basename
    auto slash = input_path.rfind('/');
    std::string basename =
        (slash == std::string::npos) ? input_path : input_path.substr(slash + 1);
    std::string stem = basename.substr(0, basename.size() - 4);

    std::string blurred_path = output_dir + "/blurred_" + stem + ".pgm";
    std::string edges_path   = output_dir + "/edges_"   + stem + ".pgm";

    double ms = ProcessImage(input_path, blurred_path, edges_path);
    if (ms > 0.0) {
      std::printf("  [%3d] %-40s  %.2f ms\n", processed + 1,
                  basename.c_str(), ms);
      total_ms += ms;
      ++processed;
    }
  }

  // -- Summary
  std::printf("\n=== Summary ===\n");
  std::printf("Images processed : %d\n", processed);
  std::printf("Total GPU time   : %.2f ms\n", total_ms);
  if (processed > 0) {
    std::printf("Avg  per image   : %.2f ms\n", total_ms / processed);
  }

  return EXIT_SUCCESS;
}
