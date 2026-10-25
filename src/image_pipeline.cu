
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d - %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

static constexpr int kTileW = 16;
static constexpr int kTileH = 16;
static constexpr int kHalo = 1;

// ---------------------------------------------------------------------------
// PGM image input/output
// ---------------------------------------------------------------------------

bool ReadPgm(const std::string& path,
             std::vector<unsigned char>* pixels,
             int* width,
             int* height) {
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
        return false;
    }

    // Read PGM header while allowing comments between header fields.
    auto read_token = [&f](std::string& token) -> bool {
        token.clear();
        char c;

        while (f.get(c)) {
            if (c == '#') {
                std::string ignored;
                std::getline(f, ignored);
                continue;
            }

            if (c != ' ' && c != '\t' &&
                c != '\r' && c != '\n') {
                token.push_back(c);
                break;
            }
        }

        if (token.empty()) {
            return false;
        }

        while (f.get(c)) {
            if (c == '#') {
                std::string ignored;
                std::getline(f, ignored);
                break;
            }

            if (c == ' ' || c == '\t' ||
                c == '\r' || c == '\n') {
                break;
            }

            token.push_back(c);
        }

        return true;
    };

    std::string token;

    if (!read_token(token) || token != "P5") {
        return false;
    }

    if (!read_token(token)) {
        return false;
    }
    try {
        *width = std::stoi(token);
    } catch (...) {
        return false;
    }

    if (!read_token(token)) {
        return false;
    }
    try {
        *height = std::stoi(token);
    } catch (...) {
        return false;
    }

    int maxval;

    if (!read_token(token)) {
        return false;
    }
    try {
        maxval = std::stoi(token);
    } catch (...) {
        return false;
    }

    if (*width <= 0 || *height <= 0 || maxval != 255) {
        return false;
    }

    const size_t count =
        static_cast<size_t>(*width) * static_cast<size_t>(*height);

    pixels->resize(count);

    f.read(reinterpret_cast<char*>(pixels->data()),
           static_cast<std::streamsize>(count));

    return f.gcount() == static_cast<std::streamsize>(count);
}

bool WritePgm(const std::string& path,
              const std::vector<unsigned char>& pixels,
              int width,
              int height) {
    std::ofstream f(path, std::ios::binary);

    if (!f.is_open()) {
        std::cerr << "Cannot write output file: " << path << "\n";
        return false;
    }

    f << "P5\n" << width << " " << height << "\n255\n";

    f.write(reinterpret_cast<const char*>(pixels.data()),
            static_cast<std::streamsize>(pixels.size()));

    return f.good();
}

// ---------------------------------------------------------------------------
// 5x5 Gaussian blur CUDA kernel
// ---------------------------------------------------------------------------

__global__ void GaussianBlur5x5Kernel(
    const unsigned char* src,
    size_t src_pitch,
    unsigned char* dst,
    size_t dst_pitch,
    int width,
    int height) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Gaussian weights: [1, 4, 6, 4, 1].
    // Their 2D outer product sums to 256.
    const int weights[5] = {1, 4, 6, 4, 1};

    int sum = 0;

    for (int ky = -2; ky <= 2; ++ky) {
        const int iy = max(0, min(height - 1, y + ky));

        const unsigned char* row =
            src + static_cast<size_t>(iy) * src_pitch;

        for (int kx = -2; kx <= 2; ++kx) {
            const int ix = max(0, min(width - 1, x + kx));

            sum += static_cast<int>(row[ix]) *
                   weights[ky + 2] *
                   weights[kx + 2];
        }
    }

    dst[static_cast<size_t>(y) * dst_pitch + x] =
        static_cast<unsigned char>((sum + 128) / 256);
}

// ---------------------------------------------------------------------------
// Sobel edge detection CUDA kernel using shared-memory tiling
// ---------------------------------------------------------------------------

__global__ void SobelKernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height,
    size_t src_pitch,
    size_t dst_pitch) {

    __shared__ unsigned char
        tile[kTileH + 2 * kHalo][kTileW + 2 * kHalo];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int gx = blockIdx.x * kTileW + tx;
    const int gy = blockIdx.y * kTileH + ty;

    const int sx = tx + kHalo;
    const int sy = ty + kHalo;

    // Load the center pixel.
    if (gx < width && gy < height) {
        tile[sy][sx] =
            src[static_cast<size_t>(gy) * src_pitch + gx];
    } else {
        tile[sy][sx] = 0;
    }

    // Load left and right halo pixels.
    if (tx == 0) {
        const int lx = gx - 1;

        tile[sy][0] =
            (lx >= 0 && gy < height)
                ? src[static_cast<size_t>(gy) * src_pitch + lx]
                : 0;
    }

    if (tx == kTileW - 1 || gx == width - 1) {
        const int rx = gx + 1;

        tile[sy][sx + 1] =
            (rx < width && gy < height)
                ? src[static_cast<size_t>(gy) * src_pitch + rx]
                : 0;
    }

    // Load top and bottom halo pixels.
    if (ty == 0) {
        const int ly = gy - 1;

        tile[0][sx] =
            (ly >= 0 && gx < width)
                ? src[static_cast<size_t>(ly) * src_pitch + gx]
                : 0;
    }

    if (ty == kTileH - 1 || gy == height - 1) {
        const int by = gy + 1;

        tile[sy + 1][sx] =
            (by < height && gx < width)
                ? src[static_cast<size_t>(by) * src_pitch + gx]
                : 0;
    }

    // Load halo corners.
    if (tx == 0 && ty == 0) {
        const int lx = gx - 1;
        const int ly = gy - 1;

        tile[0][0] =
            (lx >= 0 && ly >= 0)
                ? src[static_cast<size_t>(ly) * src_pitch + lx]
                : 0;
    }

    if (tx == kTileW - 1 && ty == 0) {
        const int rx = gx + 1;
        const int ly = gy - 1;

        tile[0][sx + 1] =
            (rx < width && ly >= 0)
                ? src[static_cast<size_t>(ly) * src_pitch + rx]
                : 0;
    }

    if (tx == 0 && ty == kTileH - 1) {
        const int lx = gx - 1;
        const int by = gy + 1;

        tile[sy + 1][0] =
            (lx >= 0 && by < height)
                ? src[static_cast<size_t>(by) * src_pitch + lx]
                : 0;
    }

    if (tx == kTileW - 1 && ty == kTileH - 1) {
        const int rx = gx + 1;
        const int by = gy + 1;

        tile[sy + 1][sx + 1] =
            (rx < width && by < height)
                ? src[static_cast<size_t>(by) * src_pitch + rx]
                : 0;
    }

    // Every thread must reach the barrier before any thread returns.
    __syncthreads();

    if (gx >= width || gy >= height) {
        return;
    }

    // Sobel horizontal gradient.
    const int gx_val =
        -static_cast<int>(tile[sy - 1][sx - 1])
        + static_cast<int>(tile[sy - 1][sx + 1])
        - 2 * static_cast<int>(tile[sy][sx - 1])
        + 2 * static_cast<int>(tile[sy][sx + 1])
        - static_cast<int>(tile[sy + 1][sx - 1])
        + static_cast<int>(tile[sy + 1][sx + 1]);

    // Sobel vertical gradient.
    const int gy_val =
        -static_cast<int>(tile[sy - 1][sx - 1])
        - 2 * static_cast<int>(tile[sy - 1][sx])
        - static_cast<int>(tile[sy - 1][sx + 1])
        + static_cast<int>(tile[sy + 1][sx - 1])
        + 2 * static_cast<int>(tile[sy + 1][sx])
        + static_cast<int>(tile[sy + 1][sx + 1]);

    const float magnitude = sqrtf(
        static_cast<float>(gx_val * gx_val + gy_val * gy_val));

    const int result = min(255, static_cast<int>(magnitude));

    dst[static_cast<size_t>(gy) * dst_pitch + gx] =
        static_cast<unsigned char>(result);
}

// ---------------------------------------------------------------------------
// Process one image: Gaussian blur -> Sobel edge detection
// ---------------------------------------------------------------------------

double ProcessImage(
    const std::string& input_path,
    const std::string& blurred_path,
    const std::string& edges_path) {

    std::vector<unsigned char> h_src;
    int width = 0;
    int height = 0;

    if (!ReadPgm(input_path, &h_src, &width, &height)) {
        std::cerr << "[SKIP] Cannot read PGM: "
                  << input_path << "\n";
        return 0.0;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    const size_t row_bytes = static_cast<size_t>(width);

    unsigned char* d_src = nullptr;
    unsigned char* d_blur = nullptr;
    unsigned char* d_edges = nullptr;

    size_t pitch_src = 0;
    size_t pitch_blur = 0;
    size_t pitch_edges = 0;

    CUDA_CHECK(cudaMallocPitch(
        reinterpret_cast<void**>(&d_src),
        &pitch_src, row_bytes, height));

    CUDA_CHECK(cudaMallocPitch(
        reinterpret_cast<void**>(&d_blur),
        &pitch_blur, row_bytes, height));

    CUDA_CHECK(cudaMallocPitch(
        reinterpret_cast<void**>(&d_edges),
        &pitch_edges, row_bytes, height));

    CUDA_CHECK(cudaMemcpy2D(
        d_src, pitch_src,
        h_src.data(), row_bytes,
        row_bytes, height,
        cudaMemcpyHostToDevice));

    // Gaussian blur.
    const dim3 blur_block(16, 16);

    const dim3 blur_grid(
        (width + blur_block.x - 1) / blur_block.x,
        (height + blur_block.y - 1) / blur_block.y);

    GaussianBlur5x5Kernel<<<blur_grid, blur_block>>>(
        d_src, pitch_src,
        d_blur, pitch_blur,
        width, height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sobel edge detection.
    const dim3 block(kTileW, kTileH);

    const dim3 grid(
        (width + kTileW - 1) / kTileW,
        (height + kTileH - 1) / kTileH);

    SobelKernel<<<grid, block>>>(
        d_blur, d_edges,
        width, height,
        pitch_blur, pitch_edges);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download both output images.
    std::vector<unsigned char> h_blur(
        static_cast<size_t>(width) * height);

    std::vector<unsigned char> h_edges(
        static_cast<size_t>(width) * height);

    CUDA_CHECK(cudaMemcpy2D(
        h_blur.data(), row_bytes,
        d_blur, pitch_blur,
        row_bytes, height,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy2D(
        h_edges.data(), row_bytes,
        d_edges, pitch_edges,
        row_bytes, height,
        cudaMemcpyDeviceToHost));

    auto t_end = std::chrono::high_resolution_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            t_end - t_start).count();

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_edges));

    if (!WritePgm(blurred_path, h_blur, width, height)) {
        std::cerr << "Failed to write blurred image: "
                  << blurred_path << "\n";
    }

    if (!WritePgm(edges_path, h_edges, width, height)) {
        std::cerr << "Failed to write edge image: "
                  << edges_path << "\n";
    }

    return elapsed_ms;
}

// ---------------------------------------------------------------------------
// Collect PGM files from a directory
// ---------------------------------------------------------------------------

std::vector<std::string> CollectPgmFiles(
    const std::string& dir) {

    namespace fs = std::filesystem;

    std::vector<std::string> files;
    std::error_code ec;

    const fs::path input_path(dir);

    if (!fs::exists(input_path, ec) || ec ||
        !fs::is_directory(input_path, ec) || ec) {

        std::cerr << "Cannot open input directory: "
                  << dir << "\n";
        return files;
    }

    fs::directory_iterator it(input_path, ec);
    const fs::directory_iterator end;

    if (ec) {
        std::cerr << "Cannot open input directory: "
                  << dir << " (" << ec.message() << ")\n";
        return files;
    }

    for (; it != end; it.increment(ec)) {
        if (ec) {
            std::cerr << "Error reading input directory: "
                      << ec.message() << "\n";
            break;
        }

        std::error_code entry_ec;

        if (!it->is_regular_file(entry_ec) || entry_ec) {
            continue;
        }

        if (it->path().extension() == ".pgm") {
            files.push_back(it->path().string());
        }
    }

    std::sort(files.begin(), files.end());

    return files;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void PrintUsage(const char* prog) {
    std::printf(
        "Usage: %s --input <dir> --output <dir> [--max <N>]\n"
        "\n"
        "Options:\n"
        "  --input  <dir>  Directory containing input .pgm images\n"
        "  --output <dir>  Directory for blurred_* and edges_* images\n"
        "  --max    <N>    Process at most N images\n"
        "  --help          Show this message\n",
        prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string input_dir;
    std::string output_dir;
    int max_images = INT_MAX;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--help") {
            PrintUsage(argv[0]);
            return EXIT_SUCCESS;
        } else if (arg == "--input" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--max" && i + 1 < argc) {
            char* end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);

            if (end == argv[i] || *end != '\0' ||
                value < 1 || value > INT_MAX) {
                std::fprintf(stderr, "Invalid --max value.\n");
                return EXIT_FAILURE;
            }

            max_images = static_cast<int>(value);
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n",
                         arg.c_str());
            PrintUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (input_dir.empty() || output_dir.empty()) {
        std::fprintf(stderr,
                     "Error: --input and --output are required.\n");
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // Display GPU information.
    int device_id = 0;
    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::printf(
        "GPU: %s (Compute %d.%d, %d SMs, %.1f GB)\n",
        prop.name,
        prop.major,
        prop.minor,
        prop.multiProcessorCount,
        static_cast<double>(prop.totalGlobalMem) / 1e9);

    // Create the output directory if necessary.
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::create_directories(fs::path(output_dir), ec);

    if (ec) {
        std::fprintf(stderr,
                     "Cannot create output directory: %s (%s)\n",
                     output_dir.c_str(),
                     ec.message().c_str());
        return EXIT_FAILURE;
    }

    // Find input images.
    const auto pgm_files = CollectPgmFiles(input_dir);

    if (pgm_files.empty()) {
        std::fprintf(stderr,
                     "No .pgm files found in: %s\n",
                     input_dir.c_str());
        return EXIT_FAILURE;
    }

    const size_t image_count = std::min(
        pgm_files.size(), static_cast<size_t>(max_images));

    std::printf("Found %zu image(s) to process.\n", image_count);
    std::printf("Output directory: %s\n\n", output_dir.c_str());

    double total_ms = 0.0;
    int processed = 0;

    for (size_t i = 0; i < image_count; ++i) {
        const std::string& input_path = pgm_files[i];

        const fs::path input_file(input_path);
        const std::string basename = input_file.filename().string();
        const std::string stem = input_file.stem().string();

        const fs::path output_path(output_dir);

        const std::string blurred_path =
            (output_path / ("blurred_" + stem + ".pgm")).string();

        const std::string edges_path =
            (output_path / ("edges_" + stem + ".pgm")).string();

        const double ms = ProcessImage(
            input_path, blurred_path, edges_path);

        if (ms > 0.0) {
            std::printf("  [%3d] %-40s  %.2f ms\n",
                        processed + 1, basename.c_str(), ms);

            total_ms += ms;
            ++processed;
        }
    }

    std::printf("\n=== Summary ===\n");
    std::printf("Images processed : %d\n", processed);
    std::printf("Total GPU time   : %.2f ms\n", total_ms);

    if (processed > 0) {
        std::printf("Avg  per image   : %.2f ms\n",
                    total_ms / processed);
    }

    return processed > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
