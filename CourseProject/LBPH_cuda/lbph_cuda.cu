#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cfloat>
#include "image.hpp"
#include "lbph_cuda.h"

constexpr float M_PI = 3.14159265358979323846;

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    }


CudaHistogramBuffer::CudaHistogramBuffer() : d_histograms(nullptr), M(0), N(0) {}

CudaHistogramBuffer::~CudaHistogramBuffer() {
    release();
}

void CudaHistogramBuffer::upload(const std::vector<Image<float>>& histograms) {
    if (histograms.empty() || histograms[0].rows != 1)
        throw std::runtime_error("Histograms must be 1-row float images.");

    M = static_cast<int>(histograms.size());
    N = histograms[0].cols;

    // Flatten the histogram data
    std::vector<float> flat(M * N);
    for (int i = 0; i < M; ++i) {
        if (histograms[i].rows != 1 || histograms[i].cols != N)
            throw std::runtime_error("All histograms must be 1-row and same size.");
        std::memcpy(&flat[i * N], histograms[i].data.data(), N * sizeof(float));
    }

    // Allocate and upload to GPU
    cudaMalloc(&d_histograms, M * N * sizeof(float));
    cudaMemcpy(d_histograms, flat.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaHistogramBuffer::release() {
    if (d_histograms) {
        cudaFree(d_histograms);
        d_histograms = nullptr;
    }
    M = 0;
    N = 0;
}

// Global memory 
__global__ void compareHistKernel_global(const float* histograms, const float* query, float* distances, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    float dist = 0.0f;
    int offset = idx * N;

    for (int i = 0; i < N; ++i) {
        float a = histograms[offset + i];
        float b = query[i];
        if (a != 0.0f) {
            float d = a - b;
            dist += (d * d) / a;
        }
    }

    distances[idx] = dist;
}

__global__ void compareHistKernel_tiled(
    const float* __restrict__ histograms,
    const float* __restrict__ query,
    float* distances,
    int M, int N
) {
    extern __shared__ float sdata[]; // partial sums in shared memory

    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int threadsPerBlock = blockDim.x;

    if (blockId >= M) return;

    float sum = 0.0f;
    int base = blockId * N;

    for (int i = tid; i < N; i += threadsPerBlock) {
        float a = histograms[base + i];
        float b = query[i];
        if (a > 0.0f) {
            float d = a - b;
            sum += (d * d) / a;
        }
    }

    // Reduction in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = threadsPerBlock / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        distances[blockId] = sdata[0];
}

void cuda_compare_all_histograms(
    const CudaHistogramBuffer& buffer,
    const Image<float>& query,
    std::vector<float>& out_distances
) {
    const int M = buffer.M;
    const int N = buffer.N;
    if (query.rows != 1 || query.cols != N)
        throw std::runtime_error("Query histogram dimension mismatch");

    float* d_query = nullptr;
    float* d_distances = nullptr;
    cudaMalloc(&d_query, N * sizeof(float));
    cudaMalloc(&d_distances, M * sizeof(float));
    cudaMemcpy(d_query, query.data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    /*int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    compareHistKernel_global<<<blocksPerGrid, threadsPerBlock>>>(
        buffer.d_histograms, d_query, d_distances, M, N
    );*/

    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    compareHistKernel_tiled<<<M, threadsPerBlock, sharedMemSize>>>(
        buffer.d_histograms, d_query, d_distances, M, N
    );

    out_distances.resize(M);
    cudaMemcpy(out_distances.data(), d_distances, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_distances);
}

__global__ void elbp_kernel(const uint8_t* src, int* dst,
                            int rows, int cols, int radius, int neighbors, int src_step) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + radius;
    int j = blockIdx.x * blockDim.x + threadIdx.x + radius;

    if (i >= rows - radius || j >= cols - radius) return;

    float center = static_cast<float>(src[i * src_step + j]);
    int lbp = 0;

    for (int n = 0; n < neighbors; n++) {
        // use double precision for sampling point
        float x = static_cast<float>(radius * cos(2.0 * M_PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0 * M_PI * n / static_cast<float>(neighbors)));

        int fx = static_cast<int>(floorf(x));
        int fy = static_cast<int>(floorf(y));
        int cx = static_cast<int>(ceilf(x));
        int cy = static_cast<int>(ceilf(y));
        float tx = x - fx;
        float ty = y - fy;

        float w1 = (1.0f - tx) * (1.0f - ty);
        float w2 = tx * (1.0f - ty);
        float w3 = (1.0f - tx) * ty;
        float w4 = tx * ty;

        float t =
            w1 * src[(i + fy) * src_step + (j + fx)] +
            w2 * src[(i + fy) * src_step + (j + cx)] +
            w3 * src[(i + cy) * src_step + (j + fx)] +
            w4 * src[(i + cy) * src_step + (j + cx)];

        if ((t > center) || (fabsf(t - center) < FLT_EPSILON)) {
            lbp |= (1 << n);
        }
    }

    dst[(i - radius) * (cols - 2 * radius) + (j - radius)] = lbp;
}
__global__ void histogram_kernel(const int* lbp, float* hist,
                                 int rows, int cols,
                                 int grid_x, int grid_y, int numPatterns) {
    extern __shared__ int local_hist[];

    int gx = blockIdx.x;
    int gy = blockIdx.y;

    int tid = threadIdx.x;
    for (int i = tid; i < numPatterns; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    int cell_w = cols / grid_x;
    int cell_h = rows / grid_y;
    int x0 = gx * cell_w;
    int y0 = gy * cell_h;

    for (int i = threadIdx.y; i < cell_h; i += blockDim.y) {
        for (int j = threadIdx.x; j < cell_w; j += blockDim.x) {
            int val = lbp[(y0 + i) * cols + (x0 + j)];
            if (val >= 0 && val < numPatterns) {
                atomicAdd(&local_hist[val], 1);
            }
        }
    }
    __syncthreads();

    int idx = (gy * grid_x + gx) * numPatterns;
    for (int i = tid; i < numPatterns; i += blockDim.x) {
        hist[idx + i] = static_cast<float>(local_hist[i]) / (cell_w * cell_h);
    }
}

Image<float> gpu_elbp_and_histogram(const Image<uint8_t>& src,
                                    int radius, int neighbors,
                                    int grid_x, int grid_y) {
    int rows = src.rows, cols = src.cols;
    int lbp_rows = rows - 2 * radius;
    int lbp_cols = cols - 2 * radius;
    int numPatterns = 1 << neighbors;

    // Allocate device memory
    uint8_t* d_src;
    int* d_lbp;
    float* d_hist;

    size_t src_size = rows * cols * sizeof(uint8_t);
    size_t lbp_size = lbp_rows * lbp_cols * sizeof(int);
    size_t hist_size = grid_x * grid_y * numPatterns * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMemcpy(d_src, src.data.data(), src_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_lbp, lbp_size));
    CUDA_CHECK(cudaMalloc(&d_hist, hist_size));
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_size));

    // Launch ELBP kernel
    dim3 block(16, 16);
    dim3 grid((lbp_cols + block.x - 1) / block.x,
              (lbp_rows + block.y - 1) / block.y);
    elbp_kernel<<<grid, block>>>(d_src, d_lbp, rows, cols, radius, neighbors, cols);
    CUDA_CHECK(cudaGetLastError());

    // Launch histogram kernel
    dim3 hist_grid(grid_x, grid_y);
    dim3 hist_block(16, 16);
    size_t shared_mem = numPatterns * sizeof(int);

    histogram_kernel<<<hist_grid, hist_block, shared_mem>>>(
        d_lbp, d_hist, lbp_rows, lbp_cols, grid_x, grid_y, numPatterns);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back
    std::vector<float> host_hist(grid_x * grid_y * numPatterns);
    CUDA_CHECK(cudaMemcpy(host_hist.data(), d_hist, hist_size, cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(d_src);
    cudaFree(d_lbp);
    cudaFree(d_hist);

    // Normalize in CPU
    for (int y = 0; y < grid_y; ++y) {
        for (int x = 0; x < grid_x; ++x) {
            int idx = (y * grid_x + x) * numPatterns;
            float sum = 0;
            for (int i = 0; i < numPatterns; ++i)
                sum += host_hist[idx + i];
            for (int i = 0; i < numPatterns; ++i)
                host_hist[idx + i] /= sum > 0 ? sum : 1.0f;
        }
    }

    return Image<float>(1, grid_x * grid_y * numPatterns, 1, host_hist.data());
}

Image<int> gpu_elbp(const Image<uint8_t>& src, int radius, int neighbors) {
    int rows = src.rows, cols = src.cols;
    int lbp_rows = rows - 2 * radius;
    int lbp_cols = cols - 2 * radius;

    uint8_t* d_src;
    int* d_lbp;

    size_t src_size = rows * cols * sizeof(uint8_t);
    size_t lbp_size = lbp_rows * lbp_cols * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMemcpy(d_src, src.data.data(), src_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_lbp, lbp_size));

    dim3 block(16, 16);
    dim3 grid((lbp_cols + block.x - 1) / block.x,
              (lbp_rows + block.y - 1) / block.y);
    elbp_kernel<<<grid, block>>>(d_src, d_lbp, rows, cols, radius, neighbors, cols);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> host_lbp(lbp_rows * lbp_cols);
    CUDA_CHECK(cudaMemcpy(host_lbp.data(), d_lbp, lbp_size, cudaMemcpyDeviceToHost));

    cudaFree(d_src);
    cudaFree(d_lbp);

    return Image<int>(lbp_rows, lbp_cols, 1, host_lbp.data());
}

Image<float> gpu_histogram(const Image<int>& lbp,
                           int grid_x, int grid_y, int numPatterns) {
    int rows = lbp.rows;
    int cols = lbp.cols;

    int* d_lbp;
    float* d_hist;

    size_t lbp_size = rows * cols * sizeof(int);
    size_t hist_size = grid_x * grid_y * numPatterns * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_lbp, lbp_size));
    CUDA_CHECK(cudaMemcpy(d_lbp, lbp.data.data(), lbp_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_hist, hist_size));
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_size));

    dim3 hist_grid(grid_x, grid_y);
    dim3 hist_block(16, 16);
    size_t shared_mem = numPatterns * sizeof(int);

    histogram_kernel<<<hist_grid, hist_block, shared_mem>>>(
        d_lbp, d_hist, rows, cols, grid_x, grid_y, numPatterns);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> host_hist(grid_x * grid_y * numPatterns);
    CUDA_CHECK(cudaMemcpy(host_hist.data(), d_hist, hist_size, cudaMemcpyDeviceToHost));

    cudaFree(d_lbp);
    cudaFree(d_hist);

    // Normalize in CPU
    for (int y = 0; y < grid_y; ++y) {
        for (int x = 0; x < grid_x; ++x) {
            int idx = (y * grid_x + x) * numPatterns;
            float sum = 0;
            for (int i = 0; i < numPatterns; ++i)
                sum += host_hist[idx + i];
            for (int i = 0; i < numPatterns; ++i)
                host_hist[idx + i] /= sum > 0 ? sum : 1.0f;
        }
    }

    return Image<float>(1, grid_x * grid_y * numPatterns, 1, host_hist.data());
}

