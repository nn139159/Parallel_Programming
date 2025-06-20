#pragma once
#include "Image.hpp"

class CudaHistogramBuffer {
public:
    float* d_histograms;
    int M; // number of histograms
    int N; // bins per histogram

    CudaHistogramBuffer();
    ~CudaHistogramBuffer();

    void upload(const std::vector<Image<float>>& histograms);
    void release();
};

void cuda_compare_all_histograms(
    const CudaHistogramBuffer& buffer,
    const Image<float>& query,
    std::vector<float>& out_distances);


Image<float> gpu_elbp_and_histogram(
    const Image<uint8_t>& src,
    int radius, int neighbors,
    int grid_x, int grid_y);

Image<int> gpu_elbp(const Image<uint8_t>& src, int radius, int neighbors);

Image<float> gpu_histogram(const Image<int>& lbp, int grid_x, int grid_y, int numPatterns);

