#pragma once
#include "LBPH.h"
#include "lbphCuda.h"
class LBPH_Cuda : public LBPH {
    void train(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, bool preserveData);
    // device-side buffers for persistent use
    CudaHistogramBuffer gpu_buffer;
public:
    LBPH_Cuda(int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX);
    LBPH_Cuda(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX);
    ~LBPH_Cuda() override;

    void train(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) override;

    void update(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) override;

    void predict(const Image<uint8_t>& image, NearestNeighborCollector& collector) const override;

    void predict(const Image<uint8_t>& src, int& label, double& confidence) const override;

     void setParameters(std::shared_ptr<LBPHParams> params) override {};
};