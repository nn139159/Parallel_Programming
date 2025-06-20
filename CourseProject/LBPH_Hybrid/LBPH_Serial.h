#pragma once
#include "LBPH.h"

class LBPH_Serial : public LBPH {
    void train(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, bool preserveData);

public:
    LBPH_Serial(int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX);
    LBPH_Serial(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX);
    ~LBPH_Serial() override;

    void train(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) override;

    void update(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) override;

    void predict(const Image<uint8_t>& image, NearestNeighborCollector& collector) const override;

    void predict(const Image<uint8_t>& src, int& label, double& confidence) const override;

    void setParameters(std::shared_ptr<LBPHParams> params) override {};
};
