#pragma once
#include <algorithm>
#include <random>
#include <limits>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cfloat>
#include "Image.hpp"
#include "NearestNeighborCollector.hpp"

constexpr double PI = 3.14159265358979323846;

struct LBPH_OpenMP_Params {
    int numThreads = 4;
    bool useSIMD = true;
};

class LBPH {
private:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;
    int _numThreads = 4;
    bool _useSIMD = true;
    std::vector<Image<float>> _histograms;
    std::vector<int> _labels;

    void train(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, bool preserveData);

public:
    LBPH(int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX);
    LBPH(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX);
    ~LBPH();

    void train(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels);

    void update(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels);

    void predict(const Image<uint8_t>& image, NearestNeighborCollector& collector) const;

    void predict(const Image<uint8_t>& src, int& label, double& confidence) const;

    bool empty() const;

    int getGridX() const;
    void setGridX(int val);
    int getGridY() const;
    void setGridY(int val);
    int getRadius() const;
    void setRadius(int val);
    int getNeighbors() const;
    void setNeighbors(int val);
    double getThreshold() const;
    void setThreshold(double val);

    const std::vector<Image<float>>& getHistograms() const;
    const std::vector<int>& getLabels() const;

    void setParameters(std::shared_ptr<LBPH_OpenMP_Params> params);
};
