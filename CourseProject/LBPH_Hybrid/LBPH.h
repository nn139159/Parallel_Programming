#pragma once
#include <algorithm>
#include <random>
#include <limits>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cfloat>
#include <cmath>
#include "Image.hpp"
#include "NearestNeighborCollector.hpp"

constexpr double PI = 3.14159265358979323846;

struct LBPHParams {
    virtual ~LBPHParams() = default;
};

class LBPH {
protected:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;

    std::vector<Image<float>> _histograms;
    std::vector<int> _labels;

    // Allow derived classes to use this
    void train(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, bool preserveData);

public:
    LBPH(int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX)
        : _grid_x(gridx), _grid_y(gridy), _radius(radius), _neighbors(neighbors), _threshold(threshold){} 
    
    virtual ~LBPH() {}

    // Pure virtual functions make this an abstract class
    virtual void train(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) = 0;
    virtual void update(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) = 0;
    virtual void predict(const Image<uint8_t>& image, NearestNeighborCollector& collector) const = 0;
    virtual void predict(const Image<uint8_t>& src, int& label, double& confidence) const = 0;

    virtual void setParameters(std::shared_ptr<LBPHParams> params) = 0;

    bool empty() const{ return _labels.empty();}

    int getGridX() const { return _grid_x; }
    void setGridX(int val) { _grid_x = val; }
    int getGridY() const { return _grid_y; }
    void setGridY(int val) { _grid_y = val; }
    int getRadius() const { return _radius; }
    void setRadius(int val) { _radius = val; }
    int getNeighbors() const { return _neighbors; }
    void setNeighbors(int val) { _neighbors = val; }
    double getThreshold() const { return _threshold; }
    void setThreshold(double val) { _threshold = val; }

    const std::vector<Image<float>>& getHistograms() const { return _histograms; }
    const std::vector<int>& getLabels() const { return _labels; }
};
