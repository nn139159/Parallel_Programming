#include "LBPH_Cuda.h"
#include <chrono>
using Clock = std::chrono::high_resolution_clock;
// Initializes this LBPH Model. The current implementation is rather fixed
// as it uses the Extended Local Binary Patterns per default.
//
// radius, neighbors are used in the local binary patterns creation.
// grid_x, grid_y control the grid size of the spatial histograms.
LBPH_Cuda::LBPH_Cuda(int radius, int neighbors, int gridx, int gridy, double threshold)
    : LBPH(radius, neighbors, gridx, gridy, threshold) {}

// Initializes and computes this LBPH Model. The current implementation is
// rather fixed as it uses the Extended Local Binary Patterns per default.
//
// (radius=1), (neighbors=8) are used in the local binary patterns creation.
// (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
LBPH_Cuda::LBPH_Cuda(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, int radius, int neighbors, int gridx, int gridy, double threshold)
    : LBPH(radius, neighbors, gridx, gridy, threshold) {
    train(src, labels);
}

LBPH_Cuda::~LBPH_Cuda() {
    gpu_buffer.release();
    std::cout << "GPU Buffer release" << std::endl;
}

// Computes a LBPH model with images in src and
// corresponding labels in labels.
void LBPH_Cuda::train(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) {
    train(images, labels, false);
}

// Updates this LBPH model with images in src and
// corresponding labels in labels.
void LBPH_Cuda::update(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) {
    train(images, labels, true);
}

void LBPH_Cuda::train(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, bool preserveData) {
    if (src.empty()) {
        throw std::runtime_error("Empty training data was given. You'll need more than one sample to learn a model.");
    }
    if (labels.size() != src.size()) {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
            "The number of samples (src) must equal the number of labels (labels). Was len(samples)=%zu, len(labels)=%zu.",
            src.size(), labels.size());
        throw std::runtime_error(buffer);
    }

    // if this model should be trained without preserving old data, delete old model data
    if (!preserveData) {
        _labels.clear();
        _histograms.clear();
    }

    // append labels to _labels matrix
    _labels.insert(_labels.end(), labels.begin(), labels.end());

    double total_time = 0.0;

    // store the spatial histograms of the original data
    for (size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        // calculate lbp image
        auto t1 = Clock::now();
        Image<float> hist = gpu_elbp_and_histogram(src[sampleIdx], _radius, _neighbors, _grid_x, _grid_y);

        auto t2 = Clock::now();
        // add to templates
        _histograms.push_back(hist);
        total_time += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    double avg_time = total_time / src.size();;
    std::cout << "[train] Number of images: " << src.size() << "\n";
    std::cout << "[train] average elbp + spatial_histogram time: " << avg_time << " ms\n";
    gpu_buffer = CudaHistogramBuffer();
    gpu_buffer.upload(_histograms); 
}

void LBPH_Cuda::predict(const Image<uint8_t>& src, NearestNeighborCollector& collector) const {
    if (_histograms.empty()) { 
        throw std::runtime_error("This LBPH model is not trained. Did you call the train method?");
    }

    // get the spatial histogram from input image
    Image<float> query = gpu_elbp_and_histogram(src, _radius, _neighbors, _grid_x, _grid_y);
    // find 1-nearest neighbor
    std::vector<float> distances;
    cuda_compare_all_histograms(
        gpu_buffer, query, distances
    );
    collector.init(static_cast<int>(_histograms.size()));
    for (size_t i = 0; i < distances.size(); ++i) {
        if (!collector.collect(_labels[i], distances[i])) break;
    }
}

void LBPH_Cuda::predict(const Image<uint8_t>& src, int& label, double& confidence) const {
    NearestNeighborCollector collector;
    predict(src, collector);
    label = collector.getLabel();
    confidence = collector.getDistance();
}
