#include "LBPH_Serial.h"
#include <chrono>
using Clock = std::chrono::high_resolution_clock;
// Initializes this LBPH Model. The current implementation is rather fixed
// as it uses the Extended Local Binary Patterns per default.
//
// radius, neighbors are used in the local binary patterns creation.
// grid_x, grid_y control the grid size of the spatial histograms.
LBPH_Serial::LBPH_Serial(int radius, int neighbors, int gridx, int gridy, double threshold)
    : LBPH(radius, neighbors, gridx, gridy, threshold) {
    std::cout << "LBPH_Serial Init\n";
}

// Initializes and computes this LBPH Model. The current implementation is
// rather fixed as it uses the Extended Local Binary Patterns per default.
//
// (radius=1), (neighbors=8) are used in the local binary patterns creation.
// (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
LBPH_Serial::LBPH_Serial(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, int radius, int neighbors, int gridx, int gridy, double threshold)
    : LBPH(radius, neighbors, gridx, gridy, threshold) {
    std::cout << "LBPH_Serial Init\n";
    train(src, labels);
}

LBPH_Serial::~LBPH_Serial() {}

// Computes a LBPH model with images in src and
// corresponding labels in labels.
void LBPH_Serial::train(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) {
    std::cout << "----- LBPH_Serial Train -----\n";
    train(images, labels, false);
}

// Updates this LBPH model with images in src and
// corresponding labels in labels.
void LBPH_Serial::update(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) {
    std::cout << "----- LBPH_Serial Update -----\n";
    train(images, labels, true);
}

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
inline void elbp_(const Image<_Tp>& src, Image<int>& dst, int radius, int neighbors) {
    dst = Image<int>(src.rows - 2 * radius, src.cols - 2 * radius);

    for (int n = 0; n < neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius * cos(2.0 * PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0 * PI * n / static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        // iterate through your data
        for (int i = radius; i < src.rows - radius; i++) {
            for (int j = radius; j < src.cols - radius; j++) {
                // calculate interpolated value
                float t = static_cast<float>(
                    w1 * src.at(i + fy, j + fx) +
                    w2 * src.at(i + fy, j + cx) +
                    w3 * src.at(i + cy, j + fx) +
                    w4 * src.at(i + cy, j + cx)
                    );
                // floating point precision, so check some machine-dependent epsilon
                float center = static_cast<float>(src.at(i, j));
                if ((t > center) || (std::abs(t - center) < std::numeric_limits<float>::epsilon())) {
                    dst.at(i - radius, j - radius) |= (1 << n);
                }
            }
        }
    }
}

template <typename T>
void elbp(const Image<T>& src, Image<int>& dst, int radius, int neighbors) {
    elbp_<T>(src, dst, radius, neighbors);
}

template <typename T>
Image<float> histc_(const Image<T>& src, int minVal = 0, int maxVal = 255, bool normed = false)
{
    int histSize = maxVal - minVal + 1;
    Image<float> hist(1, histSize);

    // Initialize to 0
    std::fill(hist.data.begin(), hist.data.end(), 0.f);

    // Calculate histogram
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int val = static_cast<int>(src.at(i, j));
            if (val >= minVal && val <= maxVal) {
                hist.at(0, val - minVal) += 1.0f;
            }
        }
    }

    if (normed && src.total() > 0) {
        float totalPixels = static_cast<float>(src.total());
        for (int k = 0; k < histSize; k++) {
            hist.at(0, k) /= totalPixels;
        }
    }

    return hist;
}

template <typename T>
Image<float> histc(const Image<T>& src, int minVal, int maxVal, bool normed) {
    return histc_<T>(src, minVal, maxVal, normed);
}

template <typename T>
Image<float> spatial_histogram(const Image<T>& src, int numPatterns,
    int grid_x, int grid_y, bool /*normed*/)
{
    // empty
    if (src.rows == 0 || src.cols == 0)
        return Image<float>(1, grid_x * grid_y * numPatterns); 

    // calculate LBP patch size
    int width = src.cols / grid_x;
    int height = src.rows / grid_y;
    // allocate memory for the spatial histogram
    Image<float> result(1, grid_x * grid_y * numPatterns);

    int resultIdx = 0;

    for (int i = 0; i < grid_y; i++) {
        for (int j = 0; j < grid_x; j++) {
            // Get sub-block ROI
            Image<T> cell = src.getROI(i * height, (i + 1) * height, j * width, (j + 1) * width);

            // Calculate the histogram of the cell
            Image<float> cell_hist = histc_(cell, 0, numPatterns - 1, true);

            // Copy the cell_hist data to the corresponding interval of result
            for (int k = 0; k < numPatterns; k++) {
                result.at(0, resultIdx++) = cell_hist.at(0, k);
            }
        }
    }

    return result;
}

//------------------------------------------------------------------------------
// wrapper to cv::elbp (extended local binary patterns)
//------------------------------------------------------------------------------
template <typename T>
Image<int> elbp(const Image<T>& src, int radius, int neighbors) {
    // Create output Image<int>
    Image<int> dst(src.rows - 2 * radius, src.cols - 2 * radius);
    // Clear
    std::fill(dst.data.begin(), dst.data.end(), 0); 

    // Call the actual elbp algorithm
    elbp_<T>(src, dst, radius, neighbors);

    return dst;
}

double compareHist(const Image<float>& H1, const Image<float>& H2) {
    if (H1.rows != 1 || H2.rows != 1 || H1.cols != H2.cols)
        throw std::invalid_argument("Histograms must be 1-row float Mats of same size.");

    double dist = 0.0;
    for (int i = 0; i < H1.cols; ++i) {
        float a = H1.data[i];
        float b = H2.data[i];
        if (a != 0.0f) {
            double d = static_cast<double>(a - b);
            dist += (d * d) / a;
        }
    }

    return dist;
}

void LBPH_Serial::train(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, bool preserveData) {
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

    double total_elbp_time = 0.0;
    double total_hist_time = 0.0;

    // store the spatial histograms of the original data
    for (size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        // calculate lbp image
        auto t1 = Clock::now();
        Image<int> lbp = elbp(src[sampleIdx], _radius, _neighbors);

        auto t2 = Clock::now();
        Image<float> hist = spatial_histogram(
            lbp, 
            static_cast<int>(pow(2.0, _neighbors)), // get spatial histogram from this lbp image
            _grid_x, 
            _grid_y, 
            true
        );
        auto t3 = Clock::now();
        // add to templates
        _histograms.push_back(hist);
        total_elbp_time += std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_hist_time += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    double avg_elbp = total_elbp_time / src.size();
    double avg_hist = total_hist_time / src.size();
    std::cout << "[train] average elbp time: " << avg_elbp << " ms\n";
    std::cout << "[train] average spatial_histogram time: " << avg_hist << " ms\n";
}

void LBPH_Serial::predict(const Image<uint8_t>& src, NearestNeighborCollector& collector) const {
    if (_histograms.empty()) {
        throw std::runtime_error("This LBPH model is not trained. Did you call the train method?");
    }

    // get the spatial histogram from input image
    Image<int> lbp_image = elbp(src, _radius, _neighbors);

    Image<float> query = spatial_histogram(
        lbp_image,
        static_cast<int>(pow(2.0, static_cast<double>(_neighbors))),
        _grid_x,
        _grid_y,
        true // normalized
    );

    // find 1-nearest neighbor
    collector.init(static_cast<int>(_histograms.size()));
    for (size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
        double dist = compareHist(_histograms[sampleIdx], query);// , HISTCMP_CHISQR_ALT);
        int label = _labels[sampleIdx];
        if (!collector.collect(label, dist)) break;
    }
}

void LBPH_Serial::predict(const Image<uint8_t>& src, int& label, double& confidence) const {
    NearestNeighborCollector collector;
    predict(src, collector);
    label = collector.getLabel();
    confidence = collector.getDistance();
}
