#include "LBPH_OpenMP.h"
#include <cmath>
#include <chrono>
#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
using Clock = std::chrono::high_resolution_clock;
// Initializes this LBPH Model. The current implementation is rather fixed
// as it uses the Extended Local Binary Patterns per default.
//
// radius, neighbors are used in the local binary patterns creation.
// grid_x, grid_y control the grid size of the spatial histograms.
LBPH_OpenMP::LBPH_OpenMP(int radius, int neighbors, int gridx, int gridy, double threshold)
    : LBPH(radius, neighbors, gridx, gridy, threshold) 
{
    std::cout << "LBPH_OpenMP Init::";
#if defined(__AVX2__)
    std::cout << "Using AVX2" << std::endl;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::cout << "Using ARM NEON" << std::endl;
#else
    std::cout << "No SIMD extensions" << std::endl;
#endif
}

// Initializes and computes this LBPH Model. The current implementation is
// rather fixed as it uses the Extended Local Binary Patterns per default.
//
// (radius=1), (neighbors=8) are used in the local binary patterns creation.
// (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
LBPH_OpenMP::LBPH_OpenMP(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, int radius, int neighbors, int gridx, int gridy, double threshold)
    : LBPH(radius, neighbors, gridx, gridy, threshold) {
    std::cout << "LBPH_OpenMP Init";
#if defined(__AVX2__)
    std::cout << "Using AVX2" << std::endl;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::cout << "Using ARM NEON" << std::endl;
#else
    std::cout << "No SIMD extensions" << std::endl;
#endif
    train(src, labels);
}

LBPH_OpenMP::~LBPH_OpenMP() {}

// Computes a LBPH model with images in src and
// corresponding labels in labels.
void LBPH_OpenMP::train(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) {
    std::cout << "----- LBPH_OpenMP Train -----\n";
    if (_useSIMD) {
        std::cout << "SIMD enabled" << std::endl;
    } else {
        std::cout << "SIMD not enabled" << std::endl;
    }
    omp_set_num_threads(_numThreads);
    std::cout << "Number of cores used: " << omp_get_max_threads() << "\n";

    train(images, labels, false);
}

// Updates this LBPH model with images in src and
// corresponding labels in labels.
void LBPH_OpenMP::update(const std::vector<Image<uint8_t>>& images, const std::vector<int>& labels) {
    std::cout << "----- LBPH_OpenMP Update -----\n";
    if (_useSIMD) {
        std::cout << "SIMD enabled" << std::endl;
    } else {
        std::cout << "SIMD not enabled" << std::endl;
    }
    omp_set_num_threads(_numThreads);
    std::cout << " Number of cores used: " << omp_get_max_threads() << "\n";

    train(images, labels, true);
}

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
#if defined(__AVX2__)
inline __m256 _mm256_abs_ps(__m256 x) {
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return _mm256_and_ps(x, sign_mask);
}
template <typename _Tp> static
void elbp_avx2(const Image<_Tp>& src, Image<int>& dst, int, int) {
    const int radius = 1;
    const int neighbors = 8;
    dst = Image<int>(src.rows - 2 * radius, src.cols - 2 * radius);

    // Precompute all offsets and weights
    int fx[8], fy[8], cx[8], cy[8];
    float w1[8], w2[8], w3[8], w4[8];
    for (int n = 0; n < neighbors; ++n) {
        float x = radius * cos(2.0f * PI * n / neighbors);
        float y = -radius * sin(2.0f * PI * n / neighbors);
        fx[n] = static_cast<int>(floor(x));
        fy[n] = static_cast<int>(floor(y));
        cx[n] = static_cast<int>(ceil(x));
        cy[n] = static_cast<int>(ceil(y));
        float tx = x - fx[n];
        float ty = y - fy[n];
        w1[n] = (1 - tx) * (1 - ty);
        w2[n] = tx * (1 - ty);
        w3[n] = (1 - tx) * ty;
        w4[n] = tx * ty;
    }

    const float epsilon = std::numeric_limits<float>::epsilon();

    #pragma omp parallel for
    for (int i = radius; i < src.rows - radius; ++i) {
        for (int j = radius; j <= src.cols - radius - 8; j += 8) {
            __m256 center = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src.at(i, j)))
            ));

            int code[8] = {0};
            for (int n = 0; n < neighbors; ++n) {
                __m256 w1v = _mm256_set1_ps(w1[n]);
                __m256 w2v = _mm256_set1_ps(w2[n]);
                __m256 w3v = _mm256_set1_ps(w3[n]);
                __m256 w4v = _mm256_set1_ps(w4[n]);

                __m128i n1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src.at(i + fy[n], j + fx[n])));
                __m128i n2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src.at(i + fy[n], j + cx[n])));
                __m128i n3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src.at(i + cy[n], j + fx[n])));
                __m128i n4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src.at(i + cy[n], j + cx[n])));

                __m256 p1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(n1));
                __m256 p2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(n2));
                __m256 p3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(n3));
                __m256 p4 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(n4));

                __m256 interp = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(w1v, p1), _mm256_mul_ps(w2v, p2)),
                    _mm256_add_ps(_mm256_mul_ps(w3v, p3), _mm256_mul_ps(w4v, p4))
                );

                // (t > center) || (abs(t - center) < eps)
                __m256 gt = _mm256_cmp_ps(interp, center, _CMP_GT_OQ);
                __m256 eq = _mm256_cmp_ps(_mm256_abs_ps(_mm256_sub_ps(interp, center)), _mm256_set1_ps(epsilon), _CMP_LT_OQ);
                __m256 mask = _mm256_or_ps(gt, eq);

                __m256i bits = _mm256_castps_si256(mask);
                __m256i shift = _mm256_set1_epi32(1 << n);
                __m256i contrib = _mm256_and_si256(bits, shift);

                alignas(32) int temp[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(temp), contrib);

                for (int k = 0; k < 8; ++k) code[k] |= temp[k];
            }

            for (int k = 0; k < 8; ++k)
                dst.at(i - radius, j - radius + k) = code[k];
        }

        // tail loop
        for (int j = ((src.cols - radius - 1) & ~7); j < src.cols - radius; ++j) {
            float center = static_cast<float>(src.at(i, j));
            int code = 0;
            for (int n = 0; n < neighbors; ++n) {
                float t =
                    w1[n] * src.at(i + fy[n], j + fx[n]) +
                    w2[n] * src.at(i + fy[n], j + cx[n]) +
                    w3[n] * src.at(i + cy[n], j + fx[n]) +
                    w4[n] * src.at(i + cy[n], j + cx[n]);
                if (t > center || std::abs(t - center) < epsilon)
                    code |= (1 << n);
            }
            dst.at(i - radius, j - radius) = code;
        }
    }
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
template <typename _Tp>
void elbp_neon(const Image<_Tp>& src, Image<int>& dst, int, int) {
    static_assert(std::is_same<_Tp, uint8_t>::value, "Only uint8_t supported in neon version");

    const int radius = 1;
    const int neighbors = 8;
    dst = Image<int>(src.rows - 2 * radius, src.cols - 2 * radius);

    // Precompute offsets and weights
    int fx[8], fy[8], cx[8], cy[8];
    float w1[8], w2[8], w3[8], w4[8];
    for (int n = 0; n < neighbors; ++n) {
        float x = radius * std::cos(2.0f * PI * n / neighbors);
        float y = -radius * std::sin(2.0f * PI * n / neighbors);
        fx[n] = static_cast<int>(std::floor(x));
        fy[n] = static_cast<int>(std::floor(y));
        cx[n] = static_cast<int>(std::ceil(x));
        cy[n] = static_cast<int>(std::ceil(y));
        float tx = x - fx[n];
        float ty = y - fy[n];
        w1[n] = (1 - tx) * (1 - ty);
        w2[n] = tx * (1 - ty);
        w3[n] = (1 - tx) * ty;
        w4[n] = tx * ty;
    }

    const float epsilon = std::numeric_limits<float>::epsilon();
    float32x4_t v_epsilon = vdupq_n_f32(epsilon);

    #pragma omp parallel for
    for (int i = radius; i < src.rows - radius; ++i) {
        for (int j = radius; j <= src.cols - radius - 8; j += 8) {
            uint8x8_t center_u8 = vld1_u8(&src.at(i, j));
            uint16x8_t center_u16 = vmovl_u8(center_u8);
            float32x4_t center_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(center_u16)));
            float32x4_t center_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(center_u16)));

            int code[8] = {0};

            for (int n = 0; n < neighbors; ++n) {
                float32x4_t vw1 = vdupq_n_f32(w1[n]);
                float32x4_t vw2 = vdupq_n_f32(w2[n]);
                float32x4_t vw3 = vdupq_n_f32(w3[n]);
                float32x4_t vw4 = vdupq_n_f32(w4[n]);

                // Low part (0–3)
                uint8x8_t p1 = vld1_u8(&src.at(i + fy[n], j + fx[n]));
                uint8x8_t p2 = vld1_u8(&src.at(i + fy[n], j + cx[n]));
                uint8x8_t p3 = vld1_u8(&src.at(i + cy[n], j + fx[n]));
                uint8x8_t p4 = vld1_u8(&src.at(i + cy[n], j + cx[n]));

                float32x4_t interp_low = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(p1)))), vw1);
                interp_low = vmlaq_f32(interp_low, vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(p2)))), vw2);
                interp_low = vmlaq_f32(interp_low, vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(p3)))), vw3);
                interp_low = vmlaq_f32(interp_low, vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(p4)))), vw4);

                float32x4_t diff_low = vsubq_f32(interp_low, center_low);
                uint32x4_t mask_low = vorrq_u32(
                    vcgtq_f32(interp_low, center_low),
                    vcltq_f32(vabsq_f32(diff_low), v_epsilon)
                );
                for (int k = 0; k < 4; ++k)
                    if (vgetq_lane_u32(mask_low, k)) code[k] |= (1 << n);

                // High part (4–7)
                float32x4_t interp_high = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(p1)))), vw1);
                interp_high = vmlaq_f32(interp_high, vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(p2)))), vw2);
                interp_high = vmlaq_f32(interp_high, vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(p3)))), vw3);
                interp_high = vmlaq_f32(interp_high, vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(p4)))), vw4);

                float32x4_t diff_high = vsubq_f32(interp_high, center_high);
                uint32x4_t mask_high = vorrq_u32(
                    vcgtq_f32(interp_high, center_high),
                    vcltq_f32(vabsq_f32(diff_high), v_epsilon)
                );
                for (int k = 0; k < 4; ++k)
                    if (vgetq_lane_u32(mask_high, k)) code[4 + k] |= (1 << n);
            }

            for (int k = 0; k < 8; ++k)
                dst.at(i - radius, j - radius + k) = code[k];
        }

        // tail loop
        for (int j = ((src.cols - radius - 1) & ~7); j < src.cols - radius; ++j) {
            float center = static_cast<float>(src.at(i, j));
            int code = 0;
            for (int n = 0; n < neighbors; ++n) {
                float t =
                    w1[n] * src.at(i + fy[n], j + fx[n]) +
                    w2[n] * src.at(i + fy[n], j + cx[n]) +
                    w3[n] * src.at(i + cy[n], j + fx[n]) +
                    w4[n] * src.at(i + cy[n], j + cx[n]);
                if (t > center || std::abs(t - center) < epsilon)
                    code |= (1 << n);
            }
            dst.at(i - radius, j - radius) = code;
        }
    }
}
#endif
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
        
        // iterate through data
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
#if defined(__AVX2__)
    elbp_avx2<T>(src, dst, radius, neighbors);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    elbp_neon<T>(src, dst, radius, neighbors);
#else
    elbp_<T>(src, dst, radius, neighbors);
#endif
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
    return histc_(src, minVal, maxVal, normed);
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
            Image<float> cell_hist = histc(cell, 0, numPatterns - 1, true);

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
Image<int> elbp(const Image<T>& src, int radius, int neighbors, bool useSIMD) {
    // Create output Image<int>
    Image<int> dst(src.rows - 2 * radius, src.cols - 2 * radius);
    // Clear
    std::fill(dst.data.begin(), dst.data.end(), 0); 

    // Call the actual elbp algorithm
    if (radius == 1 && neighbors == 8 && std::is_same<T, uint8_t>::value && useSIMD) {
        elbp<T>(src, dst, radius, neighbors);
    } else {
        elbp_<T>(src, dst, radius, neighbors);
    }

    return dst;
}

#if defined(__AVX2__)
double compareHist_avx2(const Image<float>& H1, const Image<float>& H2) {
    if (H1.rows != 1 || H2.rows != 1 || H1.cols != H2.cols)
        throw std::invalid_argument("Histograms must be 1-row float Mats of same size.");

    const int len = H1.cols;
    const float* a = H1.data.data();
    const float* b = H2.data.data();

    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= len - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
         // div by 0 might happen for a=0
        __m256 div = _mm256_div_ps(sq, va);

        // set div to 0 where a == 0 to avoid NaN
        __m256 mask = _mm256_cmp_ps(va, _mm256_setzero_ps(), _CMP_EQ_OQ);
        div = _mm256_blendv_ps(div, _mm256_setzero_ps(), mask);
        sum = _mm256_add_ps(sum, div);
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    double dist = result[0] + result[1] + result[2] + result[3] +
                  result[4] + result[5] + result[6] + result[7];

    // tail
    for (; i < len; i++) {
        if (a[i] != 0.f) {
            float d = a[i] - b[i];
            dist += (d * d) / a[i];
        }
    }

    return dist;
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
double compareHist_neon(const Image<float>& H1, const Image<float>& H2) {
    if (H1.rows != 1 || H2.rows != 1 || H1.cols != H2.cols)
        throw std::invalid_argument("Histograms must be 1-row float Mats of same size.");

    const int len = H1.cols;
    const float* a = H1.data.data();
    const float* b = H2.data.data();

    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= len - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        float32x4_t sq = vmulq_f32(diff, diff);

        // mask: a == 0
        uint32x4_t mask = vceqq_f32(va, vdupq_n_f32(0.0f));
        float32x4_t div = vdivq_f32(sq, va);

        // apply mask: if a == 0, set result to 0
        div = vbslq_f32(mask, vdupq_n_f32(0.0f), div);

        vsum = vaddq_f32(vsum, div);
    }

    // accumulate sum from 4 float32 to 1 float
    float sum_array[4];
    vst1q_f32(sum_array, vsum);
    double dist = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // tail
    for (; i < len; i++) {
        if (a[i] != 0.f) {
            float d = a[i] - b[i];
            dist += (d * d) / a[i];
        }
    }

    return dist;
}
#endif 
double compareHist_(const Image<float>& H1, const Image<float>& H2) {
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


double compareHist_OMP(const Image<float>& H1, const Image<float>& H2, bool useSIMD) {
    if (!useSIMD) return compareHist_(H1, H2);
    
#if defined(__AVX2__)
    return compareHist_avx2(H1, H2);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return compareHist_neon(H1, H2);
#else
    return compareHist_(H1, H2);
#endif
}

void LBPH_OpenMP::train(const std::vector<Image<uint8_t>>& src, const std::vector<int>& labels, bool preserveData) {
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
    std::vector<Image<float>> local_histograms(src.size());
    // store the spatial histograms of the original data
    #pragma omp parallel for num_threads(_numThreads) reduction(+:total_elbp_time, total_hist_time)
    for (size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        // calculate lbp image
        auto t1 = Clock::now();
        Image<int> lbp = elbp(src[sampleIdx], _radius, _neighbors, _useSIMD);
        
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
        //_histograms.push_back(hist);
        local_histograms[sampleIdx] = hist;
        total_elbp_time += std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_hist_time += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }
    _histograms.insert(_histograms.end(), local_histograms.begin(), local_histograms.end());
    double avg_elbp = total_elbp_time / src.size();
    double avg_hist = total_hist_time / src.size();
    std::cout << "[train] Number of images: " << src.size() << "\n";
    std::cout << "[train] average elbp time: " << avg_elbp << " ms\n";
    std::cout << "[train] average spatial_histogram time: " << avg_hist << " ms\n";
}

struct Result {
    double dist;
    int label;
};

void LBPH_OpenMP::predict(const Image<uint8_t>& src, NearestNeighborCollector& collector) const {
    if (_histograms.empty()) {
        throw std::runtime_error("This LBPH model is not trained. Did you call the train method?");
    }

    // get the spatial histogram from input image
    Image<int> lbp_image = elbp(src, _radius, _neighbors, _useSIMD);

    Image<float> query = spatial_histogram(
        lbp_image,
        static_cast<int>(pow(2.0, static_cast<double>(_neighbors))),
        _grid_x,
        _grid_y,
        true // normalized
    );
    // find 1-nearest neighbor
    std::vector<Result> results(_histograms.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(_histograms.size()); i++) {
        double dist = compareHist_OMP(_histograms[i], query, _useSIMD);
        int label = _labels[i];
        results[i] = { dist, label };
    }

    // Single thread collect nearest neighbors
    collector.init(static_cast<int>(_histograms.size()));
    for (const auto& r : results) {
        if (!collector.collect(r.label, r.dist)) return;
    }
}

void LBPH_OpenMP::predict(const Image<uint8_t>& src, int& label, double& confidence) const {
    NearestNeighborCollector collector;
    predict(src, collector);
    label = collector.getLabel();
    confidence = collector.getDistance();
}

void LBPH_OpenMP::setParameters(std::shared_ptr<LBPHParams> params) {
    auto p = std::dynamic_pointer_cast<LBPH_OpenMP_Params>(params);
    if (p) {
        _numThreads = p->numThreads;
        _useSIMD = p->useSIMD;
    } else {
        throw std::invalid_argument("Invalid parameter type for LBPH_OpenMP");
    }
}