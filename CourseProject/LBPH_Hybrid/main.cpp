#include "Image.hpp"
#include "LBPH_Serial.h"
#include "LBPH_OpenMP.h"
#include "Data_Processing.h"
#include <chrono>

#ifdef USE_CUDA
#include "LBPH_Cuda.h"
#endif

int main(int argc, char** argv) {
    using namespace std::chrono;

    int max_samples = 1000;
    double train_ratio = 0.9;
    int database = 0;
    if (argc >= 2) {
        database = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        max_samples = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        train_ratio = std::atof(argv[3]);
        if (train_ratio <= 0.0 || train_ratio >= 1.0) {
            std::cerr << "Error: train_ratio must be between 0 and 1.\n";
            return -1;
        }
    }

    std::string datasetPath;
    if (database == 1) {
        datasetPath = "../../extendedyaleb_cropped_full";
    } else {
        datasetPath = "../../att_faces";
    }

    // ---------------- [1] Dataset Loading ----------------
    auto t0 = high_resolution_clock::now();

    std::vector<Image<uint8_t>> color_faces;
    std::vector<int> labels;

    if (database == 1) {
        loadSampledExtendedYaleFaces_stb(datasetPath, color_faces, labels, 3, max_samples);
    } else {
        loadATNTDataset_stb(datasetPath, color_faces, labels, 3);
    }

    auto t1 = high_resolution_clock::now();

    if (color_faces.empty()) {
        std::cerr << "No images loaded! Please make sure the dataset path is correct." << std::endl;
        return -1;
    }

    // ---------------- [2] Train/Test Split ----------------
    std::vector<Image<uint8_t>> train_color, test_color;
    std::vector<int> train_labels, test_labels;
    train_test_split(color_faces, labels, train_color, test_color, train_labels, test_labels, 1.0 - train_ratio);
    color_faces.clear();
    labels.clear();
    // ---------------- [3] RGB to Grayscale ----------------
    std::vector<Image<uint8_t>> train_gray, test_gray;
    for (const auto& img : train_color) train_gray.push_back(rgb_to_grayscale(img));
    for (const auto& img : test_color) test_gray.push_back(rgb_to_grayscale(img));
    train_color.clear();
    test_color.clear();
    // ---------------- [4] Create LBPH Models ----------------
    std::vector<std::pair<std::string, LBPH*>> models;
    models.emplace_back("Serial", new LBPH_Serial(1, 8, 8, 8, DBL_MAX));
    models.emplace_back("OpenMP", new LBPH_OpenMP(1, 8, 8, 8, DBL_MAX));

#ifdef USE_CUDA
    models.emplace_back("CUDA", new LBPH_Cuda(1, 8, 8, 8, DBL_MAX));
#endif

    // ---------------- [5] Training & Prediction Comparison ----------------
    struct Result {
        std::string name;
        double train_time_ms;
        double pred_time_ms;
        double accuracy;
    };

    std::vector<Result> results;

    for (auto& [name, model] : models) {
        auto t_train_start = high_resolution_clock::now();
        model->train(train_gray, train_labels);
        auto t_train_end = high_resolution_clock::now();

        int correct = 0;
        auto t_pred_start = high_resolution_clock::now();
        for (size_t i = 0; i < test_gray.size(); ++i) {
            int predicted_label = -1;
            double confidence = 0.0;
            model->predict(test_gray[i], predicted_label, confidence);
            if (predicted_label == test_labels[i]) correct++;
        }
        auto t_pred_end = high_resolution_clock::now();

        double train_time = duration_cast<milliseconds>(t_train_end - t_train_start).count();
        double pred_time = duration_cast<milliseconds>(t_pred_end - t_pred_start).count();
        double accuracy = static_cast<double>(correct) / test_gray.size();

        results.push_back({name, train_time, pred_time, accuracy});
    }

    // ---------------- [6] Report Results ----------------
    const auto& baseline = results[0];  // Serial is the baseline
    std::cout << "\n--- LBPH Performance Comparison ---\n";
    std::cout << std::left << std::setw(10) << "Version"
              << std::right << std::setw(15) << "Train Time (ms)"
              << std::setw(15) << "Predict Time (ms)"
              << std::setw(15) << "Accuracy"
              << std::setw(15) << "Train Speedup"
              << std::setw(15) << "Pred Speedup" << "\n";

    for (const auto& r : results) {
        double train_speedup = baseline.train_time_ms / r.train_time_ms;
        double pred_speedup = baseline.pred_time_ms / r.pred_time_ms;

        std::cout << std::left << std::setw(10) << r.name
                  << std::right << std::setw(15) << r.train_time_ms
                  << std::setw(15) << r.pred_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.accuracy * 100.0
                  << std::setw(15) << std::fixed << std::setprecision(2) << train_speedup
                  << std::setw(15) << std::fixed << std::setprecision(2) << pred_speedup
                  << "\n";
    }

    // Cleanup
    for (auto& [_, model] : models) {
        delete model;
    }

    return 0;
}
