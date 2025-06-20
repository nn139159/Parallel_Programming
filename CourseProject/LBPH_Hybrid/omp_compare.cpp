#include "Image.hpp"
#include "LBPH_Serial.h"
#include "LBPH_OpenMP.h"
#include "Data_Processing.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdlib>

int main(int argc, char** argv) {
    using namespace std::chrono;

    // ---------------------- [0] Parse Args ----------------------
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

    // ---------------------- [1] Load Dataset ----------------------
    auto t0 = high_resolution_clock::now();
    std::vector<Image<uint8_t>> color_faces, sampled_faces;
    std::vector<int> labels, sampled_labels;
    if (database == 1) {
        loadSampledExtendedYaleFaces_stb(datasetPath, color_faces, labels, 3, max_samples);
    } else {
        loadATNTDataset_stb(datasetPath, color_faces, labels, 3);
    }
    auto t1 = high_resolution_clock::now();

    if (color_faces.empty()) {
        std::cerr << "No images loaded! Please check the dataset path." << std::endl;
        return -1;
    }

    // sample_per_class(color_faces, labels, max_samples, sampled_faces, sampled_labels);

    // Replace the original color_faces/labels
    // color_faces = std::move(sampled_faces);
    // labels = std::move(sampled_labels);

    // ---------------------- [2] Train/Test Split ----------------------
    std::vector<Image<uint8_t>> train_color, test_color;
    std::vector<int> train_labels, test_labels;
    train_test_split(color_faces, labels, train_color, test_color, train_labels, test_labels, 1.0 - train_ratio);


    // ---------------------- [3] RGB to Grayscale ----------------------
    std::vector<Image<uint8_t>> train_gray, test_gray;
    for (const auto& img : train_color) train_gray.push_back(rgb_to_grayscale(img));
    for (const auto& img : test_color) test_gray.push_back(rgb_to_grayscale(img));

    std::ostringstream output;

    // ---------------------- [4] Serial Baseline ----------------------
    LBPH_Serial serial_model(1, 8, 8, 8, DBL_MAX);
    auto t_train_start = high_resolution_clock::now();
    serial_model.train(train_gray, train_labels);
    auto t_train_end = high_resolution_clock::now();

    int correct_serial = 0;
    auto t_pred_start = high_resolution_clock::now();
    for (size_t i = 0; i < test_gray.size(); ++i) {
        int predicted = -1;
        double conf = 0;
        serial_model.predict(test_gray[i], predicted, conf);
        if (predicted == test_labels[i]) correct_serial++;
    }
    auto t_pred_end = high_resolution_clock::now();

    double serial_train_time = duration_cast<milliseconds>(t_train_end - t_train_start).count();
    double serial_pred_time = duration_cast<milliseconds>(t_pred_end - t_pred_start).count();
    double serial_acc = static_cast<double>(correct_serial) / test_gray.size();

    output << "\n--- LBPH OpenMP Performance Comparison ---\n";
    output << "Total samples: " << color_faces.size()
           << ", Train: " << train_gray.size()
           << ", Test: " << test_gray.size() << "\n";

    output << std::left << std::setw(10) << "Version"
           << std::setw(8)  << "Threads"
           << std::setw(8)  << "SIMD"
           << std::right << std::setw(15) << "Train Time (ms)"
           << std::setw(15) << "Predict Time (ms)"
           << std::setw(12) << "Accuracy"
           << std::setw(15) << "Train Speedup"
           << std::setw(15) << "Pred Speedup"
           << "\n";

    output << std::left << std::setw(10) << "Serial"
           << std::setw(8)  << "-"
           << std::setw(8)  << "-" << std::fixed << std::setprecision(2)
           << std::right << std::setw(15) << serial_train_time
           << std::setw(15) << serial_pred_time
           << std::setw(12) << std::fixed << std::setprecision(2) << serial_acc * 100.0
           << std::setw(15) << "1.00"
           << std::setw(15) << "1.00"
           << "\n";

    // ---------------------- [5] OpenMP Models ----------------------
    std::vector<int> thread_options = {1, 2, 4, 8, 12};

    for (bool use_simd : {false, true}) {
        for (int threads : thread_options) {
            auto model = new LBPH_OpenMP(1, 8, 8, 8, DBL_MAX);
            auto param = std::make_shared<LBPH_OpenMP_Params>();
            param->numThreads = threads;
            param->useSIMD = use_simd;
            model->setParameters(param);

            auto t_train_start = high_resolution_clock::now();
            model->train(train_gray, train_labels);
            auto t_train_end = high_resolution_clock::now();

            int correct = 0;
            auto t_pred_start = high_resolution_clock::now();
            for (size_t i = 0; i < test_gray.size(); ++i) {
                int predicted = -1;
                double conf = 0;
                model->predict(test_gray[i], predicted, conf);
                if (predicted == test_labels[i]) correct++;
            }
            auto t_pred_end = high_resolution_clock::now();

            double train_time = duration_cast<milliseconds>(t_train_end - t_train_start).count();
            double pred_time = duration_cast<milliseconds>(t_pred_end - t_pred_start).count();
            double acc = static_cast<double>(correct) / test_gray.size();

            output << std::left << std::setw(10) << "OpenMP"
                   << std::setw(8)  << threads
                   << std::setw(8)  << (use_simd ? "Yes" : "No")
                   << std::right << std::setw(15) << train_time
                   << std::setw(15) << pred_time
                   << std::setw(12) << std::fixed << std::setprecision(2) << acc * 100.0
                   << std::setw(15) << std::fixed << std::setprecision(2) << serial_train_time / train_time
                   << std::setw(15) << std::fixed << std::setprecision(2) << serial_pred_time / pred_time
                   << "\n";

            delete model;
        }
    }

    // ---------------------- [6] Print Result ----------------------
    std::cout << output.str();
    return 0;
}
