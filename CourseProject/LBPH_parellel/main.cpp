#include "Image.hpp"
#include "LightLBPH.h"
#include "Data_Processing.h"
#include <chrono>

int main(int, char**) {
    using namespace std::chrono;

    std::string datasetPath = "../../att_faces";
    // std::string datasetPath = "../../extendedyaleb_cropped_full";
    // ---------------- [1] Dataset Loading ----------------
    auto t0 = high_resolution_clock::now();

    std::vector<Image<uint8_t>> color_faces;
    std::vector<int> labels;
    loadATNTDataset_stb(datasetPath, color_faces, labels, 3);
    // loadExtendedYaleFaces_stb(datasetPath, color_faces, labels, 3);

    auto t1 = high_resolution_clock::now();
    if (color_faces.empty()) {
        std::cerr << "No images loaded! Please make sure the dataset path is correct." << std::endl;
        return -1;
    }

    // ---------------- [2] Train/Test Split ----------------
    auto t2 = high_resolution_clock::now();

    std::vector<Image<uint8_t>> train_color, test_color;
    std::vector<int> train_labels, test_labels;
    train_test_split(color_faces, labels, train_color, test_color, train_labels, test_labels, 0.01);

    auto t3 = high_resolution_clock::now();

    // ---------------- [3] RGB to Grayscale (Train) ----------------
    std::vector<Image<uint8_t>> train_gray;
    for (const auto& img : train_color) {
        train_gray.push_back(rgb_to_grayscale(img));
    }

    auto t4 = high_resolution_clock::now();

    // ---------------- [4] Model Training ----------------
    LBPH lbph_model(1, 8, 8, 8, DBL_MAX);
    lbph_model.train(train_gray, train_labels);

    auto t5 = high_resolution_clock::now();

    // ---------------- [5] Model Prediction ----------------
    int correct = 0;
    auto t6 = high_resolution_clock::now();
    for (size_t i = 0; i < test_color.size(); ++i) {
        Image<uint8_t> test_gray = rgb_to_grayscale(test_color[i]);
        int predicted_label = -1;
        double confidence = 0.0;
        lbph_model.predict(test_gray, predicted_label, confidence);
        // printf("true:%d, pred:%d, conf:%f\n", test_labels[i], predicted_label, confidence);

        if (predicted_label == test_labels[i]) {
            correct++;
        }
    }
    auto t7 = high_resolution_clock::now();

    double accuracy = static_cast<double>(correct) / test_color.size();
    std::cout << "\nAccuracy: " << accuracy * 100.0 << "%" << std::endl;

    // ---------------- [6] Report Time ----------------
    std::cout << "\n--- Time Analysis ---" << std::endl;
    std::cout << "Dataset Loading:         " << duration_cast<milliseconds>(t1 - t0).count() << " ms" << std::endl;
    std::cout << "Train/Test Split:        " << duration_cast<milliseconds>(t3 - t2).count() << " ms" << std::endl;
    std::cout << "Grayscale Conversion:    " << duration_cast<milliseconds>(t4 - t3).count() << " ms" << std::endl;
    std::cout << "Model Training:          " << duration_cast<milliseconds>(t5 - t4).count() << " ms" << std::endl;
    std::cout << "Prediction Phase:        " << duration_cast<milliseconds>(t7 - t6).count() << " ms" << std::endl;
    std::cout << "Each picture Pred:       " << duration_cast<milliseconds>(t7 - t6).count() / test_color.size() << " ms" << std::endl;
    return 0;
}
