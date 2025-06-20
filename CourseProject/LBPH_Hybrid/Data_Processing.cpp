#include "Data_Processing.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

Image<uint8_t> rgb_to_grayscale(const Image<uint8_t>& color_img) {
    if (color_img.channels != 3) {
        throw std::invalid_argument("Input image is not 3-channel RGB.");
    }

    Image<uint8_t> gray_img(color_img.rows, color_img.cols, 1);

    for (int i = 0; i < color_img.rows; ++i) {
        for (int j = 0; j < color_img.cols; ++j) {
            int base_idx = (i * color_img.cols + j) * 3;
            uint8_t r = color_img.data[base_idx];
            uint8_t g = color_img.data[base_idx + 1];
            uint8_t b = color_img.data[base_idx + 2];

            // Luminosity method
            uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
            gray_img.at(i, j) = gray;
        }
    }

    return gray_img;
}

void loadATNTDataset_stb(const std::string& datasetPath,
    std::vector<Image<uint8_t>>& images,
    std::vector<int>& labels,
    int desired_channels) {
    
    for (const auto& person_dir : fs::directory_iterator(datasetPath)) {
        if (!person_dir.is_directory()) continue;

        std::string person_name = person_dir.path().filename().string();
        if (person_name.empty() || person_name[0] != 's') continue;

        int label = std::stoi(person_name.substr(1));

        for (const auto& img_path : fs::directory_iterator(person_dir.path())) {
            if (!img_path.is_regular_file()) continue;

            int width, height, channels;
            unsigned char* data = stbi_load(img_path.path().string().c_str(), &width, &height, &channels, desired_channels);
            if (!data) continue;

            int actual_channels = desired_channels > 0 ? desired_channels : channels;
            Image<uint8_t> img(height, width, actual_channels);
            std::memcpy(img.data.data(), data, width * height * actual_channels);
            images.push_back(std::move(img));
            labels.push_back(label);

            stbi_image_free(data);
        }
    }
}


void loadYaleDataset_stb(const std::string& datasetPath,
                         std::vector<Image<uint8_t>>& images,
                         std::vector<int>& labels,
                         int desired_channels) {
                            
    for (const auto& file : fs::directory_iterator(datasetPath)) {
        if (!file.is_regular_file()) continue;

        std::string filename = file.path().filename().string();

        // Check if it starts with subject (e.g. subject01.normal.pgm)
        if (filename.rfind("subject", 0) != 0) continue;

        // Extract the label (e.g. "subject01.happy.pgm" -> 01 -> label 1)
        std::string label_str = filename.substr(7, 2);  // Take 2 digits starting from index 7
        int label = std::stoi(label_str);

        int width, height, channels;
        unsigned char* data = stbi_load(file.path().string().c_str(), &width, &height, &channels, desired_channels);
        if (!data) continue;

        int actual_channels = desired_channels > 0 ? desired_channels : channels;
        Image<uint8_t> img(height, width, actual_channels);
        std::memcpy(img.data.data(), data, width * height * actual_channels);

        images.push_back(std::move(img));
        labels.push_back(label);

        stbi_image_free(data);
    }
}

bool loadExtendedYaleFaces_stb(const std::string& datasetPath,
                               std::vector<Image<uint8_t>>& images,
                               std::vector<int>& labels,
                               int desired_channels) {
    std::map<std::string, int> nameToLabel;
    int currentLabel = 0;

    for (const auto& personEntry : fs::directory_iterator(datasetPath)) {
        if (!personEntry.is_directory()) continue;

        std::string personName = personEntry.path().filename().string();
        std::string personPath = personEntry.path().string();

        if (nameToLabel.find(personName) == nameToLabel.end()) {
            nameToLabel[personName] = currentLabel++;
        }

        int label = nameToLabel[personName];

        for (const auto& fileEntry : fs::directory_iterator(personPath)) {
            std::string ext = fileEntry.path().extension().string();
            if (ext != ".pgm") continue;

            int width, height, channels;
            unsigned char* data = stbi_load(fileEntry.path().string().c_str(), &width, &height, &channels, desired_channels);

            if (!data) continue;

            int actual_channels = desired_channels > 0 ? desired_channels : channels;
            Image<uint8_t> img(height, width, actual_channels);
            std::memcpy(img.data.data(), data, width * height * actual_channels);
            stbi_image_free(data);

            images.push_back(std::move(img));
            labels.push_back(label);
        }
    }

    return !images.empty();
}

void sample_per_class(
    const std::vector<Image<uint8_t>>& images,
    const std::vector<int>& labels,
    int max_total_samples,
    std::vector<Image<uint8_t>>& out_images,
    std::vector<int>& out_labels)
{
    out_images.clear();
    out_labels.clear();

    // Group information: label -> image list
    std::unordered_map<int, std::vector<Image<uint8_t>>> class_images;
    for (size_t i = 0; i < labels.size(); ++i)
        class_images[labels[i]].push_back(images[i]);

    int num_classes = class_images.size();
    if (num_classes == 0) return;

    // Average quota
    int per_class_limit = max_total_samples / num_classes;

    std::random_device rd;
    std::mt19937 rng(rd());

    for (auto& [label, imgs] : class_images) {
        std::shuffle(imgs.begin(), imgs.end(), rng);

        int use_count = std::min(static_cast<int>(imgs.size()), per_class_limit);
        for (int i = 0; i < use_count; ++i) {
            out_images.push_back(imgs[i]);
            out_labels.push_back(label);
        }
    }

    // If max_total_samples is not reached, it can be filled from the remaining samples
    std::vector<std::pair<Image<uint8_t>, int>> remaining;
    for (auto& [label, imgs] : class_images) {
        for (int i = per_class_limit; i < static_cast<int>(imgs.size()); ++i) {
            remaining.emplace_back(imgs[i], label);
        }
    }

    std::shuffle(remaining.begin(), remaining.end(), rng);
    while (out_images.size() < static_cast<size_t>(max_total_samples) && !remaining.empty()) {
        out_images.push_back(remaining.back().first);
        out_labels.push_back(remaining.back().second);
        remaining.pop_back();
    }
}

void train_test_split(
    const std::vector<Image<uint8_t>>& x, const std::vector<int>& y,
    std::vector<Image<uint8_t>>& x_train, std::vector<Image<uint8_t>>& x_test,
    std::vector<int>& y_train, std::vector<int>& y_test,
    double test_ratio)
{
    size_t total = x.size();
    std::vector<size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(indices.begin(), indices.end(), rng);
    
    size_t test_count = static_cast<size_t>(total * test_ratio);

    for (size_t i = 0; i < total; ++i) {
        size_t idx = indices[i];
        if (i < test_count) {
            x_test.push_back(x[idx]);
            y_test.push_back(y[idx]);
        }
        else {
            x_train.push_back(x[idx]);
            y_train.push_back(y[idx]);
        }
    }
}

bool loadSampledExtendedYaleFaces_stb(const std::string& datasetPath,
                                      std::vector<Image<uint8_t>>& images,
                                      std::vector<int>& labels,
                                      int desired_channels,
                                      int max_total_samples)
{
    std::map<std::string, int> nameToLabel;
    int currentLabel = 0;
    std::unordered_map<int, std::vector<fs::path>> class_image_paths;

    // Step 1: Collect all .pgm file paths grouped by label
    for (const auto& personEntry : fs::directory_iterator(datasetPath)) {
        if (!personEntry.is_directory()) continue;

        std::string personName = personEntry.path().filename().string();
        std::string personPath = personEntry.path().string();

        if (nameToLabel.find(personName) == nameToLabel.end()) {
            nameToLabel[personName] = currentLabel++;
        }
        int label = nameToLabel[personName];

        for (const auto& fileEntry : fs::directory_iterator(personPath)) {
            if (fileEntry.path().extension() != ".pgm") continue;
            class_image_paths[label].push_back(fileEntry.path());
        }
    }

    int num_classes = class_image_paths.size();
    if (num_classes == 0) return false;

    int per_class_limit = max_total_samples / num_classes;

    std::random_device rd;
    std::mt19937 rng(rd());

    images.clear();
    labels.clear();

    // Step 2: Randomly sample and load images from each class
    std::vector<std::pair<fs::path, int>> remaining;

    for (auto& [label, paths] : class_image_paths) {
        std::shuffle(paths.begin(), paths.end(), rng);

        int use_count = std::min(per_class_limit, static_cast<int>(paths.size()));
        for (int i = 0; i < use_count; ++i) {
            int width, height, channels;
            unsigned char* data = stbi_load(paths[i].string().c_str(), &width, &height, &channels, desired_channels);
            if (!data) continue;

            int actual_channels = desired_channels > 0 ? desired_channels : channels;
            Image<uint8_t> img(height, width, actual_channels);
            std::memcpy(img.data.data(), data, width * height * actual_channels);
            stbi_image_free(data);

            images.push_back(std::move(img));
            labels.push_back(label);
        }

        // Save remaining for potential fill-in
        for (int i = use_count; i < static_cast<int>(paths.size()); ++i) {
            remaining.emplace_back(paths[i], label);
        }
    }

    // Step 3: Fill remaining quota
    std::shuffle(remaining.begin(), remaining.end(), rng);
    while (images.size() < static_cast<size_t>(max_total_samples) && !remaining.empty()) {
        auto [path, label] = remaining.back();
        remaining.pop_back();

        int width, height, channels;
        unsigned char* data = stbi_load(path.string().c_str(), &width, &height, &channels, desired_channels);
        if (!data) continue;

        int actual_channels = desired_channels > 0 ? desired_channels : channels;
        Image<uint8_t> img(height, width, actual_channels);
        std::memcpy(img.data.data(), data, width * height * actual_channels);
        stbi_image_free(data);

        images.push_back(std::move(img));
        labels.push_back(label);
    }

    return !images.empty();
}

