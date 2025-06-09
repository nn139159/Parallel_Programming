#pragma once
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstring>
#include <map>
#include "Image.hpp"

Image<uint8_t> rgb_to_grayscale(const Image<uint8_t>& color_img);

void loadATNTDataset_stb(const std::string& datasetPath,
    std::vector<Image<uint8_t>>& images,
    std::vector<int>& labels,
    int desired_channels = 1);

void loadYaleDataset_stb(const std::string& datasetPath,
                        std::vector<Image<uint8_t>>& images,
                        std::vector<int>& labels,
                        int desired_channels = 1);

bool loadExtendedYaleFaces_stb(const std::string& datasetPath,
                               std::vector<Image<uint8_t>>& images,
                               std::vector<int>& labels,
                               int desired_channels = 1);
void train_test_split(
    const std::vector<Image<uint8_t>>& x, const std::vector<int>& y,
    std::vector<Image<uint8_t>>& x_train, std::vector<Image<uint8_t>>& x_test,
    std::vector<int>& y_train, std::vector<int>& y_test,
    double test_ratio = 0.1);

