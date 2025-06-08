#include "Image.hpp"
#include "LightLBPH.h"
#include "Data_Processing.h"
#include <chrono>

#include <array>

// output video寫字需要
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

// kwan:
// 寫字用
void draw_text_stb(Image<uint8_t>& img, const std::string& text,
                   int row, int col, const std::string& font_path,
                   float font_size, std::array<uint8_t, 3> color)
{
    // 1. 載入 TTF 檔案
    FILE* fp = fopen(font_path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Font file not found!\n");
        return;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<unsigned char> font_buffer(size);
    fread(font_buffer.data(), 1, size, fp);
    fclose(fp);

    // 2. 初始化 font
    stbtt_fontinfo font;
    if (!stbtt_InitFont(&font, font_buffer.data(), 0)) {
        fprintf(stderr, "Failed to init font\n");
        return;
    }

    // 3. 計算 scale
    float scale = stbtt_ScaleForPixelHeight(&font, font_size);
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(&font, &ascent, &descent, &lineGap);
    int baseline = (int)(ascent * scale);

    int x = col;
    int y = row + baseline;

    // 4. 每個字元 raster 到影像
    for (char c : text) {
        int ax;
        int lsb;
        stbtt_GetCodepointHMetrics(&font, c, &ax, &lsb);

        int cx0, cy0, cx1, cy1;
        stbtt_GetCodepointBitmapBox(&font, c, scale, scale, &cx0, &cy0, &cx1, &cy1);

        int w = cx1 - cx0;
        int h = cy1 - cy0;

        std::vector<unsigned char> bitmap(w * h);
        stbtt_MakeCodepointBitmap(&font, bitmap.data(), w, h, w, scale, scale, c);

        // 將 bitmap 貼到 image 上
        for (int i = 0; i < h; ++i) {
            int py = y + cy0 + i;
            if (py < 0 || py >= img.rows) continue;

            for (int j = 0; j < w; ++j) {
                int px = x + cx0 + j;
                if (px < 0 || px >= img.cols) continue;

                // uint8_t val = bitmap[i * w + j];
                // for (int ch = 0; ch < 3; ++ch) {
                //     img.at(py, px, ch) = 255 - val; // 白底黑字
                // }

                uint8_t val = bitmap[i * w + j];
                if (val > 0) {
                    img.at(py, px, 0) = color[0] * val / 255;
                    img.at(py, px, 1) = color[1] * val / 255;
                    img.at(py, px, 2) = color[2] * val / 255;
                }
            }
        }

        x += (int)(ax * scale);
    }
}


// kwan:
// 開始讀取影像+辨識+輸出
void process_video_stream(LBPH& model, int width, int height) {
    size_t frame_size = width * height * 3;  // RGB24
    std::vector<uint8_t> buffer(frame_size);
    std::vector<uint8_t> output_buffer(frame_size);

    auto t_start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (std::fread(buffer.data(), 1, frame_size, stdin) == frame_size) {
        frame_count++;

        // 1. 將 buffer 包成 Image<uint8_t>
        Image<uint8_t> rgb_img(height, width, 3, buffer.data());

        // 2. 灰階 + 預測
        Image<uint8_t> gray_img = rgb_to_grayscale(rgb_img);
        int predicted = -1;
        double conf = 0.0;
        model.predict(gray_img, predicted, conf);

        // 3. 每秒顯示一次 fps
        auto t_now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count();

        double fps = 1000.0 * frame_count / std::max(1.0, static_cast<double>(elapsed));

        // 準備顯示的字串
        std::ostringstream oss;
        oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
        oss << " ID: S" << predicted;


        // 4. 把 fps 畫到 RGB buffer 上（左上角）
        std::string font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        FILE* fp = fopen(font_path.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "❌ Font file not found: %s\n", font_path.c_str());
        }

        draw_text_stb(rgb_img, oss.str(), 10, 10, font_path, 10.0, {255, 0, 0});


        // 5. 輸出
        std::cerr << "writing frame: " << frame_count << std::endl;
        // std::fwrite(buffer.data(), 1, frame_size, stdout);
        auto written = std::fwrite(rgb_img.data.data(), 1, frame_size, stdout);
        if (written != frame_size) {
            std::cerr << "⚠️ fwrite failed: wrote " << written << " / " << frame_size << " bytes" << std::endl;
        }
        // std::cout.flush();
    }
}


int main(int, char**) {
    using namespace std::chrono;

    std::string datasetPath = "att_faces";

    // ---------------- [1] Dataset Loading ----------------
    auto t0 = high_resolution_clock::now();

    std::vector<Image<uint8_t>> color_faces;
    std::vector<int> labels;
    loadATNTDataset_stb(datasetPath, color_faces, labels, 3);

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
    // std::cout << "\nAccuracy: " << accuracy * 100.0 << "%" << std::endl;

    // kwan:
    // 模擬從 stdin 讀 raw video，然後每張做 predict
    int width = 92, height = 112;  // 視訊大小要與 ffmpeg 輸出對齊
    process_video_stream(lbph_model, width, height);

    // ---------------- [6] Report Time ----------------
    // 先註解掉避免寫入stdout
    // std::cout << "\n--- Time Analysis ---" << std::endl;
    // std::cout << "Dataset Loading:         " << duration_cast<milliseconds>(t1 - t0).count() << " ms" << std::endl;
    // std::cout << "Train/Test Split:        " << duration_cast<milliseconds>(t3 - t2).count() << " ms" << std::endl;
    // std::cout << "Grayscale Conversion:    " << duration_cast<milliseconds>(t4 - t3).count() << " ms" << std::endl;
    // std::cout << "Model Training:          " << duration_cast<milliseconds>(t5 - t4).count() << " ms" << std::endl;
    // std::cout << "Prediction Phase:        " << duration_cast<milliseconds>(t7 - t6).count() << " ms" << std::endl;
    // std::cout << "Each picture Pred:       " << duration_cast<milliseconds>(t7 - t6).count() / test_color.size() << " ms" << std::endl;
    return 0;
}
