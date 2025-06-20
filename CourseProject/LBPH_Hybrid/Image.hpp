#pragma once
#include <vector>
#include <string>

template <typename T>
class Image {
public:
    int rows, cols, channels;
    std::vector<T> data;

    Image();
    Image(int _rows, int _cols, int _channels = 1);
    Image(int _rows, int _cols, int _channels, const T* src_data);

    inline T& at(int i, int j, int c = 0);
    inline const T& at(int i, int j, int c = 0) const;

    int total() const;
    bool empty() const;

    Image<T> getROI(int row_start, int row_end, int col_start, int col_end) const;
};

// ---------------- Implementation ----------------

template <typename T>
Image<T>::Image() : rows(0), cols(0), channels(1) {}

template <typename T>
Image<T>::Image(int _rows, int _cols, int _channels)
    : rows(_rows), cols(_cols), channels(_channels), data(_rows * _cols * _channels, T(0)) {}

template <typename T>
Image<T>::Image(int _rows, int _cols, int _channels, const T* src_data)
    : rows(_rows), cols(_cols), channels(_channels), data(src_data, src_data + _rows * _cols * _channels) {}

template <typename T>
inline T& Image<T>::at(int i, int j, int c) {
    return data[(i * cols + j) * channels + c];
}

template <typename T>
inline const T& Image<T>::at(int i, int j, int c) const {
    return data[(i * cols + j) * channels + c];
}

template <typename T>
int Image<T>::total() const {
    return rows * cols * channels;
}

template <typename T>
bool Image<T>::empty() const {
    return data.empty();
}

template <typename T>
Image<T> Image<T>::getROI(int row_start, int row_end, int col_start, int col_end) const {
    int h = row_end - row_start;
    int w = col_end - col_start;
    Image<T> roi(h, w, channels);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int c = 0; c < channels; ++c) {
                roi.at(i, j, c) = at(row_start + i, col_start + j, c);
            }
        }
    }

    return roi;
}
