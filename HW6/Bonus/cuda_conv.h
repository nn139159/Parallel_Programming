#ifndef CUDA_CONV_H
#define CUDA_CONV_H

#ifdef __cplusplus
extern "C" {
#endif

void host_fe_cuda(int filter_width,
                  float *filter,
                  int image_height,
                  int image_width,
                  float *input_image,
                  float *output_image);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CONV_H

