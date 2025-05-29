#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__device__ int mandel(float c_re, 
		      float c_im, 
		      int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {
        float z_re_t = z_re * z_re;
	float z_im_t = z_im * z_im;
        if (z_re_t + z_im_t > 4.f)
            break;

        float new_re = z_re_t - z_im_t;
        float new_im = 2.f * z_re * z_im;
	z_re = c_re + new_re;
	z_im = c_im + new_im;
    }

    return i;
}


__global__ void mandel_kernel(int *device_img, 
			      float lower_x, 
			      float lower_y,
                              float step_x, 
			      float step_y,
                              int res_x, 
			      int res_y,
                              int max_iterations)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (this_x >= res_x || this_y >= res_y) 
        return;
    
    float x = lower_x + this_x * step_x;
    float y = lower_y + this_y * step_y;

    int idx = this_y * res_x + this_x;
    device_img[idx] = mandel(x, y, max_iterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int *device_img;
    size_t size = res_x * res_y * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&device_img, size);

    // Define block and grid dimensions
    dim3 block_dim(8, 8);
    dim3 grid_dim((res_x + block_dim.x - 1) / block_dim.x, (res_y + block_dim.y - 1) / block_dim.y);

    // Launch kernel
    mandel_kernel<<<grid_dim, block_dim>>>(device_img, lower_x, lower_y, step_x, step_y, res_x, res_y, max_iterations);

    // Ensure kernel completion
    cudaDeviceSynchronize();

    // Copy result from device to image buffer
    cudaMemcpy(img, device_img, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(device_img);
}
