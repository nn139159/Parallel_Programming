#include <cstdio>
#include <cstdlib>
#include <cuda.h>

// Each thread computes a GROUP_SIZE x GROUP_SIZE block
#define GROUP_SIZE 4 

__device__ int mandel(float c_re, 
		      float c_im, 
		      int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandel_kernel(int *device_img, 
			      size_t pitch,
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

    // Determine the starting pixel for this threads block
    int this_x = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int this_y = (blockIdx.y * blockDim.y + threadIdx.y) * GROUP_SIZE;

    // Process the block
    for (int dy = 0; dy < GROUP_SIZE; dy++) {
        int y_index = this_y + dy;

        if (y_index >= res_y) 
	    break;

        int *row = (int*)((char*)device_img + y_index * pitch);

        for (int dx = 0; dx < GROUP_SIZE; dx++) {
            int x_index = this_x + dx;

            if (x_index >= res_x) 
	        break;

            float x = lower_x + x_index * step_x;
            float y = lower_y + y_index * step_y;
            row[x_index] = mandel(x, y, max_iterations);
        }
    }
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
    int *host_buffer;
    size_t pitch;

    // Allocate pinned host memory
    cudaHostAlloc((void**)&host_buffer, res_x * res_y * sizeof(int), cudaHostAllocDefault);

    // Allocate pitched device memory
    cudaMallocPitch((void**)&device_img, &pitch, res_x * sizeof(int), res_y);

    // Compute grid dimensions based on groups
    int groups_x = (res_x + GROUP_SIZE - 1) / GROUP_SIZE;
    int groups_y = (res_y + GROUP_SIZE - 1) / GROUP_SIZE;

    // Set up execution configuration
    dim3 block_dim(16, 16);
    dim3 grid_dim((groups_x + block_dim.x - 1) / block_dim.x, (groups_y + block_dim.y - 1) / block_dim.y);

    // Launch kernel
    mandel_kernel<<<grid_dim, block_dim>>>(device_img, pitch, lower_x, lower_y, step_x, step_y, res_x, res_y, max_iterations);

    // Ensure kernel completion
    cudaDeviceSynchronize();

    // Copy data from pitched device memory to pinned host memory
    cudaMemcpy2D(host_buffer, (res_x * sizeof(int)), device_img, pitch, (res_x * sizeof(int)), res_y, cudaMemcpyDeviceToHost);

    // Copy result into the provided image buffer
    for (int i = 0; i < res_x * res_y; ++i) {
        img[i] = host_buffer[i];
    }

    // Clean up
    cudaFree(device_img);
    cudaFreeHost(host_buffer);
}
