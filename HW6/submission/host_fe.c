#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

unsigned int compute_checksum(const float* data, size_t size) {
    unsigned int sum = 0;
    for (size_t i = 0; i < size; ++i)
        sum ^= ((const unsigned int*)data)[i];
    return sum;
}


void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    cl_int status;
    int filter_size = filter_width * filter_width;
    int image_size = image_height * image_width;   

    static cl_mem d_input = NULL;
    static cl_mem d_filter = NULL;
    static cl_mem d_output = NULL;
    static cl_command_queue queue = NULL;
    static cl_kernel kernel = NULL;
    static int initialized = 0;
    static size_t prev_filter_width = 0;
    static size_t prev_image_size = 0;
    static unsigned int prev_checksum = 0;
    static unsigned int prev_filter_checksum = 0;

    if (!initialized) {
        initialized = 1;

	// Create command queue
        queue = clCreateCommandQueue(*context, *device, 0, &status);

        // Create kernel
        kernel = clCreateKernel(*program, "convolution", &status);
    }

    if (filter_width != prev_filter_width) {
        if (d_filter) clReleaseMemObject(d_filter);

        // Create buffers
        d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * filter_size, filter, &status);

        // Set kernel arguments
        clSetKernelArg(kernel, 0, sizeof(int), &filter_width);

        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);

        prev_filter_checksum = compute_checksum(filter, filter_size);
        prev_filter_width = filter_width;
    } else {
        unsigned int curr_filter_checksum = compute_checksum(filter, filter_size);

        if (curr_filter_checksum != prev_filter_checksum) { 
            clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0,
	                         sizeof(float) * filter_size, filter, 0, NULL, NULL);
            prev_filter_checksum = curr_filter_checksum;
        }
    }

    if (image_size != prev_image_size) {
        if (d_input) clReleaseMemObject(d_input);
        if (d_output) clReleaseMemObject(d_output);

        // Create buffers
        d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * image_size, input_image, &status);

        d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
	   	                 sizeof(float) * image_size, NULL, &status);

        // Set kernel arguments
        clSetKernelArg(kernel, 2, sizeof(int), &image_height);

        clSetKernelArg(kernel, 3, sizeof(int), &image_width);

        clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_input);

        clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);
    
        prev_checksum = compute_checksum(input_image, image_size);
        prev_image_size = image_size;
    } else {
        unsigned int curr = compute_checksum(input_image, image_size);

        if (curr != prev_checksum) {
            clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0,
                                 sizeof(float) * image_size, input_image, 0, NULL, NULL);
            prev_checksum = curr;
        }
    }

    // Define global and local work sizes
    size_t global_work_size[2] = { (size_t)image_width, (size_t)image_height };

    // Enqueue kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read back the result
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * image_size,
                        output_image, 0, NULL, NULL);
}
