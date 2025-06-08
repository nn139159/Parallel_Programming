__kernel void convolution(const int filter_width,
                          __constant float *filter,
                          const int image_height,
                          const int image_width,
                          __global const float *input_image,
                          __global float *output_image)
{
    // column(width)
    int j = get_global_id(0); 
    // row(height)
    int i = get_global_id(1); 
    // The offset of the filter center from the edge
    int halffilter_size = filter_width / 2;
    int k, l;
    float sum = 0.0f;

    int is_inner = (i >= halffilter_size && i < image_height - halffilter_size &&
                    j >= halffilter_size && j < image_width - halffilter_size);

    if (is_inner) {
        for (k = -halffilter_size; k <= halffilter_size; k++) 
        {
            for (l = -halffilter_size; l <= halffilter_size; l++) 
            {
                int y = i + k;
                int x = j + l;

                sum += input_image[(y * image_width) + x]
                       * filter[((k + halffilter_size) * filter_width) + l 
                                + halffilter_size];
            }
        }
    } else {
        for (k = -halffilter_size; k <= halffilter_size; k++)
        {
            for (l = -halffilter_size; l <= halffilter_size; l++)
            {
	        int y = i + k;
	        int x = j + l;
            
	        // Skip pixels beyond the border ( zero-padding )
                if (y >= 0 && y < image_height && x >= 0 && x < image_width)
                {
                    sum += input_image[(y * image_width) + x]
                           * filter[((k + halffilter_size) * filter_width) + l
                                    + halffilter_size];
                }
            }
        }
    }
    // Write the results to the corresponding positions in the one-dimensional output array.
    output_image[(i * image_width) + j] = sum;
}
