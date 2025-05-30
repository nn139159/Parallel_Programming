#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

void write_ppm_image(int *data, int width, int height, const char *filename, int max_iterations)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        perror("fopen() failed");
        exit(EXIT_FAILURE);
    }

    // write ppm header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");

    for (int i = 0; i < width * height; ++i)
    {

        // Clamp iteration count for this pixel, then scale the value
        // to 0-1 range.  Raise resulting value to a power (<1) to
        // increase brightness of low iteration count
        // pixels. a.k.a. Make things look cooler.

        float mapped = std::pow(
            std::min(static_cast<float>(max_iterations), static_cast<float>(data[i])) / 256.f, .5f);

        // convert back into 0-255 range, 8-bit channels
        unsigned char result = static_cast<unsigned char>(255.f * mapped);
        for (int j = 0; j < 3; ++j)
            fputc(result, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", filename);
}
