#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#include "eigen.h"
#include "database.h"
#include "misc.h"

#define THREADS_PER_BLOCK 256

// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__ ))
static void gpuCheckError(cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                file, line );
        exit(EXIT_FAILURE);
    }
}

// returns NULL if error, otherwise returns pointer to average
struct Image * compute_average_cpu(struct Dataset * dataset) {
    int w = dataset->w;
    int h = dataset->h;
    int n = dataset->num_original_images;

    if (w <= 0 || h <= 0) {
        PRINT("WARN", "Dataset's width and/or height incorrect(s)\n");
        return NULL;
    }
    if (n <= 0) {
        PRINT("WARN", "No image in dataset\n");
        return NULL;
    }

    struct Image *average = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(average);

    average->data = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    TEST_MALLOC(average->data);
    average->w = w;
    average->h = h;
    average->comp = 1;

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int sum = 0;
            for (int i = 0; i < n; i++)
                sum += get_pixel(dataset->original_images[i], x, y, 0);
            average->data[y * w + x + 0] = (sum / n);
        }
    }
    dataset->average = average;
    return average;
}
