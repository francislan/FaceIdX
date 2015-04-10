#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#include "eigen.h"
#include "database.h"
#include "misc.h"

#define THREADS_PER_BLOCK 256

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
                sum += GET_PIXEL(dataset->original_images[i], x, y, 0);
            average->data[y * w + x + 0] = (sum / n);
        }
    }
    dataset->average = average;
    return average;
}



__global__
void compute_average_gpu(struct Dataset * dataset, struct Image * average) {
    int w = dataset->w;
    int h = dataset->h;
    int n = dataset->num_original_images;

    if (w <= 0 || h <= 0) {
        //PRINT("WARN", "Dataset's width and/or height incorrect(s)\n");
        printf("[Warning]: Dataset's width and/or height incorrect(s)\n");
        return;
    }
    if (n <= 0) {
        //PRINT("WARN", "No image in dataset\n");
        printf("[Warning]: No image in dataset\n");
        return;
    }

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if( x>=w || y>=h )  return;

    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += GET_PIXEL(dataset->original_images[i], x, y, 0);
    average->data[y * w + x + 0] = (sum / n);
    return;
}
