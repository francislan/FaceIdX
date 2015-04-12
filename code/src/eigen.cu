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


struct Image * compute_average_gpu(struct Dataset * dataset) {
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


    struct Dataset *d_dataset;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_dataset, sizeof(struct Dataset))
    );
    GPU_CHECKERROR(
    cudaMemcpy((void*) d_dataset,
               (void*) dataset,
               sizeof(struct Dataset),
               cudaMemcpyHostToDevice)
    );

    struct Image *h_average = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(h_average);

    struct Image *d_average;
    GPU_CHECKERROR(
    cudaMalloc((void**)&d_average, sizeof(struct Image))
    );
		
    dim3 dimOfGrid(ceil(dataset->w * 1.0 / 32), ceil(dataset->h * 1.0 / 32), 1);
    dim3 dimOfBlock(32, 32, 1);
    compute_average_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(dataset->original_images, w, h, 1, n, d_average);

    GPU_CHECKERROR(
    cudaMemcpy((void*) h_average,
               (void*) d_average,
               sizeof(struct Image),
               cudaMemcpyDeviceToHost)
    );

    cudaDeviceSynchronize();

    GPU_CHECKERROR(
    cudaFree(d_average)
    );

    GPU_CHECKERROR(
    cudaFree(d_dataset)
    );
		
		return h_average;
}


__global__
void compute_average_gpu_kernel(unsigned char **images, int w, int h, int comp, int num_image, struct Image *average){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if( x>=w || y>=h )  return;

    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += GET_PIXEL(images[i], x, y, 0);
    average->data[y * w + x + 0] = (sum / n);
    return;
}


