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
    printf("entering compute_average_gpu()...\n");
    if (w <= 0 || h <= 0) {
        PRINT("WARN", "Dataset's width and/or height incorrect(s)\n");
        return NULL;
    }
    if (n <= 0) {
        PRINT("WARN", "No image in dataset\n");
        return NULL;
    }
/*
    unsigned char *h_images = (unsigned char*)malloc(n*w*h*sizeof(unsigned char ));
    for(int i=0; i<n; i++){
        for(int j=0; j<w*h; j++){
            unsigned char *data = (dataset->original_images)[i]->data;
            h_images[w*h*i+j] = data[j];
        }
    }
*/
    unsigned char *d_images;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_images, n * w * h * sizeof(unsigned char))
    );
    for(int i = 0; i < n; i++){
        GPU_CHECKERROR(
        cudaMemcpy((void*)(d_images + i * w * h),
                   (void*)(dataset->original_images)[i]->data,
                   w * h * sizeof(unsigned char),
                   cudaMemcpyHostToDevice)
        );
    }

    unsigned char *h_average_image = (unsigned char*)malloc(w * h * sizeof(unsigned char));
    TEST_MALLOC(h_average_image);
    unsigned char *d_average_image;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_average_image, w * h * sizeof(unsigned char))
    );
    GPU_CHECKERROR(
    cudaMemset((void*)d_average_image, 0, w * h * sizeof(unsigned char))
    );


    dim3 dimOfGrid(ceil(w * 1.0 / 32), ceil(h * 1.0 / 32), 1);
    dim3 dimOfBlock(32, 32, 1);
    compute_average_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(d_images, w, h, n, d_average_image);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS) {
        PRINT("WARN", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        return NULL;
    }

    GPU_CHECKERROR(
    cudaMemcpy((void*)h_average_image,
               (void*)d_average_image,
               w * h * sizeof(unsigned char),
               cudaMemcpyDeviceToHost)
    );

    struct Image *h_average = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(h_average);
    h_average->data = h_average_image;
    h_average->w = w;
    h_average->h = h;
    h_average->comp = 1;

    GPU_CHECKERROR(
    cudaFree(d_average_image)
    );
    GPU_CHECKERROR(
    cudaFree(d_images)
    );
    dataset->average = h_average;
    printf("exiting compute_average_gpu()...\n");    
    return h_average;
}


__global__
void compute_average_gpu_kernel(unsigned char *images, int w, int h, int num_image, unsigned char *average){
    //printf("entering kernel...\n");
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= w || y >= h)
        return;
    //printf("000...\n");
    int sum = 0;
    for (int i = 0; i < num_image; i++)
        sum += images[i * w * h + y * w + x + 0];
    average[y * w + x + 0] = (sum / num_image);
    //printf("exiting kernel...\n");
    return;
}


