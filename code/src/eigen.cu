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
struct Image * compute_average_cpu(struct Dataset * dataset)
{
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


struct Image * compute_average_gpu(struct Dataset * dataset)
{
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
void compute_average_gpu_kernel(unsigned char *images, int w, int h, int num_image, unsigned char *average)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= w || y >= h)
        return;
    int sum = 0;
    for (int i = 0; i < num_image; i++)
        sum += images[i * w * h + y * w + x + 0];
    average[y * w + x + 0] = (sum / num_image);
    return;
}

int dot_product_cpu(float *a, float *b, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
        sum += a[i] * b[i];

    return sum;
}

// Expect v to be initialized to 0
void jacobi_cpu(float *a, int n, float *v, float *e)
{
    int p, q, flag, t = 0;
    float temp;
    float theta, zero = 1e-4, max, pi = 3.141592654, c, s;

    for(int i = 0; i < n; i++)
        v[i * n + i] = 1;

    while(1) {
        flag = 0;
        p = 0;
        q = 1;
        max = fabs(a[0 * n + 1]);
        for(int i = 0; i < n; i++)
            for(int j = i + 1; j < n; j++) {
                temp = fabs(a[i * n + j]);
                if (temp > zero) {
                    flag = 1;
                    if (temp > max) {
                        max = temp;
                        p = i;
                        q = j;
                    }
                }
            }
        if (!flag)
            break;
        t++;
        if(a[p * n + p] == a[q * n + q]) {
            if(a[p * n + q] > 0)
                theta = pi/4;
            else
                theta = -pi/4;
        } else {
            theta = 0.5 * atan(2 * a[p * n + q] / (a[p * n + p] - a[q * n + q]));
        }
        c = cos(theta);
        s = sin(theta);

        for(int i = 0; i < n; i++) {
            temp = c * a[p * n + i] + s * a[q * n + i];
            a[q * n + i] = -s * a[p * n + i] + c * a[q * n + i];
            a[p * n + i] = temp;
        }

        for(int i = 0; i < n; i++) {
            temp = c * a[i * n  + p] + s * a[i * n + q];
            a[i * n + q] = -s * a[i * n + p] + c * a[i * n + q];
            a[i * n + p] = temp;
        }

        for(int i = 0; i < n; i++) {
            temp = c * v[i * n + p] + s * v[i * n + q];
            v[i * n + q] = -s * v[i * n + p] + c * v[i * n + q];
            v[i * n + p] = temp;
        }

    }
/*
    printf("Nb of iterations: %d\n", t);
    printf("The eigenvalues are \n");
    for(int i = 0; i < n; i++)
        printf("%8.5f ", a[i * n + i]);

    printf("\nThe corresponding eigenvectors are \n");
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++)
            printf("% 8.5f,",v[i * n + j]);
        printf("\n");
    }*/
    for (int i = 0; i < n; i++) {
        e[2 * i + 0] = a[i * n + i];
        e[2 * i + 1] = i;
    }
}

// Sorts in place the eigenvalues in descending order
int comp_eigenvalues(const void *a, const void *b)
{
    return (fabs(*(float *)a) < fabs(*(float *)b)) - (fabs(*(float *)a) > fabs(*(float *)b));
}

int compute_eigenfaces_cpu(struct Dataset * dataset, int num_to_keep)
{
    int n = dataset->num_original_images;
    int w = dataset->w;
    int h = dataset->h;

    float **images_minus_average = (float **)malloc(n * sizeof(float *));
    TEST_MALLOC(images_minus_average);

    for (int i = 0; i < n; i++) {
        images_minus_average[i] = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(images_minus_average[i]);
    }

    // Test minus average
/*
    struct Image *minus_average = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(minus_average);
    minus_average->w = w;
    minus_average->h = h;
    minus_average->comp = 1;
    minus_average->req_comp = 1;
*/
    // Substract average to images
    struct Image *average = dataset->average;
    for (int i = 0; i < n; i++) {
        struct Image *current_image = dataset->original_images[i];
        // Maybe switching the 2 loops results in faster computation
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                images_minus_average[i][y * w + x] = (float)GET_PIXEL(current_image, x, y, 0) - GET_PIXEL(average, x, y, 0);
/*        sprintf(minus_average->filename, "minus/Minus Average %d.png", i);
        minus_average->data = (unsigned char *)malloc(w * h * 1 * sizeof(unsigned char));
        for (int j = 0; j < w * h; j++)
            minus_average->data[j] = images_minus_average[i][j] > 0 ?
                (unsigned char)((images_minus_average[i][j] / 256) * 127 + 128) :
                (unsigned char)(128 - (images_minus_average[i][j] / -256) * 128);
        save_image_to_disk(minus_average, minus_average->filename);
        free(minus_average->data);
*/    }
    PRINT("DEBUG", "Substracting average to images... done\n");

    // Construct the Covariance Matrix
    float *covariance_matrix = (float *)malloc(n * n * sizeof(float));
    TEST_MALLOC(covariance_matrix);

    for (int i = 0; i < n; i++) {
        covariance_matrix[i * n + i] = dot_product_cpu(images_minus_average[i], images_minus_average[i], n) / n;
        for (int j = i + 1; j < n; j++) {
            covariance_matrix[i * n + j] = dot_product_cpu(images_minus_average[i], images_minus_average[j], n) / n;
            covariance_matrix[j * n + i] = covariance_matrix[i * n + j];
        }
    }
    PRINT("DEBUG", "Building covariance matrix... done\n");

    // Compute eigenfaces
    float *eigenfaces = (float *)calloc(n * n, sizeof(float));
    TEST_MALLOC(eigenfaces);
    // eigenvalues stores couple (ev, index), makes it easier to get the top K
    // later
    float *eigenvalues = (float *)malloc(2 * n * sizeof(float));
    TEST_MALLOC(eigenvalues);
    jacobi_cpu(covariance_matrix, n, eigenfaces, eigenvalues);
    PRINT("DEBUG", "Computing eigenfaces... done\n");

    // Keep only top num_to_keep eigenfaces.
    // Assumes num_to_keep is in the correct range.
    qsort(eigenvalues, n, 2 * sizeof(float), comp_eigenvalues);
    for (int i = 0; i < n; i++)
        PRINT("DEBUG", "Eigenvalue #%d (index %d): %f\n", i, (int)eigenvalues[2 * i + 1], eigenvalues[2 * i]);

    // Convert size n*n eigenfaces to size w*h
    dataset->num_eigenfaces = num_to_keep;
    dataset->eigenfaces = (float **)malloc(num_to_keep * sizeof(float *));
    TEST_MALLOC(dataset->eigenfaces);
    for (int i = 0; i < num_to_keep; i++) {
        dataset->eigenfaces[i] = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(dataset->eigenfaces[i]);
    }
    float sqrt_n = sqrt(n);
    for (int i = 0; i < num_to_keep; i++) {
        int index = (int)eigenvalues[2 * i + 1];
        for (int j = 0; j < w * h; j++) {
            float temp = 0;
            for (int k = 0; k < n; k++)
                temp += images_minus_average[k][j] * eigenfaces[k * n + index];
            dataset->eigenfaces[i][j] = temp / sqrt_n;
        }
    }
    PRINT("DEBUG", "Transforming eigenfaces... done\n");

    return 0;
}

void compute_weighs(struct Dataset *dataset)
{
    
}
