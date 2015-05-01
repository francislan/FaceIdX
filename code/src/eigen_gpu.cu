#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#include "eigen_gpu.h"
#include "database_gpu.h"
#include "misc.h"
#include "load_save_image.h"

#define TILE_WIDTH 32
#define THRES_EIGEN 1.0


struct DatasetGPU * create_dataset_and_compute_all_gpu(const char *path, const char *name)
{
    struct Timer timer;
    INITIALIZE_TIMER(timer);

    START_TIMER(timer);
    printf("\nCreating database...\n");
    struct DatasetGPU *dataset = create_dataset_gpu(path, name);
    STOP_TIMER(timer);
    PRINT("DEBUG", "Time for creating database on GPU: %fms\n", timer.time);
    if (dataset == NULL) {
        PRINT("BUG","Dataset creation failed\n");
        return NULL;
    }
    printf("Creating database... Done!\n");

    printf("Computing average...\n");
    START_TIMER(timer);
    compute_average_gpu(dataset);
    STOP_TIMER(timer);
    PRINT("DEBUG", "Time for computing average on GPU: %fms\n", timer.time);
    printf("Computing average... Done!\n");

    // Eigenfaces
    printf("Computing eigenfaces...\n");
    START_TIMER(timer);
    compute_eigenfaces_gpu(dataset, dataset->num_original_images); // 2nd param can be changed
    STOP_TIMER(timer);
    PRINT("DEBUG", "Time for computing eigenfaces on GPU: %fms\n", timer.time);
    printf("Computing eigenfaces... Done!\n");

    printf("Compute images coordinates...\n");
    START_TIMER(timer);
    compute_weighs_gpu(dataset, NULL, 1, dataset->num_original_images, 1);
    STOP_TIMER(timer);
    PRINT("DEBUG", "Time for computing faces coordinates on GPU: %fms\n", timer.time);
    printf("Compute images coordinates... Done!\n");

    FREE_TIMER(timer);
    return dataset;
}

__global__
void sum_gpu_kernel(float *d_a, int size, float *d_partial_sum)
{
    extern __shared__ float s_thread_sums[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    s_thread_sums[t] = (i < size ? d_a[i] : 0);
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (t < stride)
            s_thread_sums[t] += s_thread_sums[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        volatile float *cache = s_thread_sums;
        cache[t] += cache[t + 32];
        cache[t] += cache[t + 16];
        cache[t] += cache[t + 8];
        cache[t] += cache[t + 4];
        cache[t] += cache[t + 2];
        cache[t] += cache[t + 1];
    }
    if (t == 0)
        d_partial_sum[blockIdx.x] = s_thread_sums[0];
}

void normalize_gpu(float *d_array, int size)
{
    int num_blocks = ceil(size / 1024.0);
    dim3 dimOfGrid(num_blocks, 1, 1);
    dim3 dimOfBlock(1024, 1, 1);

    // Normalize in two steps.
    // First, divide by the mean
    // Then compute the norm, otherwise there are float precision issues.
    float mean = 0;
    int size_shared_mem = dimOfBlock.x * sizeof(float);

    float *d_partial_sum;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_partial_sum, num_blocks * sizeof(float))
    );
    float *h_partial_sum = (float *)malloc(num_blocks * sizeof(float));
    TEST_MALLOC(h_partial_sum);

    sum_gpu_kernel<<<dimOfGrid, dimOfBlock, size_shared_mem>>>(d_array, size, d_partial_sum);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    GPU_CHECKERROR(
    cudaMemcpy((void*)h_partial_sum,
               (void*)d_partial_sum,
               num_blocks * sizeof(float),
               cudaMemcpyDeviceToHost)
    );
    cudaDeviceSynchronize();

    for (int i = 0; i < num_blocks; i++)
        mean += h_partial_sum[i];
    mean /= size;

    divide_by_float_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(d_array, mean, size);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    float norm = sqrt(dot_product_gpu(d_array, d_array, size, 1, NULL, NULL, 0, 0));

    divide_by_float_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(d_array, norm, size);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    free(h_partial_sum);
    GPU_CHECKERROR(cudaFree(d_partial_sum));
}

__global__
void divide_by_float_gpu_kernel(float *d_array, float constant, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        d_array[i] /= constant;
        i += gridDim.x * blockDim.x;
    }
}

void compute_average_gpu(struct DatasetGPU * dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int n = dataset->num_original_images;
    Timer timer;
    INITIALIZE_TIMER(timer);
    if (w <= 0 || h <= 0) {
        PRINT("WARN", "DatasetGPU's width and/or height incorrect(s)\n");
        return;
    }
    if (n <= 0) {
        PRINT("WARN", "No image in dataset\n");
        return;
    }

    START_TIMER(timer);
    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_average), w * h * sizeof(float))
    );
    STOP_TIMER(timer);
    PRINT("DEBUG", "Time allocating average Image on GPU: %fms\n", timer.time);

    START_TIMER(timer);
    dim3 dimOfGrid(ceil(w * h / 256.0), 1, 1);
    dim3 dimOfBlock(256, 1, 1);
    compute_average_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(dataset->d_original_images, w * h, n, dataset->d_average);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("WARN", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "Time average kernel: %fms\n", timer.time);

    FREE_TIMER(timer);
}

__global__
void compute_average_gpu_kernel(float *d_images, int size, int num_image, float *d_average)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x >= size)
        return;
    float sum = 0;
    for (int i = 0; i < num_image; i++)
        sum += d_images[i * size + x];
    d_average[x] = (sum / num_image);
}

__global__
void dot_product_gpu_kernel(float *d_a, float *d_b, int size, float *d_partial_sum)
{
    extern __shared__ float s_thread_sums[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    float temp = 0;
    while (i < size) {
        temp += d_a[i] * d_b[i];
        i += gridDim.x * blockDim.x;
    }
    s_thread_sums[t] = temp;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (t < stride)
            s_thread_sums[t] += s_thread_sums[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        volatile float *cache = s_thread_sums;
        cache[t] += cache[t + 32];
        cache[t] += cache[t + 16];
        cache[t] += cache[t + 8];
        cache[t] += cache[t + 4];
        cache[t] += cache[t + 2];
        cache[t] += cache[t + 1];
    }
    if (t == 0)
        d_partial_sum[blockIdx.x] = s_thread_sums[0];
}

// d/h_partial_sum can be provided by caller. In the case of multiple
// dot_products calls (same vector size), it can be significantly faster.
// Otherwise set to NULL
float dot_product_gpu(float *d_a, float *d_b, int size, int allocate, float *d_partial_sum_user, float *h_partial_sum_user, int num_blocks, int num_threads)
{
    if (num_blocks == 0 && allocate)
        num_blocks = ceil(size / 1024.0 / 4);
    if (num_threads == 0 && allocate)
        num_threads = 1024;
    dim3 dimOfGrid(num_blocks, 1, 1);
    dim3 dimOfBlock(num_threads, 1, 1);
    int size_shared_mem = dimOfBlock.x * sizeof(float);

    float *d_partial_sum;
    float *h_partial_sum;
    if (allocate) {
        GPU_CHECKERROR(
        cudaMalloc((void **)&d_partial_sum, num_blocks * sizeof(float))
        );
        h_partial_sum = (float *)malloc(num_blocks * sizeof(float));
        TEST_MALLOC(h_partial_sum);
    } else {
        d_partial_sum = d_partial_sum_user;
        h_partial_sum = h_partial_sum_user;
    }

    float result = 0;

    dot_product_gpu_kernel<<<dimOfGrid, dimOfBlock, size_shared_mem>>>(d_a, d_b, size, d_partial_sum);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    GPU_CHECKERROR(
    cudaMemcpy((void*)h_partial_sum,
               (void*)d_partial_sum,
               num_blocks * sizeof(float),
               cudaMemcpyDeviceToHost)
    );
    cudaDeviceSynchronize();

    for (int i = 0; i < num_blocks; i++)
        result += h_partial_sum[i];

    if (allocate) {
        GPU_CHECKERROR(cudaFree(d_partial_sum));
        free(h_partial_sum);
    }

    return result;
}

// Expect v to be initialized to 0
void jacobi_gpu(float *a, const int n, float *v, float *e)
{
    int p;
    int q;
    int flag; // stops when flag == 0
    float temp;
    float theta;
    float zero = 1e-5;
    float max;
    float pi = 3.141592654;
    float c; // = cos(theta)
    float s; // = sin(theta)

    for (int i = 0; i < n; i++)
        v[i * n + i] = 1;

    while (1) {
        flag = 0;
        p = 0;
        q = 1;
        max = fabs(a[0 * n + 1]);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
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
        if (a[p * n + p] == a[q * n + q]) {
            if (a[p * n + q] > 0)
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
    for (int i = 0; i < n; i++) {
        e[2 * i + 0] = a[i * n + i];
        e[2 * i + 1] = i;
    }
}

// Sorts in place the eigenvalues in descending order
int comp_eigenvalues_gpu(const void *a, const void *b)
{
    return (fabs(*(float *)a) < fabs(*(float *)b)) - (fabs(*(float *)a) > fabs(*(float *)b));
}

__global__
void substract_average_gpu_kernel(float *d_data, float *d_average, int size, int size_image)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        d_data[i] -= d_average[i % (size_image)];
        i += gridDim.x * blockDim.x;
    }
}

void substract_average_gpu(struct ImageGPU **images, float *d_average, int num_images, int size_image)
{
    float *d_data;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_data, num_images * size_image * sizeof(float))
    );

    for (int i = 0; i < num_images; i++) {
        GPU_CHECKERROR(
        cudaMemcpy((void*)(d_data + i * size_image),
                   (void*)(images[i]->data),
                   size_image * sizeof(float),
                   cudaMemcpyHostToDevice)
        );
    }

    dim3 dimOfGrid(ceil(size_image * num_images / 1024.0 / 4), 1, 1);
    dim3 dimOfBlock(1024, 1, 1);
    substract_average_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(d_data, d_average, size_image * num_images, size_image);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("WARN", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_images; i++) {
        GPU_CHECKERROR(
        cudaMemcpy((void*)(images[i]->data),
                   (void*)(d_data + i *size_image),
                   size_image * sizeof(float),
                   cudaMemcpyDeviceToHost)
        );
    }
    GPU_CHECKERROR(cudaFree(d_data));
}

// total_size = unitary_size * count
__global__
void transpose_matrix_gpu_kernel(float *d_input, float *d_output, int total_size, int unitary_size, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < total_size) {
        d_output[(i % unitary_size) * count + (i / unitary_size)] = d_input[i];
        i += gridDim.x * blockDim.x;
    }
}

// The number of orignal images is limited by the GPU's memory size
// For 2Gb of memory, about 10k of 200*250 images can be stored
// So the number of threads is not a bottleneck
int compute_eigenfaces_gpu(struct DatasetGPU * dataset, int num_to_keep)
{
    int n = dataset->num_original_images;
    int w = dataset->w;
    int h = dataset->h;
    Timer timer;
    INITIALIZE_TIMER(timer);

    // Substract average to images
    dim3 dimOfGrid(ceil(w * h * n / 1024.0 / 4), 1, 1);
    dim3 dimOfBlock(1024, 1, 1);

    START_TIMER(timer);
    substract_average_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(dataset->d_original_images, dataset->d_average, n * w * h, w * h);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("WARN", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to substract average: %fms\n", timer.time);
    PRINT("INFO", "Substracting average to images... done\n");

    // Construct the Covariance Matrix
    START_TIMER(timer);
    float *covariance_matrix = (float *)malloc(n * n * sizeof(float));
    TEST_MALLOC(covariance_matrix);

    float *d_A;
    float *d_tA;
    float *d_covariance_matrix;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_A, n * w * h * sizeof(float))
    );
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_tA, n * w * h * sizeof(float))
    );
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_covariance_matrix, n * n * sizeof(float))
    );
    GPU_CHECKERROR(
    cudaMemcpy((void*)d_tA,
               (void*)dataset->d_original_images,
               n * w * h * sizeof(float),
               cudaMemcpyDeviceToDevice)
    );
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to allocate matrix for covariance and transfer to GPU: %fms\n", timer.time);

    START_TIMER(timer);
    float sqrt_n = sqrt(n);
    divide_by_float_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(d_tA, sqrt_n, n * w * h);
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to divide tA by sqrt_n: %fms\n", timer.time);

    START_TIMER(timer);
    transpose_matrix_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(dataset->d_original_images, d_A, n * w * h, w * h, n);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to transpose matrix tA: %f\n", timer.time);

    START_TIMER(timer);
    dim3 dimOfGrid_mult(ceil(n * 1.0 / TILE_WIDTH), ceil(n * 1.0 / TILE_WIDTH), 1);
    dim3 dimOfBlock_mult(TILE_WIDTH, TILE_WIDTH, 1);
    matrix_mult_gpu_kernel<<<dimOfGrid_mult, dimOfBlock_mult>>>(d_tA, d_A, d_covariance_matrix, w * h, n, n, w * h);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("WARN", "kernel matrix_mult_gpu_kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to compute covariance matrix: %fms\n", timer.time);

    GPU_CHECKERROR(
    cudaMemcpy((void*)covariance_matrix,
               (void*)d_covariance_matrix,
               n * n * sizeof(float),
               cudaMemcpyDeviceToHost)
    );

    PRINT("INFO", "Building covariance matrix... done\n");

    // Compute eigenfaces
    float *eigenfaces = (float *)calloc(n * n, sizeof(float));
    TEST_MALLOC(eigenfaces);
    // eigenvalues stores couple (ev, index), makes it easier to get the top K
    // later
    float *eigenvalues = (float *)malloc(2 * n * sizeof(float));
    TEST_MALLOC(eigenvalues);

    START_TIMER(timer);
    jacobi_gpu(covariance_matrix, n, eigenfaces, eigenvalues);
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to do jacobi gpu: %fms\n", timer.time);
    PRINT("INFO", "Computing eigenfaces... done\n");

    // Keep only top num_to_keep eigenfaces.
    // Assumes num_to_keep is in the correct range.
    int num_eigenvalues_not_zero = 0;
    qsort(eigenvalues, n, 2 * sizeof(float), comp_eigenvalues_gpu);
    for (int i = 0; i < n; i++) {
        //PRINT("DEBUG", "Eigenvalue #%d (index %d): %f\n", i, (int)eigenvalues[2 * i + 1], eigenvalues[2 * i]);
        if (eigenvalues[2 * i] > THRES_EIGEN)
            num_eigenvalues_not_zero++;
    }
    num_to_keep = num_eigenvalues_not_zero;
    dataset->num_eigenfaces = num_to_keep;

    // Convert size n eigenfaces to size w*h
    START_TIMER(timer);
    float *h_small_eigenfaces = (float *)malloc(num_to_keep * n * sizeof(float));
    TEST_MALLOC(h_small_eigenfaces);
    for (int i = 0; i < num_to_keep; i++) {
        int index = (int)eigenvalues[2 * i + 1];
        for (int j = 0; j < n; j++)
            h_small_eigenfaces[j * num_to_keep + i] = eigenfaces[j * n + index];
    }

    float *d_small_eigenfaces;
    float *d_big_eigenfaces;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_small_eigenfaces, num_to_keep * n * sizeof(float))
    );
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_big_eigenfaces, num_to_keep * w * h * sizeof(float))
    );
    GPU_CHECKERROR(
    cudaMemcpy((void*)d_small_eigenfaces,
               (void*)h_small_eigenfaces,
               num_to_keep * n * sizeof(float),
               cudaMemcpyHostToDevice)
    );
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time for preparing matrix for transforming small_eigens to big_eigens: %fms\n", timer.time);


    START_TIMER(timer);
    dimOfGrid_mult.x = ceil(num_to_keep * 1.0 / TILE_WIDTH);
    dimOfGrid_mult.y = ceil(w * h * 1.0 / TILE_WIDTH);
    matrix_mult_gpu_kernel<<<dimOfGrid_mult, dimOfBlock_mult>>>(d_A, d_small_eigenfaces, d_big_eigenfaces, n, w * h, num_to_keep, n);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("WARN", "kernel matrix_mult_gpu_kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to transform eigenfaces to w * h (before normalization): %fms\n", timer.time);
    PRINT("INFO", "Transforming eigenfaces... done\n");

    START_TIMER(timer);
    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_eigenfaces), num_to_keep * w * h * sizeof(float))
    );

    dimOfGrid.x = ceil(w * h * num_to_keep / 1024.0 / 4);
    transpose_matrix_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(d_big_eigenfaces, dataset->d_eigenfaces, num_to_keep * w * h, num_to_keep, w * h);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to allocate and copy eigenfaces (transposing them) to GPU: %fms\n", timer.time);

    // Normalizing eigenfaces on GPU
    START_TIMER(timer);
    for (int i = 0; i < num_to_keep; i++) {
        normalize_gpu(dataset->d_eigenfaces + i * w * h, w * h);
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            PRINT("BUG", "kernel launch failed with error \"%s\"\n",
                  cudaGetErrorString(cudaerr));
            exit(EXIT_FAILURE);
        }
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "compute_eigenfaces_gpu: Time to normalize eigenfaces on GPU: %fms\n", timer.time);
    PRINT("INFO", "Normalizing eigenfaces... done\n");

    GPU_CHECKERROR(cudaFree(d_A));
    GPU_CHECKERROR(cudaFree(d_tA));
    GPU_CHECKERROR(cudaFree(d_covariance_matrix));
    GPU_CHECKERROR(cudaFree(d_small_eigenfaces));
    GPU_CHECKERROR(cudaFree(d_big_eigenfaces));
    free(covariance_matrix);
    free(eigenfaces);
    free(eigenvalues);
    free(h_small_eigenfaces);
    FREE_TIMER(timer);

    return 0;
}

__global__
void matrix_mult_gpu_kernel(float *M, float *N, float *C, int w_M, int h_M, int w_N, int h_N)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // allocate tiles in __shared__ memory
    __shared__ float s_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_N[TILE_WIDTH][TILE_WIDTH];

    // calculate the row & col index
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float result = 0;

    for (int p = 0; p < (w_M + TILE_WIDTH - 1) / TILE_WIDTH; p++) {
        // collaboratively load tiles into __shared__
        if (p * TILE_WIDTH + tx >= w_M || row >= h_M)
            s_M[ty][tx] = 0;
        else
            s_M[ty][tx] = M[row * w_M + p * TILE_WIDTH + tx];

        if (p * TILE_WIDTH + ty >= h_N || col >= w_N)
            s_N[ty][tx] = 0;
        else
            s_N[ty][tx] = N[(p * TILE_WIDTH + ty) * w_N + col];
        __syncthreads();

        // dot product between row of s_M and col of s_N
        for(int k = 0; k < TILE_WIDTH; k++)
          result += s_M[ty][k] * s_N[k][tx];
        __syncthreads();
    }

    if (row < h_M && col < w_N)
        C[row * w_N + col] = result;
}


// Assumes images are valid and dataset not NULL
// Set use_original_images to 1 to compute coordinates of original images
// (already loaded on GPU), otherwise set it to 0 and use images
struct FaceCoordinatesGPU ** compute_weighs_gpu(struct DatasetGPU *dataset, struct ImageGPU **images, int use_original_images, int k, int add_to_dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int num_eigens = dataset->num_eigenfaces;
    int n = dataset->num_faces;

    float *d_images_to_use;
    if (use_original_images) {
        d_images_to_use = dataset->d_original_images;
    } else {
        GPU_CHECKERROR(
        cudaMalloc((void **)&d_images_to_use, k * w * h * sizeof(float))
        );
        for (int i = 0; i < k; i++) {
            GPU_CHECKERROR(
            cudaMemcpy((void*)(d_images_to_use + i * w * h),
            (void*)(images[i]->data),
            w * h * sizeof(float),
            cudaMemcpyHostToDevice)
            );
        }
    }

    struct FaceCoordinatesGPU **new_faces = (struct FaceCoordinatesGPU **)malloc(k * sizeof(struct FaceCoordinatesGPU *));
    TEST_MALLOC(new_faces);

    int num_blocks = ceil(w * h / 1024.0 / 4);
    float *d_partial_sum;
    float *h_partial_sum;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_partial_sum, num_blocks * sizeof(float))
    );
    //h_partial_sum = (float *)malloc(num_blocks * sizeof(float));
    GPU_CHECKERROR(
    cudaMallocHost((void**)&h_partial_sum, num_blocks * sizeof(float))
    );
    TEST_MALLOC(h_partial_sum);

    for (int i = 0; i < k; i++) {
        new_faces[i] = (struct FaceCoordinatesGPU *)malloc(sizeof(struct FaceCoordinatesGPU));
        TEST_MALLOC(new_faces[i]);
        struct FaceCoordinatesGPU *current_face = new_faces[i];
        if (use_original_images)
            strcpy(current_face->name, dataset->original_names[i]);
        else
            strcpy(current_face->name, images[i]->filename);

        char *c = strrchr(current_face->name, '.');
        if (c)
            *c = '\0';

        current_face->num_eigenfaces = num_eigens;
        current_face->coordinates = (float *)malloc(num_eigens * sizeof(float));
        TEST_MALLOC(current_face->coordinates);

        for (int j = 0; j < num_eigens; j++)
            current_face->coordinates[j] = dot_product_gpu(d_images_to_use + i * w * h, dataset->d_eigenfaces + j * w * h, w * h, 0, d_partial_sum, h_partial_sum, num_blocks, 1024);
    }

    if (add_to_dataset) {
        dataset->faces = (struct FaceCoordinatesGPU **)realloc(dataset->faces, (n + k) * sizeof(struct FaceCoordinatesGPU *));
        TEST_MALLOC(dataset->faces);
        dataset->num_faces = n + k;

        for (int i = n; i < n + k; i++)
            dataset->faces[i] = new_faces[i - n];
    }

    //free(h_partial_sum);
    cudaFreeHost(h_partial_sum);
    GPU_CHECKERROR(cudaFree(d_partial_sum));
    if (!use_original_images) {
        GPU_CHECKERROR(cudaFree(d_images_to_use));
    }
    return new_faces;
}

__global__
void euclidian_distance_square_gpu_kernel(float *d_a, float *d_b, int size, float *d_partial_sum)
{
    extern __shared__ float s_thread_sums[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    if (i < size) {
        float diff = d_a[i] - d_b[i];
        s_thread_sums[t] = diff * diff;
    } else {
        s_thread_sums[t] = 0;
    }
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (t < stride)
            s_thread_sums[t] += s_thread_sums[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        volatile float *cache = s_thread_sums;
        cache[t] += cache[t + 32];
        cache[t] += cache[t + 16];
        cache[t] += cache[t + 8];
        cache[t] += cache[t + 4];
        cache[t] += cache[t + 2];
        cache[t] += cache[t + 1];
    }
    if (t == 0)
        d_partial_sum[blockIdx.x] = s_thread_sums[0];
}

float euclidian_distance_gpu(float *d_a, float *d_b, int size)
{
    int num_blocks = ceil(size / 1024.0);
    dim3 dimOfGrid(num_blocks, 1, 1);
    dim3 dimOfBlock(1024, 1, 1);
    if (num_blocks == 1) {
        dimOfBlock.x = 32;
        while (dimOfBlock.x < size)
            dimOfBlock.x *= 2;
    }
    int size_shared_mem = dimOfBlock.x * sizeof(float);

    float *d_partial_sum;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_partial_sum, num_blocks * sizeof(float))
    );
    float *h_partial_sum = (float *)malloc(num_blocks * sizeof(float));
    TEST_MALLOC(h_partial_sum);
    float result = 0;

    euclidian_distance_square_gpu_kernel<<<dimOfGrid, dimOfBlock, size_shared_mem>>>(d_a, d_b, size, d_partial_sum);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    GPU_CHECKERROR(
    cudaMemcpy((void*)h_partial_sum,
               (void*)d_partial_sum,
               num_blocks * sizeof(float),
               cudaMemcpyDeviceToHost)
    );
    cudaDeviceSynchronize();

    for (int i = 0; i < num_blocks; i++)
        result += h_partial_sum[i];
    result = sqrt(result);

    GPU_CHECKERROR(cudaFree(d_partial_sum));
    free(h_partial_sum);

    return result;
}

struct FaceCoordinatesGPU * get_closest_match_gpu(struct DatasetGPU *dataset, struct FaceCoordinatesGPU *face)
{
    float min = INFINITY;
    struct FaceCoordinatesGPU *closest = NULL;
    int num_eigens = dataset->num_eigenfaces;
    int num_faces = dataset->num_faces;
    Timer timer;
    INITIALIZE_TIMER(timer);

    START_TIMER(timer);
    float *d_faces;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_faces, (num_faces + 1) * num_eigens * sizeof(float))
    );
    for (int i = 0; i < num_faces; i++) {
        GPU_CHECKERROR(
        cudaMemcpy((void*)(d_faces + i * num_eigens),
                   (void*)dataset->faces[i]->coordinates,
                   num_eigens * sizeof(float),
                   cudaMemcpyHostToDevice)
        );
    }
    GPU_CHECKERROR(
    cudaMemcpy((void*)(d_faces + num_faces * num_eigens),
               (void*)face->coordinates,
               num_eigens * sizeof(float),
               cudaMemcpyHostToDevice)
    );
    STOP_TIMER(timer);
    PRINT("DEBUG", "get_closest_match_gpu: Time to allocate and copy faces to GPU: %fms\n", timer.time);

    START_TIMER(timer);
    for (int i = 0; i < num_faces; i++) {
        float distance = euclidian_distance_gpu(d_faces + num_faces * num_eigens, d_faces + i * num_eigens, num_eigens);
        PRINT("DEBUG", "Distance between %s and %s is %f\n", face->name, dataset->faces[i]->name, distance);
        if (distance < min) {
            min = distance;
            closest = dataset->faces[i];
        }
    }
    STOP_TIMER(timer);
    PRINT("DEBUG", "get_closest_match_gpu: Time to compute euclidian distance: %fms\n", timer.time);

    GPU_CHECKERROR(cudaFree(d_faces));
    FREE_TIMER(timer);
    return closest;
}
