#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#include "eigen.h"
#include "database.h"
#include "misc.h"

#define THREADS_PER_BLOCK 256

struct Dataset * create_dataset_and_compute_all(const char *path, const char *name)
{
    struct Timer timer_cpu, timer_gpu;
    INITIALIZE_TIMER(timer_cpu);
    INITIALIZE_TIMER(timer_gpu);
    FILE *f = fopen("timer_log.txt", "w");
    if(f == NULL) {
        PRINT("BUG", "Error opening timer_log.txt file!\n");
        exit(EXIT_FAILURE);
    }

    START_TIMER(timer_cpu);
    printf("\nCreating database...\n\n");
    struct Dataset *dataset = create_dataset(path, name);
    STOP_TIMER(timer_cpu);
    fprintf(f, "Time taken for creating database on cpu: %3.1f ms\n", timer_cpu.time);
    if (dataset == NULL) {
        PRINT("BUG","Dataset creation failed\n");
        return NULL;
    }
    printf("\nCreating database... Done!\n\n");

    printf("Computing average... ");
    START_TIMER(timer_cpu);
    struct Image *average = compute_average_cpu(dataset);
    STOP_TIMER(timer_cpu);
    fprintf(f, "Time taken for computing average face on cpu: %3.1f ms\n", timer_cpu.time);
    if (average == NULL) {
        PRINT("BUG","\naverage computation failed\n");
        return NULL;
    }
    printf("Done!\n");

    START_TIMER(timer_cpu);
    save_image_to_disk(average, "average_cpu.png");
    STOP_TIMER(timer_cpu);
    fprintf(f, "Time taken for saving average on disk on cpu: %3.1f ms\n", timer_cpu.time);

    // Eigenfaces
    printf("Computing eigenfaces...\n");
    START_TIMER(timer_cpu);
    compute_eigenfaces_cpu(dataset, dataset->num_original_images);
    //compute_eigenfaces_cpu(dataset, 50);
    STOP_TIMER(timer_cpu);
    fprintf(f, "Time taken for computing eigenfaces on cpu: %3.1f ms\n", timer_cpu.time);
    printf("Computing eigenfaces... Done!\n");

    printf("Compute images coordinates...\n");
    START_TIMER(timer_cpu);
    compute_weighs_cpu(dataset, dataset->original_images, dataset->num_original_images, 1);
    STOP_TIMER(timer_cpu);
    fprintf(f, "Time taken for computing weighs on cpu: %3.1f ms\n", timer_cpu.time);
    printf("Compute images coordinates... Done!\n");
    /*for (int i = 0; i < dataset->num_faces; i++)
        PRINT("INFO", "The Closest match of %s is %s.\n", dataset->faces[i]->name, get_closest_match_cpu(dataset, dataset->faces[i])->name);
*/
  //  save_dataset_to_disk(dataset, "dataset1.dat");


    START_TIMER(timer_gpu);
    struct Image *average_gpu = compute_average_gpu(dataset);
    STOP_TIMER(timer_gpu);

    fprintf(f, "Time taken for computing average face on gpu: %3.1f ms\n", timer_gpu.time);
    // not working, has to find another way to test average
    if (average_gpu == NULL) {
        PRINT("BUG","average computation failed\n");
        return NULL;
    }
    //PRINT("", "grey 0, 0: %f\n", GET_PIXEL(average_gpu, 0, 0, 0));
    //PRINT("", "grey 156, 15: %f\n", GET_PIXEL(average_gpu, 156, 15, 0));

    save_image_to_disk(average_gpu, "average_gpu.png");

    fclose(f);
    FREE_TIMER(timer_cpu);
    FREE_TIMER(timer_gpu);

    return dataset;
}

void normalize_cpu(float *array, int size)
{
    float mean = 0;
    for (int j = 0; j < size; j++)
        mean += array[j];
    mean /= size;
    for (int j = 0; j < size; j++)
        array[j] /= mean;
    float norm = sqrt(dot_product_cpu(array, array, size));
    for (int j = 0; j < size; j++)
        array[j] /= norm;
}

// returns NULL if error, otherwise returns pointer to average
struct Image * compute_average_cpu(struct Dataset * dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int n = dataset->num_original_images;
    Timer timer;
    INITIALIZE_TIMER(timer);

    if (w <= 0 || h <= 0) {
        PRINT("WARN", "Dataset's width and/or height incorrect(s)\n");
        return NULL;
    }
    if (n <= 0) {
        PRINT("WARN", "No image in dataset\n");
        return NULL;
    }

    START_TIMER(timer);
    struct Image *average = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(average);

    average->w = w;
    average->h = h;
    average->comp = 1;
    average->data = (float *)malloc(w * h * sizeof(float));
    TEST_MALLOC(average->data);
    STOP_TIMER(timer);
    PRINT("INFO", "Time allocating average Image: %f\n", timer.time);

    START_TIMER(timer);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            float sum = 0;
            for (int i = 0; i < n; i++)
                sum += GET_PIXEL(dataset->original_images[i], x, y, 0);
            average->data[y * w + x + 0] = (sum / n);
        }
    }
    STOP_TIMER(timer);
    PRINT("INFO", "Time computing: %f\n", timer.time);

    // Normalize?
    //normalize_cpu(average->data, w * h);
    dataset->average = average;
    return average;
}


struct Image * compute_average_gpu(struct Dataset * dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int n = dataset->num_original_images;
    Timer timer;
    INITIALIZE_TIMER(timer);
    printf("entering compute_average_gpu()...\n");
    if (w <= 0 || h <= 0) {
        PRINT("WARN", "Dataset's width and/or height incorrect(s)\n");
        return NULL;
    }
    if (n <= 0) {
        PRINT("WARN", "No image in dataset\n");
        return NULL;
    }

    START_TIMER(timer);
    float *h_average_image = (float *)malloc(w * h * sizeof(float));
    TEST_MALLOC(h_average_image);
    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_average), w * h * sizeof(float))
    );
    STOP_TIMER(timer);
    PRINT("INFO", "Time allocating average Image to GPU: %f\n", timer.time);

    START_TIMER(timer);
    dim3 dimOfGrid(ceil(w * 1.0 / 32), ceil(h * 1.0 / 32), 1);
    dim3 dimOfBlock(32, 32, 1);
    compute_average_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(dataset->d_original_images, w, h, n, dataset->d_average);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS) {
        PRINT("WARN", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        return NULL;
    }
    STOP_TIMER(timer);
    PRINT("INFO", "Time computing on GPU: %f\n", timer.time);

    START_TIMER(timer);
    GPU_CHECKERROR(
    cudaMemcpy((void*)h_average_image,
               (void*)dataset->d_average,
               w * h * sizeof(float),
               cudaMemcpyDeviceToHost)
    );
    STOP_TIMER(timer);
    PRINT("INFO", "Time copying average back to host: %f\n", timer.time);


    struct Image *h_average = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(h_average);
    h_average->data = h_average_image;
    h_average->w = w;
    h_average->h = h;
    h_average->comp = 1;

    dataset->average = h_average;
    printf("exiting compute_average_gpu()...\n");
    return h_average;
}


__global__
void compute_average_gpu_kernel(float *images, int w, int h, int num_image, float *average)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= w || y >= h)
        return;
    float sum = 0;
    for (int i = 0; i < num_image; i++)
        sum += images[i * w * h + y * w + x + 0];
    average[y * w + x + 0] = (sum / num_image);
    return;
}

float dot_product_cpu(float *a, float *b, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
        sum += a[i] * b[i];

    return sum;
}

// Makes sure the thread size is greater of equal to the size of the vectors
__global__
void dot_product_gpu_kernel(float *a, float *b, int size, float *result)
{
    extern __shared__ float s_thread_sums[];
    int i = threadIdx.x;
    s_thread_sums[i] = i < size ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (i < stride)
            s_thread_sums[i] += s_thread_sums[i + stride];
        __syncthreads();
    }
    if (i < 32) {
        volatile float *cache = s_thread_sums;
        cache[i] += cache[i + 32];
        cache[i] += cache[i + 16];
        cache[i] += cache[i + 8];
        cache[i] += cache[i + 4];
        cache[i] += cache[i + 2];
        cache[i] += cache[i + 1];
    }
    if (i == 0)
        *result = s_thread_sums[0];
    return;
}

// Expect v to be initialized to 0
void jacobi_cpu(const float *a, const int n, float *v, float *e)
{
    int p, q, flag, t = 0;
    float temp;
    float theta, zero = 1e-5, max, pi = 3.141592654, c, s;
    float *d = (float *)malloc(n * n * sizeof(float)); // no need to copy a, has to be changed
    Timer timer;
    INITIALIZE_TIMER(timer);

    START_TIMER(timer);
    for (int i = 0; i < n * n; i++)
        d[i] = a[i];

    for(int i = 0; i < n; i++)
        v[i * n + i] = 1;
    STOP_TIMER(timer);
    PRINT("INFO", "Jacobi: Time to copy and initialize matrix: %fms\n", timer.time);

    START_TIMER(timer);
    while(1) {
        flag = 0;
        p = 0;
        q = 1;
        max = fabs(d[0 * n + 1]);
        for(int i = 0; i < n; i++)
            for(int j = i + 1; j < n; j++) {
                temp = fabs(d[i * n + j]);
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
        if(d[p * n + p] == d[q * n + q]) {
            if(d[p * n + q] > 0)
                theta = pi/4;
            else
                theta = -pi/4;
        } else {
            theta = 0.5 * atan(2 * d[p * n + q] / (d[p * n + p] - d[q * n + q]));
        }
        c = cos(theta);
        s = sin(theta);

        for(int i = 0; i < n; i++) {
            temp = c * d[p * n + i] + s * d[q * n + i];
            d[q * n + i] = -s * d[p * n + i] + c * d[q * n + i];
            d[p * n + i] = temp;
        }

        for(int i = 0; i < n; i++) {
            temp = c * d[i * n  + p] + s * d[i * n + q];
            d[i * n + q] = -s * d[i * n + p] + c * d[i * n + q];
            d[i * n + p] = temp;
        }

        for(int i = 0; i < n; i++) {
            temp = c * v[i * n + p] + s * v[i * n + q];
            v[i * n + q] = -s * v[i * n + p] + c * v[i * n + q];
            v[i * n + p] = temp;
        }

    }
    STOP_TIMER(timer);
    PRINT("INFO", "Jacobi: time for main loop: %fms\n", timer.time);

    //printf("Nb of iterations: %d\n", t);
/*  printf("The eigenvalues are \n");
    for(int i = 0; i < n; i++)
        printf("%8.5f ", d[i * n + i]);

    printf("\nThe corresponding eigenvectors are \n");
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++)
            printf("% 8.5f,",v[i * n + j]);
        printf("\n");
    }*/
    for (int i = 0; i < n; i++) {
        e[2 * i + 0] = d[i * n + i];
        e[2 * i + 1] = i;
    }
    free(d);
}

// Expect v to be initialized to 0
void jacobi_gpu(const float *a, const int n, float *v, float *e)
{
    int p, q, flag, t = 0;
    float temp;
    float theta, zero = 1e-5, max, pi = 3.141592654, c, s;
    float *d = (float *)malloc(n * n * sizeof(float));
    Timer timer;
    INITIALIZE_TIMER(timer);

    START_TIMER(timer);
    for (int i = 0; i < n * n; i++)
        d[i] = a[i];

    for(int i = 0; i < n; i++)
        v[i * n + i] = 1;
    STOP_TIMER(timer);
    PRINT("INFO", "Jacobi: Time to copy and initialize matrix: %fms\n", timer.time);

    START_TIMER(timer);
    while(1) {
        flag = 0;
        p = 0;
        q = 1;
        max = fabs(d[0 * n + 1]);
        for(int i = 0; i < n; i++)
            for(int j = i + 1; j < n; j++) {
                temp = fabs(d[i * n + j]);
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
        //if (t % 1000 == 0)
        //    PRINT("DEBUG", "Iteration %d, max = %f\n", t, max);
        t++;
        if(d[p * n + p] == d[q * n + q]) {
            if(d[p * n + q] > 0)
                theta = pi/4;
            else
                theta = -pi/4;
        } else {
            theta = 0.5 * atan(2 * d[p * n + q] / (d[p * n + p] - d[q * n + q]));
        }
        c = cos(theta);
        s = sin(theta);

        for(int i = 0; i < n; i++) {
            temp = c * d[p * n + i] + s * d[q * n + i];
            d[q * n + i] = -s * d[p * n + i] + c * d[q * n + i];
            d[p * n + i] = temp;
        }

        for(int i = 0; i < n; i++) {
            temp = c * d[i * n  + p] + s * d[i * n + q];
            d[i * n + q] = -s * d[i * n + p] + c * d[i * n + q];
            d[i * n + p] = temp;
        }

        for(int i = 0; i < n; i++) {
            temp = c * v[i * n + p] + s * v[i * n + q];
            v[i * n + q] = -s * v[i * n + p] + c * v[i * n + q];
            v[i * n + p] = temp;
        }

    }
    STOP_TIMER(timer);
    PRINT("INFO", "Jacobi: time for main loop: %fms\n", timer.time);

    //printf("Nb of iterations: %d\n", t);
/*  printf("The eigenvalues are \n");
    for(int i = 0; i < n; i++)
        printf("%8.5f ", d[i * n + i]);

    printf("\nThe corresponding eigenvectors are \n");
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++)
            printf("% 8.5f,",v[i * n + j]);
        printf("\n");
    }*/
    for (int i = 0; i < n; i++) {
        e[2 * i + 0] = d[i * n + i];
        e[2 * i + 1] = i;
    }
    free(d);
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
    Timer timer;
    INITIALIZE_TIMER(timer);

    START_TIMER(timer);
    float **images_minus_average = (float **)malloc(n * sizeof(float *));
    TEST_MALLOC(images_minus_average);
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to allocate images_minus_average: %f\n", timer.time);

    START_TIMER(timer);
    for (int i = 0; i < n; i++)
        images_minus_average[i] = dataset->original_images[i]->data;

    // Substract average to images
    struct Image *average = dataset->average;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < w * h; j++)
            images_minus_average[i][j] = images_minus_average[i][j] - average->data[j];
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to substract average %f\n", timer.time);
    PRINT("DEBUG", "Substracting average to images... done\n");

    // Construct the Covariance Matrix
    START_TIMER(timer);
    float *covariance_matrix = (float *)malloc(n * n * sizeof(float));
    TEST_MALLOC(covariance_matrix);
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to allocate covariance matrix %f\n", timer.time);

    START_TIMER(timer);
    for (int i = 0; i < n; i++) {
        covariance_matrix[i * n + i] = dot_product_cpu(images_minus_average[i], images_minus_average[i], w * h) / n;
        for (int j = i + 1; j < n; j++) {
            covariance_matrix[i * n + j] = dot_product_cpu(images_minus_average[i], images_minus_average[j],  w * h) / n;
            covariance_matrix[j * n + i] = covariance_matrix[i * n + j];
        }
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to compute covariance matrix %f\n", timer.time);
    PRINT("DEBUG", "Building covariance matrix... done\n");

    // Compute eigenfaces
    START_TIMER(timer);
    float *eigenfaces = (float *)calloc(n * n, sizeof(float));
    TEST_MALLOC(eigenfaces);
    // eigenvalues stores couple (ev, index), makes it easier to get the top K
    // later
    float *eigenvalues = (float *)malloc(2 * n * sizeof(float));
    TEST_MALLOC(eigenvalues);
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to allocate arrays for jacobi %f\n", timer.time);

    START_TIMER(timer);
    jacobi_cpu(covariance_matrix, n, eigenfaces, eigenvalues);
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to do jacobi CPU %f\n", timer.time);
    PRINT("DEBUG", "Computing eigenfaces... done\n");


    // Keep only top num_to_keep eigenfaces.
    // Assumes num_to_keep is in the correct range.
    START_TIMER(timer);
    int num_eigenvalues_not_zero = 0;
    qsort(eigenvalues, n, 2 * sizeof(float), comp_eigenvalues);
    for (int i = 0; i < n; i++)
        if (eigenvalues[2 * i] > 0.5)
            num_eigenvalues_not_zero++;
    num_to_keep = num_eigenvalues_not_zero;
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to sort eigenvalues %f\n", timer.time);

    // Convert size n eigenfaces to size w*h
    START_TIMER(timer);
    dataset->num_eigenfaces = num_to_keep;
    dataset->eigenfaces = (struct Image **)malloc(num_to_keep * sizeof(struct Image *));
    TEST_MALLOC(dataset->eigenfaces);
    for (int i = 0; i < num_to_keep; i++) {
        dataset->eigenfaces[i] = (struct Image *)malloc(sizeof(struct Image));
        TEST_MALLOC(dataset->eigenfaces[i]);
        dataset->eigenfaces[i]->data = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(dataset->eigenfaces[i]->data);
        dataset->eigenfaces[i]->w = w;
        dataset->eigenfaces[i]->h = h;
        dataset->eigenfaces[i]->comp = 1;
        dataset->eigenfaces[i]->req_comp = 1;
        sprintf(dataset->eigenfaces[i]->filename, "Eigen_%d", i);
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to allocate eigenfaces %f\n", timer.time);

    START_TIMER(timer);
    float sqrt_n = sqrt(n);
    for (int i = 0; i < num_to_keep; i++) {
        int index = (int)eigenvalues[2 * i + 1];
        for (int j = 0; j < w * h; j++) {
            float temp = 0;
            for (int k = 0; k < n; k++)
                temp += images_minus_average[k][j] * eigenfaces[k * n + index];
            dataset->eigenfaces[i]->data[j] = temp / sqrt_n;
        }
        normalize_cpu(dataset->eigenfaces[i]->data, w * h);
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to transform eigenfaces to w * h %f\n", timer.time);

    PRINT("DEBUG", "Transforming eigenfaces... done\n");

    // Test if eigenfaces are orthogonal
    for (int i = 0; i < num_to_keep; i++)
        PRINT("DEBUG", "<0|%d> = %f\n", i, dot_product_cpu(dataset->eigenfaces[0]->data, dataset->eigenfaces[i]->data, w * h));

    // Test if eigenfaces before transform are orthogonal
    float *original_eigenfaces_5 = (float *)malloc(n * sizeof(float));
    float *original_eigenfaces_i = (float *)malloc(n * sizeof(float));
    for (int j = 0; j < n; j++)
        original_eigenfaces_5[j] = eigenfaces[j * n + 5];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            original_eigenfaces_i[j] = eigenfaces[j * n + i];
        PRINT("DEBUG", "<0|%d> = %f\n", i, dot_product_cpu(original_eigenfaces_5, original_eigenfaces_i, n));
    }

    free(covariance_matrix);
    free(eigenfaces);
    free(eigenvalues);
    return 0;
}

__global__
void substract_average_gpu_kernel(float *data, float *average, int size)
{
    int i = blockDim.x * threadIdx.y + threadIdx.x;
    if (i >= size)
        return;
    data[i] -= average[i];
}

// not finished at all
int compute_eigenfaces_gpu(struct Dataset * dataset, int num_to_keep)
{
    int n = dataset->num_original_images;
    int w = dataset->w;
    int h = dataset->h;
    Timer timer;
    INITIALIZE_TIMER(timer);

    // Substract average to images
    dim3 dimOfGrid(n, 1, 1);
    dim3 dimOfGridUnitary(1, 1, 1);
    dim3 dimOfBlock(w * h > 1024 ? 1024 : ceil(w * h / 32), w * h / 1024, 1 );

    START_TIMER(timer);
    substract_average_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(dataset->d_original_images, dataset->d_average, w * h);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS) {
        PRINT("WARN", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        return NULL;
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_gpu: Time to substract average: %fms\n", timer.time);

    PRINT("DEBUG", "Substracting average to images... done\n");

    // Construct the Covariance Matrix
    float *covariance_matrix = (float *)malloc(n * n * sizeof(float));
    TEST_MALLOC(covariance_matrix);

    START_TIMER(timer);
    for (int i = 0; i < n; i++) {

        dot_product_gpu_kernel<<<dimOfGridUnitary, dimOfBlock>>>(&(dataset->d_original_images[i * w * h]), &(dataset->d_original_images[i * w * h]), w * h, &covariance_matrix[i * n + i]);
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) {
            PRINT("WARN", "kernel launch failed with error \"%s\"\n",
                cudaGetErrorString(cudaerr));
            return NULL;
        }
        covariance_matrix[i * n + i] /= n;
        for (int j = i + 1; j < n; j++) {
            dot_product_gpu_kernel<<<dimOfGridUnitary, dimOfBlock>>>(&(dataset->d_original_images[i * w * h]), &(dataset->d_original_images[j * w * h]), w * h, &covariance_matrix[i * n + j]);
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != CUDA_SUCCESS) {
                PRINT("WARN", "kernel launch failed with error \"%s\"\n",
                    cudaGetErrorString(cudaerr));
                return NULL;
            }
            covariance_matrix[i * n + j] /= n;
            covariance_matrix[j * n + i] = covariance_matrix[i * n + j];
        }
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_gpu: Time to compute covariance matrix: %fms\n", timer.time);
    PRINT("DEBUG", "Building covariance matrix... done\n");

    // Compute eigenfaces
    float *eigenfaces = (float *)calloc(n * n, sizeof(float));
    TEST_MALLOC(eigenfaces);
    // eigenvalues stores couple (ev, index), makes it easier to get the top K
    // later
    float *eigenvalues = (float *)malloc(2 * n * sizeof(float));
    TEST_MALLOC(eigenvalues);

    START_TIMER(timer);
    jacobi_cpu(covariance_matrix, n, eigenfaces, eigenvalues);
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_gpu: Time to do jacobi cpu: %fms\n", timer.time);

    PRINT("DEBUG", "Computing eigenfaces... done\n");


    // Keep only top num_to_keep eigenfaces.
    // Assumes num_to_keep is in the correct range.
    int num_eigenvalues_not_zero = 0;
    qsort(eigenvalues, n, 2 * sizeof(float), comp_eigenvalues);
    for (int i = 0; i < n; i++) {
        //PRINT("DEBUG", "Eigenvalue #%d (index %d): %f\n", i, (int)eigenvalues[2 * i + 1], eigenvalues[2 * i]);
        if (eigenvalues[2 * i] > 0.5)
            num_eigenvalues_not_zero++;
    }
    num_to_keep = num_eigenvalues_not_zero;

    // Convert size n eigenfaces to size w*h
    dataset->num_eigenfaces = num_to_keep;
    dataset->eigenfaces = (struct Image **)malloc(num_to_keep * sizeof(struct Image *));
    TEST_MALLOC(dataset->eigenfaces);
    for (int i = 0; i < num_to_keep; i++) {
        dataset->eigenfaces[i] = (struct Image *)malloc(sizeof(struct Image));
        TEST_MALLOC(dataset->eigenfaces[i]);
        dataset->eigenfaces[i]->data = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(dataset->eigenfaces[i]->data);
        dataset->eigenfaces[i]->w = w;
        dataset->eigenfaces[i]->h = h;
        dataset->eigenfaces[i]->comp = 1;
        dataset->eigenfaces[i]->req_comp = 1;
        sprintf(dataset->eigenfaces[i]->filename, "Eigen_%d", i);
    }
/*
    START_TIMER(timer);
    float sqrt_n = sqrt(n);
    for (int i = 0; i < num_to_keep; i++) {
        int index = (int)eigenvalues[2 * i + 1];
        for (int j = 0; j < w * h; j++) {
            float temp = 0;
            for (int k = 0; k < n; k++)
                temp += images_minus_average[k][j] * eigenfaces[k * n + index];
            dataset->eigenfaces[i]->data[j] = temp / sqrt_n;
        }
        normalize_cpu(dataset->eigenfaces[i]->data, w * h);
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_gpu: Time to transform eigenfaces to w * h: %f\n", timer.time);
*/
    PRINT("DEBUG", "Transforming eigenfaces... done\n");

    // Copying eigenfaces to GPU
    START_TIMER(timer);
    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_eigenfaces), num_to_keep * w * h * sizeof(float))
    );
    for (int i = 0; i < num_to_keep; i++) {
        GPU_CHECKERROR(
        cudaMemcpy((void*)&(dataset->d_eigenfaces[i * w * h]),
                   (void*)dataset->eigenfaces[i]->data,
                   w * h * sizeof(float),
                   cudaMemcpyHostToDevice)
        );
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_gpu: Time to copy eigenfaces to GPU: %f\n", timer.time);

    free(covariance_matrix);
    free(eigenfaces);
    free(eigenvalues);
    return 0;
}

// Assumes images is valid and dataset not NULL
struct FaceCoordinates ** compute_weighs_cpu(struct Dataset *dataset, struct Image **images, int k, int add_to_dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int num_eigens = dataset->num_eigenfaces;
    int n = dataset->num_faces;

    struct FaceCoordinates **new_faces = (struct FaceCoordinates **)malloc(k * sizeof(struct FaceCoordinates *));
    TEST_MALLOC(new_faces);

    for (int i = 0; i < k; i++) {
        new_faces[i] = (struct FaceCoordinates *)malloc(sizeof(struct FaceCoordinates));
        TEST_MALLOC(new_faces[i]);
        struct FaceCoordinates *current_face = new_faces[i];
        struct Image *current_image = images[i];
        strcpy(current_face->name, current_image->filename);
        char *c = strrchr(current_face->name, '.');
        if (c)
            *c = '\0';

        //PRINT("DEBUG", "Name: %s\n", current_face->name);

        current_face->num_eigenfaces = num_eigens;
        current_face->coordinates = (float *)malloc(num_eigens * sizeof(float));
        TEST_MALLOC(current_face->coordinates);

        for (int j = 0; j < num_eigens; j++)
            current_face->coordinates[j] = dot_product_cpu(current_image->data,
                                                dataset->eigenfaces[j]->data, w * h);

        /*for (int j = 0; j < num_eigens; j++)
            printf("%f ", current_face->coordinates[j]);
        printf("\n");*/
    }

    if (add_to_dataset) {
        dataset->faces = (struct FaceCoordinates **)realloc(dataset->faces, (n + k) * sizeof(struct FaceCoordinates *));
        TEST_MALLOC(dataset->faces);
        dataset->num_faces = n + k;

        for (int i = n; i < n + k; i++)
            dataset->faces[i] = new_faces[i - n];
    }
    return new_faces;
}

// Assumes images is valid and dataset not NULL
// If the images are already loaded on GPU, set images to NULL and use
// d_images, otherwise set d_images to NULL and use images
struct FaceCoordinates ** compute_weighs_gpu(struct Dataset *dataset, struct Image **images, float *d_images, int k, int add_to_dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int num_eigens = dataset->num_eigenfaces;
    int n = dataset->num_faces;

    struct FaceCoordinates **new_faces = (struct FaceCoordinates **)malloc(k * sizeof(struct FaceCoordinates *));
    TEST_MALLOC(new_faces);

    for (int i = 0; i < k; i++) {
        new_faces[i] = (struct FaceCoordinates *)malloc(sizeof(struct FaceCoordinates));
        TEST_MALLOC(new_faces[i]);
        struct FaceCoordinates *current_face = new_faces[i];
        struct Image *current_image = images[i];
        strcpy(current_face->name, current_image->filename);
        char *c = strrchr(current_face->name, '.');
        if (c)
            *c = '\0';

        //PRINT("DEBUG", "Name: %s\n", current_face->name);

        current_face->num_eigenfaces = num_eigens;
        current_face->coordinates = (float *)malloc(num_eigens * sizeof(float));
        TEST_MALLOC(current_face->coordinates);

        for (int j = 0; j < num_eigens; j++)
            current_face->coordinates[j] = dot_product_cpu(current_image->data,
                                                dataset->eigenfaces[j]->data, w * h);

        /*for (int j = 0; j < num_eigens; j++)
            printf("%f ", current_face->coordinates[j]);
        printf("\n");*/
    }

    if (add_to_dataset) {
        dataset->faces = (struct FaceCoordinates **)realloc(dataset->faces, (n + k) * sizeof(struct FaceCoordinates *));
        TEST_MALLOC(dataset->faces);
        dataset->num_faces = n + k;

        for (int i = n; i < n + k; i++)
            dataset->faces[i] = new_faces[i - n];
    }
    return new_faces;
}

struct FaceCoordinates * get_closest_match_cpu(struct Dataset *dataset, struct FaceCoordinates *face)
{
    float min = INFINITY;
    struct FaceCoordinates *closest = NULL;
    int num_eigens = face->num_eigenfaces;
    float *diff = (float *)malloc(num_eigens * sizeof(float));
    TEST_MALLOC(diff);

    for (int i = 0; i < dataset->num_faces; i++) {
        for (int j = 0; j < num_eigens; j++)
            diff[j] = face->coordinates[j] - dataset->faces[i]->coordinates[j];
        float distance = sqrt(dot_product_cpu(diff, diff, num_eigens));
        PRINT("DEBUG", "Distance between %s and %s is %f\n", face->name, dataset->faces[i]->name, distance);
        if (distance < min) {
            min = distance;
            closest = dataset->faces[i];
        }
    }
    free(diff);
    return closest;
}
