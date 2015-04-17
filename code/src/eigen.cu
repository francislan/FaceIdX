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
    average->minus_average = (float *)malloc(w * h * sizeof(float));
    TEST_MALLOC(average->minus_average);

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int sum = 0;
            for (int i = 0; i < n; i++)
                sum += GET_PIXEL(dataset->original_images[i], x, y, 0);
            average->data[y * w + x + 0] = (sum / n);
        }
    }

    // Normalise
    float mean = 0;
    for (int j = 0; j < w * h; j++)
        mean += average->data[j];
    mean /= (w * h);
    for (int j = 0; j < w * h; j++)
        average->minus_average[j] =  average->data[j] / mean;
    float norm = sqrt(dot_product_cpu(average->minus_average, average->minus_average, w * h));
    for (int j = 0; j < w * h; j++)
        average->minus_average[j] /= norm;

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
    h_average->minus_average = NULL;

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

float dot_product_cpu(float *a, float *b, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
        sum += a[i] * b[i];

    return sum;
}

// Expect v to be initialized to 0
void jacobi_cpu(const float *a, const int n, float *v, float *e)
{
    int p, q, flag, t = 0;
    float temp;
    float theta, zero = 1e-8, max, pi = 3.141592654, c, s;
    float *d = (float *)malloc(n * n * sizeof(float));
    for (int i = 0; i < n * n; i++)
        d[i] = a[i];

    for(int i = 0; i < n; i++)
        v[i * n + i] = 1;

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

    printf("Nb of iterations: %d\n", t);
    printf("The eigenvalues are \n");
    for(int i = 0; i < n; i++)
        printf("%8.5f ", d[i * n + i]);

    printf("\nThe corresponding eigenvectors are \n");
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++)
            printf("% 8.5f,",v[i * n + j]);
        printf("\n");
    }
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

    float **images_minus_average = (float **)malloc(n * sizeof(float *));
    TEST_MALLOC(images_minus_average);

    for (int i = 0; i < n; i++) {
        dataset->original_images[i]->minus_average = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(dataset->original_images[i]->minus_average);
        images_minus_average[i] = dataset->original_images[i]->minus_average;
    }

    // Test minus average
/*
    struct Image *minus_average = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(minus_average);
    minus_average->w = w;
    minus_average->h = h;
    minus_average->comp = 1;
    minus_average->req_comp = 1;
    minus_average->minus_average = NULL;
*/
    // Substract average to images
    struct Image *average = dataset->average;
    for (int i = 0; i < n; i++) {
        struct Image *current_image = dataset->original_images[i];
        // Maybe switching the 2 loops results in faster computation
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                images_minus_average[i][y * w + x] = (float)GET_PIXEL(current_image, x, y, 0) - GET_PIXEL(average, x, y, 0);
        // Normalize images_minus_average
        /*float norm = sqrt(dot_product_cpu(images_minus_average[i], images_minus_average[i], w * h));
        PRINT("INFO", "Norm: %f\n", norm);
        for (int j = 0; j < w * h; j++)
            images_minus_average[i][j] /= norm;
*/
/*        sprintf(minus_average->filename, "minus/Minus Average %d.png", i);
        minus_average->data = (unsigned char *)malloc(w * h * 1 * sizeof(unsigned char));
        for (int j = 0; j < w * h; j++)
            minus_average->data[j] = images_minus_average[i][j] > 0 ?
                (unsigned char)((images_minus_average[i][j] / 256) * 127 + 128) :
                (unsigned char)(128 - (images_minus_average[i][j] / -256) * 128);
        save_image_to_disk(minus_average, minus_average->filename);
        free(minus_average->data);
*/  }
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

    // Check eigenvectors are correct
    PRINT("DEBUG", "Eigenvalues are:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", eigenvalues[2*i+0]);
    }

    for (int i = 0; i < n; i++) {
        float temp = 0;
        for (int j = 0; j < n; j++)
            temp += covariance_matrix[i * n + j] * eigenfaces[j * n + 1];
        PRINT("DEBUG", "%f %f\n", eigenvalues[2*1+0], eigenfaces[i * n + 1]);
        PRINT("DEBUG", "C*v %d = %f, lambda * v %d = %f\n", i, temp, i, eigenvalues[2*1+0] * eigenfaces[i * n + 1]);
    }


    // Keep only top num_to_keep eigenfaces.
    // Assumes num_to_keep is in the correct range.
    qsort(eigenvalues, n, 2 * sizeof(float), comp_eigenvalues);
    for (int i = 0; i < n; i++)
        PRINT("DEBUG", "Eigenvalue #%d (index %d): %f\n", i, (int)eigenvalues[2 * i + 1], eigenvalues[2 * i]);

    // Convert size n eigenfaces to size w*h
    dataset->num_eigenfaces = num_to_keep;
    dataset->eigenfaces = (float **)malloc(num_to_keep * sizeof(float *));
    TEST_MALLOC(dataset->eigenfaces);
    for (int i = 0; i < num_to_keep; i++) {
        dataset->eigenfaces[i] = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(dataset->eigenfaces[i]);
    }

    // Normalize eigenfaces in the loop
    // Unprobable corner case if all values are >0 or <0
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

/*  for (int i = 0; i < w * h; i++)
        printf("%f ", dataset->eigenfaces[0][i]);
    printf("\n");
*/
    // normalize new eigenfaces
    for (int i = 0; i < num_to_keep; i++) {
        float mean = 0;
        for (int j = 0; j < w * h; j++)
            mean += dataset->eigenfaces[i][j];
        mean /= (w + h);
        for (int j = 0; j < w * h; j++)
            dataset->eigenfaces[i][j] /= mean;
        float norm = sqrt(dot_product_cpu(dataset->eigenfaces[i], dataset->eigenfaces[i], w * h));
        for (int j = 0; j < w * h; j++)
            dataset->eigenfaces[i][j] /= norm;
    }

    for (int i = 0; i < w * h; i++)
        printf("%f ", dataset->eigenfaces[0][i]);
    printf("\n");


    // Test if eigenfaces are orthogonal
    for (int i = 0; i < n; i++)
        PRINT("DEBUG", "<0|%d> = %f\n", i, dot_product_cpu(dataset->eigenfaces[0], dataset->eigenfaces[i], w * h));

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

void compute_weighs_cpu(struct Dataset *dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int num_eigens = dataset->num_eigenfaces;
    int n = dataset->num_original_images;

    dataset->faces = (struct FaceCoordinates **)malloc(n * sizeof(struct FaceCoordinates *));
    TEST_MALLOC(dataset->faces);
    dataset->num_faces = n;

    for (int i= 0; i < n; i++) {
        dataset->faces[i] = (struct FaceCoordinates *)malloc(sizeof(struct FaceCoordinates));
        TEST_MALLOC(dataset->faces[i]);
    }

    for (int i = 0; i < n; i++) {
        struct FaceCoordinates *current_face = dataset->faces[i];
        struct Image *current_image = dataset->original_images[i];
        strcpy(current_face->name, current_image->filename);
        char *c = strrchr(current_face->name, '.');
        if (c)
            *c = '\0';

        PRINT("DEBUG", "Name: %s\n", current_face->name);

        current_face->num_eigenfaces = num_eigens;
        current_face->coordinates = (float *)malloc(num_eigens * sizeof(float));
        TEST_MALLOC(current_face->coordinates);

        for (int j = 0; j < num_eigens; j++)
            current_face->coordinates[j] = dot_product_cpu(current_image->minus_average,
                                                dataset->eigenfaces[j], w * h);

        float mean = 0;
        for (int j = 0; j < num_eigens; j++)
            mean += current_face->coordinates[j];
        mean /= num_eigens;
        for (int j = 0; j < num_eigens; j++)
            current_face->coordinates[j] /= mean;
        float norm = sqrt(dot_product_cpu(current_face->coordinates, current_face->coordinates, num_eigens));
        for (int j = 0; j < num_eigens; j++)
            current_face->coordinates[j] /= norm;

        for (int j = 0; j < num_eigens; j++)
            printf("%f ", current_face->coordinates[j]);
        printf("\n");
    }
}

struct FaceCoordinates * get_closest_match_cpu(struct Dataset *dataset, struct FaceCoordinates *face)
{
    float min = 255; // is that the max?
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
    return closest;
}












