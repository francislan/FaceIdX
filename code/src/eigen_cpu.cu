#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>

#include "eigen_cpu.h"
#include "database_cpu.h"
#include "misc.h"
#include "load_save_image.h"

#define THRES_EIGEN 1.0

struct DatasetCPU * create_dataset_and_compute_all_cpu(const char *path, const char *name)
{
    struct Timer timer_cpu;
    INITIALIZE_TIMER(timer_cpu);
    FILE *f = fopen("timer_log.txt", "w");
    if(f == NULL) {
        PRINT("BUG", "Error opening timer_log.txt file!\n");
        exit(EXIT_FAILURE);
    }

    START_TIMER(timer_cpu);
    printf("\nCreating database...\n\n");
    struct DatasetCPU *dataset = create_dataset_cpu(path, name);
    STOP_TIMER(timer_cpu);
    fprintf(f, "Time taken for creating database on cpu: %3.1f ms\n", timer_cpu.time);
    if (dataset == NULL) {
        PRINT("BUG","DatasetCPU creation failed\n");
        return NULL;
    }
    printf("\nCreating database... Done!\n\n");

    printf("Computing average... ");
    START_TIMER(timer_cpu);
    struct ImageCPU *average = compute_average_cpu(dataset);
    STOP_TIMER(timer_cpu);
    fprintf(f, "Time taken for computing average face on cpu: %3.1f ms\n", timer_cpu.time);
    if (average == NULL) {
        PRINT("BUG","\naverage computation failed\n");
        return NULL;
    }
    printf("Done!\n");

    START_TIMER(timer_cpu);
    save_image_to_disk_cpu(average, "average_cpu.png");
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


    fclose(f);
    FREE_TIMER(timer_cpu);

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
struct ImageCPU * compute_average_cpu(struct DatasetCPU * dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int n = dataset->num_original_images;
    Timer timer;
    INITIALIZE_TIMER(timer);

    if (w <= 0 || h <= 0) {
        PRINT("WARN", "DatasetCPU's width and/or height incorrect(s)\n");
        return NULL;
    }
    if (n <= 0) {
        PRINT("WARN", "No image in dataset\n");
        return NULL;
    }

    START_TIMER(timer);
    struct ImageCPU *average = (struct ImageCPU *)malloc(sizeof(struct ImageCPU));
    TEST_MALLOC(average);

    average->w = w;
    average->h = h;
    average->comp = 1;
    average->data = (float *)malloc(w * h * sizeof(float));
    TEST_MALLOC(average->data);
    STOP_TIMER(timer);
    PRINT("INFO", "Time allocating average ImageCPU: %f\n", timer.time);

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



float dot_product_cpu(float *a, float *b, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
        sum += a[i] * b[i];

    return sum;
}

// Expect v to be initialized to 0
void jacobi_cpu(float *a, const int n, float *v, float *e)
{
    int p, q, flag;
    float temp;
    float theta, zero = 1e-6, max, pi = 3.141592654, c, s;

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

    for (int i = 0; i < n; i++) {
        e[2 * i + 0] = a[i * n + i];
        e[2 * i + 1] = i;
    }
}

// Sorts in place the eigenvalues in descending order
int comp_eigenvalues_cpu(const void *a, const void *b)
{
    return (fabs(*(float *)a) < fabs(*(float *)b)) - (fabs(*(float *)a) > fabs(*(float *)b));
}

int compute_eigenfaces_cpu(struct DatasetCPU * dataset, int num_to_keep)
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
    struct ImageCPU *average = dataset->average;
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
    qsort(eigenvalues, n, 2 * sizeof(float), comp_eigenvalues_cpu);
    for (int i = 0; i < n; i++) {
        //PRINT("DEBUG", "Eigenvalue #%d (index %d): %f\n", i, (int)eigenvalues[2 * i + 1], eigenvalues[2 * i]);
        if (eigenvalues[2 * i] > THRES_EIGEN)
            num_eigenvalues_not_zero++;
    }
    num_to_keep = num_eigenvalues_not_zero;
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to sort eigenvalues %f\n", timer.time);

    // Convert size n eigenfaces to size w*h
    START_TIMER(timer);
    dataset->num_eigenfaces = num_to_keep;
    dataset->eigenfaces = (struct ImageCPU **)malloc(num_to_keep * sizeof(struct ImageCPU *));
    TEST_MALLOC(dataset->eigenfaces);
    for (int i = 0; i < num_to_keep; i++) {
        dataset->eigenfaces[i] = (struct ImageCPU *)malloc(sizeof(struct ImageCPU));
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
    /*printf("Eigen %d\n", 0);
    for (int j = 0; j < w * h; j++)
        printf("%f ", dataset->eigenfaces[0]->data[j]);
    printf("\n");*/
    STOP_TIMER(timer);
    PRINT("INFO", "compute_eigenfaces_cpu: Time to transform eigenfaces to w * h %f\n", timer.time);

    free(images_minus_average);
    free(covariance_matrix);
    free(eigenfaces);
    free(eigenvalues);
    return 0;
}

// Assumes images is valid and dataset not NULL
struct FaceCoordinatesCPU ** compute_weighs_cpu(struct DatasetCPU *dataset, struct ImageCPU **images, int k, int add_to_dataset)
{
    int w = dataset->w;
    int h = dataset->h;
    int num_eigens = dataset->num_eigenfaces;
    int n = dataset->num_faces;
    Timer timer;
    INITIALIZE_TIMER(timer);

    START_TIMER(timer);
    struct FaceCoordinatesCPU **new_faces = (struct FaceCoordinatesCPU **)malloc(k * sizeof(struct FaceCoordinatesCPU *));
    TEST_MALLOC(new_faces);

    for (int i = 0; i < k; i++) {
        new_faces[i] = (struct FaceCoordinatesCPU *)malloc(sizeof(struct FaceCoordinatesCPU));
        TEST_MALLOC(new_faces[i]);
        struct FaceCoordinatesCPU *current_face = new_faces[i];
        struct ImageCPU *current_image = images[i];
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
    STOP_TIMER(timer);
    PRINT("INFO", "compute_weighs_cpu: Time to commpute coordinates (including allocation): %fms\n", timer.time);

    START_TIMER(timer);
    if (add_to_dataset) {
        dataset->faces = (struct FaceCoordinatesCPU **)realloc(dataset->faces, (n + k) * sizeof(struct FaceCoordinatesCPU *));
        TEST_MALLOC(dataset->faces);
        dataset->num_faces = n + k;

        for (int i = n; i < n + k; i++)
            dataset->faces[i] = new_faces[i - n];
    }
    STOP_TIMER(timer);
    PRINT("INFO", "compute_weighs_cpu: Time to add to database: %fms\n", timer.time);
    FREE_TIMER(timer);
    return new_faces;
}


struct FaceCoordinatesCPU * get_closest_match_cpu(struct DatasetCPU *dataset, struct FaceCoordinatesCPU *face)
{
    float min = INFINITY;
    struct FaceCoordinatesCPU *closest = NULL;
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
