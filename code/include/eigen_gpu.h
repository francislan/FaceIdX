#ifndef EIGEN_GPU_H
#define EIGEN_GPU_H

struct DatasetGPUGPU * create_dataset_and_compute_all_gpu(const char *path, const char *name);
void normalize_cpu(float *array, int size);
struct ImageGPU * compute_average_cpu(struct DatasetGPU * dataset);
struct ImageGPU * compute_average_gpu(struct DatasetGPU * dataset);
__global__ void compute_average_gpu_kernel(float *images, int w, int h, int num_image, float *average);
float dot_product_cpu(float *a, float *b, int size);
__global__ void dot_product_gpu(float *a, float *b, int size, float *result);
void jacobi_cpu(const float *a, const int n, float *v, float *e);
void jacobi_gpu(const float *a, const int n, float *v, float *e);
int comp_eigenvalues(const void *a, const void *b);
int compute_eigenfaces_cpu(struct DatasetGPU * dataset, int num_to_keep);
struct FaceCoordinatesGPU * get_closest_match_cpu(struct DatasetGPU *dataset, struct FaceCoordinatesGPU *face);
struct FaceCoordinatesGPU ** compute_weighs_cpu(struct DatasetGPU *dataset, struct ImageGPU **images, int k, int add_to_dataset);
struct FaceCoordinatesGPU ** compute_weighs_gpu(struct DatasetGPU *dataset, struct ImageGPU **images, float *d_images, int k, int add_to_dataset);
int compute_eigenfaces_gpu(struct DatasetGPU * dataset, int num_to_keep);
#endif
