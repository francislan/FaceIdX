#ifndef EIGEN_H
#define EIGEN_H

struct Dataset * create_dataset_and_compute_all(const char *path, const char *name);
void normalize_cpu(float *array, int size);
struct Image * compute_average_cpu(struct Dataset * dataset);
struct Image * compute_average_gpu(struct Dataset * dataset);
__global__ void compute_average_gpu_kernel(float *images, int w, int h, int num_image, float *average);
float dot_product_cpu(float *a, float *b, int size);
__global__ void dot_product_gpu(float *a, float *b, int size, float *result);
void jacobi_cpu(const float *a, const int n, float *v, float *e);
void jacobi_gpu(const float *a, const int n, float *v, float *e);
int comp_eigenvalues(const void *a, const void *b);
int compute_eigenfaces_cpu(struct Dataset * dataset, int num_to_keep);
struct FaceCoordinates * get_closest_match_cpu(struct Dataset *dataset, struct FaceCoordinates *face);
struct FaceCoordinates ** compute_weighs_cpu(struct Dataset *dataset, struct Image **images, int k, int add_to_dataset);
struct FaceCoordinates ** compute_weighs_gpu(struct Dataset *dataset, struct Image **images, float *d_images, int k, int add_to_dataset);
int compute_eigenfaces_gpu(struct Dataset * dataset, int num_to_keep);
#endif
