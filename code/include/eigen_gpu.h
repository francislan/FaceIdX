#ifndef EIGEN_GPU_H
#define EIGEN_GPU_H

struct DatasetGPU * create_dataset_and_compute_all_gpu(const char *path, const char *name);

//TODO
void normalize_gpu(float *array, int size);

//TODO: no need to return average, is returned only to save it on disk
struct ImageGPU * compute_average_gpu(struct DatasetGPU * dataset);
__global__ void compute_average_gpu_kernel(float *images, int w, int h, int num_image, float *average);

//TODO: not working
__global__ void dot_product_gpu(float *a, float *b, int size, float *result);

//TODO
void jacobi_gpu(const float *a, const int n, float *v, float *e);

//TODO
int compute_eigenfaces_gpu(struct DatasetGPU * dataset, int num_to_keep);

//TODO
struct FaceCoordinatesGPU ** compute_weighs_gpu(struct DatasetGPU *dataset, struct ImageGPU **images, float *d_images, int k, int add_to_dataset);

//TODO
struct FaceCoordinatesGPU * get_closest_match_gpu(struct DatasetGPU *dataset, struct FaceCoordinatesGPU *face);


#endif
