#ifndef EIGEN_GPU_H
#define EIGEN_GPU_H

//TODO: calling cudaDeviceSynchronize after each kernel may cause some slow
//down. Can be removed later

struct DatasetGPU * create_dataset_and_compute_all_gpu(const char *path, const char *name);

__global__ void sum_gpu_kernel(float *d_a, int size, float *d_partial_sum);
void normalize_gpu(float *d_array, int size);
__global__ void divide_by_float_gpu_kernel(float *d_array, float constant, int size);

//No need to return average, is returned only to save it on disk
struct ImageGPU * compute_average_gpu(struct DatasetGPU * dataset);
__global__ void compute_average_gpu_kernel(float *d_images, int w, int h, int num_image, float *d_average);

__global__ void dot_product_gpu(float *d_a, float *d_b, int size, float *d_result);
float dot_product_gpu(float *d_a, float *d_b, int size, int allocate, float *d_partial_sum, float *h_partial_sum, int num_blocks, int num_threads);

__global__ void transpose_matrix_gpu_kernel(float *d_input, float *d_output, int total_size, int unitary_size, int count);

void jacobi_gpu(float *a, const int n, float *v, float *e);

int compute_eigenfaces_gpu(struct DatasetGPU * dataset, int num_to_keep);
__global__ void matrix_mult_gpu_kernel(float *d_A, float *d_B, float *d_C, int w_A, int h_A, int w_B, int h_B);

__global__ void substract_average_gpu_kernel(float *d_data, float *d_average, int size, int size_image);

void substract_average_gpu(struct ImageGPU **images, float *d_average, int num_images, int size_image);

struct FaceCoordinatesGPU ** compute_weighs_gpu(struct DatasetGPU *dataset, struct ImageGPU **images,int use_original_images, int k, int add_to_dataset);

__global__ void euclidian_distance_square_gpu_kernel(float *d_a, float *d_b, int size, float *d_partial_sum);
float euclidian_distance_gpu(float *d_a, float *d_b, int size);
struct FaceCoordinatesGPU * get_closest_match_gpu(struct DatasetGPU *dataset, struct FaceCoordinatesGPU *face);

#endif
