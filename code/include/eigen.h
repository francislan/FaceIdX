#ifndef EIGEN_H
#define EIGEN_H

// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__ ))
static void gpuCheckError(cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                file, line );
        exit(EXIT_FAILURE);
    }
}


void normalize_cpu(float *array, int size);
struct Image * compute_average_cpu(struct Dataset * dataset);
struct Image * compute_average_gpu(struct Dataset * dataset);
__global__ void compute_average_gpu_kernel(float *images, int w, int h, int num_image, float *average);
float dot_product_cpu(float *a, float *b, int size);
void jacobi_cpu(const float *a, const int n, float *v, float *e);
int comp_eigenvalues(const void *a, const void *b);
int compute_eigenfaces_cpu(struct Dataset * dataset, int num_to_keep);
void compute_weighs_cpu(struct Dataset *dataset);
struct FaceCoordinates * get_closest_match_cpu(struct Dataset *dataset, struct FaceCoordinates *face);

int compute_eigenfaces_gpu(struct Dataset * dataset, int num_to_keep);
#endif
