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



struct Image * compute_average_cpu(struct Dataset * dataset);
__global__ void compute_average_gpu(struct Dataset * dataset, struct Image * average);

#endif


// pixel_grey = get_pixel(images[0], 0, 2, 0);

/*
struct Image *images;
// load images
int num_images = 100;


struct Image average;

average = compute_average_gpu(images, num_images);

save_average_to_dataset()

*/
