#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#include "database.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__ ))
static void gpuCheckError(cudaError_t err,
                         const char *file,
                         int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                file, line);
        exit(EXIT_FAILURE);
    }
}

struct Image loadImage(char *filename, int req_comp) {
    struct Image image;
    image.data = stbi_load(filename, &(image.w), &(image.h), &(image.comp), req_comp);
    image.filename = filename;
    image.req_comp = req_comp;
    return image;
}

void freeImage(unsigned char *data) {
    stbi_image_free(data);
}
