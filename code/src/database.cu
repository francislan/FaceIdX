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

unsigned char * loadImage(char *filename, int *w, int *h, int *comp, int req_comp) {
    return stbi_load(filename, w, h, comp, req_comp);
}

void freeImage(unsigned char *data) {
    stbi_image_free(data);
}
