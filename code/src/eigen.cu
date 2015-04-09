#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#include "eigen.h"
#include "database.h"
#include "misc.h"

#define THREADS_PER_BLOCK 256

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

