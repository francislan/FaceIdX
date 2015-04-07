#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>

#include "nice_print.h"
#include "eigen.h"
#include "database.h"

int main(int argc, char **argv)
{
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }

    unsigned char *data = NULL;
    int w, h, n;

    data = loadImage("../../data/nottingham/f005a.png", &w, &h, &n, 0);
    if (data == NULL) {
        printf(KYEL "[Warning]: file could not be loaded.");
    } else {
        printf("Image width: %d, height: %d\n", w, h);
    }

    count_prime(argc, argv);
    return EXIT_SUCCESS;
}
