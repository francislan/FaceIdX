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

//    data = loadImage("../../Data/nottingham/original/f005a.png", &w, &h, &n, 1);
    if (data == NULL) {
        printf(KYEL "[Warning]: file could not be loaded.");
    } else {
        printf("Image width: %d, height: %d, comp: %d\n", w, h, n);
        printf("grey: %d\n", data[(0*w+0)* n + 0]);
        printf("grey: %d\n", data[(75*w+125)* n + 0]);
    }

    return EXIT_SUCCESS;
}
