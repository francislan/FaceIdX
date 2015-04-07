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

    struct Image image = loadImage("../../Data/nottingham/original/f005a.png", 1);
    if (image.data == NULL) {
        printf(KYEL "[Warning]: file could not be loaded.");
    } else {
        printf("Image width: %d, height: %d, comp: %d\n", image.w, image.h, image.comp);
        printf("grey: %d\n", image.data[(0*w+0)* n + 0]);
        printf("grey: %d\n", image.data[(75*w+125)* n + 0]);
    }

    return EXIT_SUCCESS;
}
