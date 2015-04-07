#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>

#include "nice_print.h"
#include "eigen.h"

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

    count_prime(argc, argv);
    return EXIT_SUCCESS;
}
