#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>

#include "misc.h"
#include "eigen.h"
#include "database.h"

int main(int argc, char **argv)
{
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        PRINT("BUG", "Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    struct Image *image = load_image("../../Data/nottingham/normalized/f005a.png", 1);
    if (image->data == NULL) {
        PRINT("WARN", "file could not be loaded.\n");
    } else {
        PRINT("", "Image width: %d, height: %d, comp: %d\n", image->w, image->h, image->comp);
        PRINT("", "grey: %d\n", get_pixel(image, 0, 0, 0));
        PRINT("", "grey: %d\n", get_pixel(image, 156, 15, 0));
    }
    free_image(image);

    struct Dataset *dataset = create_dataset("../../Data/nottingham/normalized", "./dataset.dat", "Set 1");
    if (dataset == NULL) {
        PRINT("BUG","Dataset creation failed\n");
        return EXIT_FAILURE;
    }
    PRINT("", "Dataset name: %s\n", dataset->name);
    PRINT("", "Dataset path: %s\n", dataset->path);
    PRINT("", "Dataset num_original_images: %d\n", dataset->num_original_images);
    for (int i = 0; i < dataset->num_original_images; i++) {
        PRINT("", "\tImage %d: %s\n", i + 1, dataset->original_images[i]->filename);
        PRINT("", "grey 0, 0: %d\n", get_pixel(dataset->original_images[i], 0, 0, 0));
        PRINT("", "grey 156, 15: %d\n", get_pixel(dataset->original_images[i], 156, 15, 0));
    }
    struct Image *average = compute_average_cpu(dataset);
    if (average == NULL) {
        PRINT("BUG","average computation failed\n");
        return EXIT_FAILURE;
    }
    PRINT("", "grey 0, 0: %d\n", get_pixel(average, 0, 0, 0));
    PRINT("", "grey 156, 15: %d\n", get_pixel(average, 156, 15, 0));

    free_dataset(dataset);
    return EXIT_SUCCESS;
}
