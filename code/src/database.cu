#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#include "database.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// User has to call free_image
struct Image load_image(char *filename, int req_comp) {
    struct Image image;
    image.data = stbi_load(filename, &(image.w), &(image.h), &(image.comp), req_comp);
    image.filename = filename;
    image.req_comp = req_comp;
    return image;
}

void free_image(struct Image image) {
    stbi_image_free(image.data);
}

// Assumes the image is loaded and x and y are correct coordinates
unsigned char get_pixel(struct Image image, int x, int y, int comp) {
    return image.data[(y * image.w + x) * image.comp + comp];
}
