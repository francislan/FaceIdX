#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "database.h"
#include "nice_print.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// User has to call free_image
struct Image load_image(const char *filename, int req_comp) {
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


struct Dataset create_dataset(const char *directory, const char *dataset_path, char *name) {
    char * line = NULL;
    size_t len = 0;
    int num_images = 0;
    struct Dataset dataset = {0};

    FILE *fp = popen("ls `directory` | grep png", "r");
    if (fp == NULL) {
        printf(KRED "[Error]: Cannot scan directory!");
        exit(EXIT_FAILURE);
    }

    while (getline(&line, &len, fp) != -1) {
        if (strstr(line, "No such file or directory")) {
            printf(KYEL "[Warning]: No such directory.");
            goto end;
        }
        num_images++;
    }

    if (!num_images) {
        printf(KYEL "[Warning]: No image in directory.");
        goto end;
    }

    printf(KBLU "[Info]: %d images found in directory.", num_images);


end:
    fclose(fp);
    if (line)
        free(line);

    return dataset;
}
