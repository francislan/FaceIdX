#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "database.h"
#include "nice_print.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// User has to call free_image
struct Image * load_image(const char *filename, int req_comp) {
    struct Image *image = (struct Image *)malloc(sizeof(struct Image));
    // check malloc
    image->data = stbi_load(filename, &(image->w), &(image->h), &(image->comp), req_comp);
    strcpy(image->filename, filename); // buffer overflow
    image->req_comp = req_comp;
    return image;
}

void free_image(struct Image *image) {
    stbi_image_free(image->data);
    free(image);
}

// Assumes the image is loaded and x and y are correct coordinates
unsigned char get_pixel(struct Image *image, int x, int y, int comp) {
    return image->data[(y * image->w + x) * image->comp + comp];
}


struct Dataset * create_dataset(const char *directory, const char *dataset_path, const char *name) {
    char * line = NULL;
    size_t len = 0;
    int num_images = 0;
    int i = 0;
    struct Dataset *dataset = NULL;
    char command[200] = ""; // careful buffer overflow

    sprintf(command, "ls %s | grep png", directory);

    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        printf(KRED "[Error]: Cannot scan directory!\n");
        exit(EXIT_FAILURE);
    }

    while (getline(&line, &len, fp) != -1) {
        if (strstr(line, "No such file or directory") || strstr(line, "not found")) {
            printf(KYEL "[Warning]: No such directory.\n");
            goto end;
        }
        printf(KNRM "filename: %s\n", line);
        num_images++;
    }

    if (!num_images) {
        printf(KYEL "[Warning]: No image in directory.\n");
        goto end;
    }

    printf(KBLU "[Info]: %d images found in directory.\n", num_images);

    fclose(fp);
    FILE *fp = popen(command, "r"); // run the command twice, not optimal, and possible exploit

    dataset = (struct Dataset *)malloc(sizeof(struct Dataset));
    // check malloc
    dataset->name = name;
    dataset->path = dataset_path;
    dataset->num_images = num_images;
    dataset->original_images = (struct Image **)malloc(num_images * sizeof(struct Image *));
    // check malloc

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, directory);
        strcat(image_name, "/");
        strcat(image_name, line);
        dataset->original_images[i++] = load_image(image_name, 1);
        printf(KGRN "[Debug]: dataset->original_images[i-1]->filename");
    }


end:
    fclose(fp);
    if (line)
        free(line);

    return dataset;
}
