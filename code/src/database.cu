#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "database.h"
#include "misc.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// User has to call free_image
struct Image * load_image(const char *filename, int req_comp) {
    struct Image *image = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(image);
    image->data = stbi_load(filename, &(image->w), &(image->h), &(image->comp), req_comp);
    strcpy(image->filename, filename); // buffer overflow
    image->req_comp = req_comp;
    return image;
}

void free_image(struct Image *image) {
    if (image == NULL)
	return;
    stbi_image_free(image->data);
    free(image);
}

struct Dataset * create_dataset(const char *directory, const char *dataset_path, const char *name) {
    char * line = NULL;
    size_t len = 0;
    int num_images = 0;
    int i = 0;
    int w = 0, h = 0;
    struct Dataset *dataset = NULL;
    char command[200] = ""; // careful buffer overflow

    sprintf(command, "ls %s | grep png", directory);

    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        PRINT("BUG", "Cannot scan directory!\n");
        exit(EXIT_FAILURE);
    }

    while (getline(&line, &len, fp) != -1) {
        if (strstr(line, "No such file or directory") || strstr(line, "not found")) {
            PRINT("WARN", "No such directory.\n");
            goto end;
        }
        num_images++;
    }

    if (!num_images) {
        PRINT("WARN", "No image in directory.\n");
        goto end;
    }

    PRINT("INFO", "%d images found in directory.\n", num_images);

    fclose(fp);
    fp = popen(command, "r"); // run the command twice, not optimal, and possible exploit

    dataset = (struct Dataset *)malloc(sizeof(struct Dataset));

    TEST_MALLOC(dataset);
    dataset->name = name;
    dataset->path = dataset_path;
    dataset->num_original_images = num_images;
    dataset->original_images = (struct Image **)malloc(num_images * sizeof(struct Image *));
    TEST_MALLOC(dataset->original_images);

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, directory);
        strcat(image_name, "/");
        strcat(image_name, line);
        dataset->original_images[i] = load_image(image_name, 1);
        if (i == 0) {
            w = dataset->original_images[0]->w;
            h = dataset->original_images[0]->h;
        } else {
            if (w != dataset->original_images[i]->w || h != dataset->original_images[i]->h) {
                PRINT("WARN", "Images in directory have different width and/or height. Aborting\n");
                free_dataset(dataset);
                dataset = NULL;
                goto end;
            }
        }
        i++;
        PRINT("DEBUG", "filename: %s\n", dataset->original_images[i-1]->filename);
    }
    dataset->w = w;
    dataset->h = h;

end:
    fclose(fp);
    if (line)
        free(line);

    return dataset;
}

void free_dataset(struct Dataset *dataset) {
    for (int i = 0; i < dataset->num_original_images; i++)
	free_image(dataset->original_images[i]);
    free(dataset->original_images);

    for (int i = 0; i < dataset->num_eigenfaces; i++)
	free_image(dataset->eigenfaces[i]);
    free(dataset->eigenfaces);

/*    for (int i = 0; i < dataset->num_faces; i++)
	free_face(dataset->faces[i]);
    free(dataset->faces);
*/
    free_image(dataset->average);
    free(dataset);
}

void save_image_to_disk(struct Image *image, const char *name) {
    stbi_write_png(name, image->w, image->h, 1, image->data, 0);
}
