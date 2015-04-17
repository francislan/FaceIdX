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
struct Image * load_image(const char *filename, int req_comp)
{
    struct Image *image = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(image);
    image->data = stbi_load(filename, &(image->w), &(image->h), &(image->comp), req_comp);
    strcpy(image->filename, filename); // buffer overflow
    image->req_comp = req_comp;
    image->minus_average = NULL;
    return image;
}

void free_image(struct Image *image)
{
    if (image == NULL)
	return;
    stbi_image_free(image->data);
    if (image->minus_average != NULL)
	    free(image->minus_average);
    free(image);
}

void free_face(struct FaceCoordinates *face)
{
    if (face == NULL)
	return;
    free(face->coordinates);
    free(face);
}

struct Dataset * create_dataset(const char *directory, const char *dataset_path, const char *name)
{
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
	strcpy(dataset->original_images[i]->filename, line); // buffer overflow
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
    dataset->num_eigenfaces = 0;
    dataset->num_faces = 0;
    dataset->average = NULL;
    dataset->eigenfaces = NULL;
    dataset->faces = NULL;

end:
    fclose(fp);
    if (line)
        free(line);

    return dataset;
}

void free_dataset(struct Dataset *dataset)
{
    if (dataset == NULL)
	return;
    for (int i = 0; i < dataset->num_original_images; i++)
	free_image(dataset->original_images[i]);
    free(dataset->original_images);

    for (int i = 0; i < dataset->num_eigenfaces; i++)
	free(dataset->eigenfaces[i]);
    free(dataset->eigenfaces);

    for (int i = 0; i < dataset->num_faces; i++)
	free_face(dataset->faces[i]);
    free(dataset->faces);

    free_image(dataset->average);
    free(dataset);
}

void save_image_to_disk(struct Image *image, const char *name)
{
    stbi_write_png(name, image->w, image->h, 1, image->data, 0);
}

void save_eigenfaces_to_disk(struct Dataset *dataset)
{
    int n = dataset->num_eigenfaces;
    int w = dataset->w;
    int h = dataset->h;
    struct Image *image = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(image);
    image->data = (unsigned char *)malloc(w * h * 1 * sizeof(unsigned char));
    TEST_MALLOC(image->data);
    image->w = w;
    image->h = h;
    image->comp = 1;
    image->minus_average = NULL;

    for (int i = 0 ; i < n; i++) {
        float min = dataset->eigenfaces[i][0];
        float max = dataset->eigenfaces[i][0];
        for (int j = 1; j < w * h; j++) {
            float current = dataset->eigenfaces[i][j];
            if (current > max) {
                max = current;
            } else if (current < min) {
                min = current;
            }
        }
        sprintf(image->filename, "eigen/Eigenface %d.png", i);
        // TODO: bad conversion, to fix later
        for (int j = 0; j < w * h; j++)
            image->data[j] = dataset->eigenfaces[i][j] > 0 ?
                (unsigned char)((dataset->eigenfaces[i][j] / max) * 127 + 128) :
                (unsigned char)(128 - (dataset->eigenfaces[i][j] / min) * 128);
	save_image_to_disk(image, image->filename);
    }
    free_image(image);
}


void save_reconstructed_face_to_disk(struct Dataset *dataset, struct FaceCoordinates *face, int num_eigenfaces)
{
    struct Image *image = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(image);
    image->w = dataset->w;
    image->h = dataset->h;
    image->comp = 1;
    image->req_comp = 1;
    image->minus_average = (float *)calloc(image->w * image->h, sizeof(float));
    TEST_MALLOC(image->minus_average);
    image->data = (unsigned char *)malloc(image->w * image->h * sizeof(unsigned char));
    TEST_MALLOC(image->data);

    int n = num_eigenfaces > face->num_eigenfaces ? face->num_eigenfaces : num_eigenfaces;
    for (int i = 0; i < n; i++) {
        float weight = face->coordinates[i];
        for (int j = 0; j < image->w * image->h; j++)
            image->minus_average[j] += weight * dataset->eigenfaces[i][j];
    }

    for (int j = 0; j < image->w * image->h; j++)
        image->minus_average[j] += dataset->average->minus_average[j];


    float min = image->minus_average[0];
    float max = image->minus_average[0];
    for (int j = 1; j < image->w * image->h; j++) {
        float current = image->minus_average[j];
        if (current > max) {
            max = current;
        } else if (current < min) {
            min = current;
        }
    }
    PRINT("INFO", "Min: %f, Max: %f\n", min, max);
    for (int j = 0; j < image->w * image->h; j++)
        image->data[j] = (unsigned char)((image->minus_average[j] - min) / (max - min) * 255);

    sprintf(image->filename, "reconstructed/%s_with_%d.png", face->name, n);
    save_image_to_disk(image, image->filename);
    free_image(image);
}
