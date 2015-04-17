#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "database.h"
#include "misc.h"
#include "eigen.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// User has to call free_image
struct Image * load_image(const char *filename, int req_comp)
{
    struct Image *image = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(image);
    unsigned char *image_data = stbi_load(filename, &(image->w), &(image->h), &(image->comp), req_comp);
    strcpy(image->filename, filename); // buffer overflow
    image->req_comp = req_comp;
    image->data = (float *)malloc(image->w * image->h * sizeof(float));
    TEST_MALLOC(image->data);

    for (int j = 0; j < image->w * image->h; j++)
        image->data[j] = image_data[j];
    normalize_cpu(image->data, image->w * image->h);

    stbi_image_free(image_data);
    return image;
}

void free_image(struct Image *image)
{
    if (image == NULL)
	return;
    if (image->data != NULL)
	    free(image->data);
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
    dataset->num_new_faces = 0;
    dataset->average = NULL;
    dataset->eigenfaces = NULL;
    dataset->faces = NULL;

end:
    fclose(fp);
    if (line)
        free(line);

    return dataset;
}

struct Dataset * load_dataset(const char *path)
{
    return NULL;
}

// Call this only if dataset is well defined (not NULL, average computed, etc)
int save_dataset_to_disk(struct Dataset *dataset, const char *path)
{
    // No safe, TOCTOU, etc
    if(access(path, F_OK) != -1 ) {
        // file exists
        FILE *f = fopen(path, "ab");
        if (f == NULL) {
            PRINT("WARNING", "Unable to append to file %s!\n", path);
            return 1;
        }
        for (int i = dataset->num_faces - dataset->num_new_faces; i < dataset->num_faces; i++) {
            struct FaceCoordinates *face = dataset->faces[i];
            fwrite(face->name, sizeof(char), strlen(face->name), f);
            fwrite("\0", sizeof(char), 1, f);
            for (int j = 0; j < face->num_eigenfaces; j++)
                fwrite(&(face->coordinates[j]), sizeof(float), 1, f);
            fwrite("\0\0\0\0", sizeof(float), 1, f);
        }
        fclose(f);
    } else {
        FILE *f = fopen(path, "w");
        if (f == NULL) {
            PRINT("WARNING", "Unable to create file %s!\n", path);
            return 1;
        }
        fwrite(dataset->name, sizeof(char), strlen(dataset->name), f);
        fwrite("\0", sizeof(char), 1, f);
        fwrite(&(dataset->w), sizeof(int), 1, f);
        fwrite(&(dataset->h), sizeof(int), 1, f);
        fwrite(&(dataset->num_eigenfaces), sizeof(int), 1, f);
        float *data = NULL;

        fwrite("Average", sizeof(char), strlen("Average"), f);
        fwrite("\0", sizeof(char), 1, f);
        data = dataset->average->data;
        for (int j = 0; j < dataset->w * dataset->h; j++)
            fwrite(&(data[j]), sizeof(float), 1, f);
        fwrite("\0\0\0\0", sizeof(float), 1, f);

        char name[100] = "";

        for (int i = 0; i < dataset->num_eigenfaces; i++) {
            sprintf(name, "Eigen %d", i);
            fwrite(name, sizeof(char), strlen(name), f);
            fwrite("\0", sizeof(char), 1, f);
            data = dataset->eigenfaces[i]->data;
            for (int j = 0; j < dataset->w * dataset->h; j++)
                fwrite(&(data[j]), sizeof(float), 1, f);
            fwrite("\0\0\0\0", sizeof(float), 1, f);
        }

        for (int i = 0; i < dataset->num_faces; i++) {
            fwrite(dataset->faces[i]->name, sizeof(char), strlen(dataset->faces[i]->name), f);
            fwrite("\0", sizeof(char), 1, f);
            data = dataset->faces[i]->coordinates;
            for (int j = 0; j < dataset->num_eigenfaces; j++)
                fwrite(&(data[j]), sizeof(float), 1, f);
            fwrite("\0\0\0\0", sizeof(float), 1, f);
        }
        fclose(f);
    }
    return 0;
}

void free_dataset(struct Dataset *dataset)
{
    if (dataset == NULL)
	return;
    for (int i = 0; i < dataset->num_original_images; i++)
	free_image(dataset->original_images[i]);
    free(dataset->original_images);

    for (int i = 0; i < dataset->num_eigenfaces; i++)
	free_image(dataset->eigenfaces[i]);
    free(dataset->eigenfaces);

    for (int i = 0; i < dataset->num_faces; i++)
	free_face(dataset->faces[i]);
    free(dataset->faces);

    free_image(dataset->average);
    free(dataset);
}

void save_image_to_disk(struct Image *image, const char *name)
{
    int w = image->w;
    int h = image->h;
    unsigned char *image_data = (unsigned char *)malloc(w * h * 1 * sizeof(unsigned char));
    TEST_MALLOC(image_data);

    float min = image->data[0];
    float max = image->data[0];
    for (int j = 1; j < w * h; j++) {
        float current = image->data[j];
        if (current > max) {
            max = current;
        } else if (current < min) {
            min = current;
        }
    }
    // bad conversion from float to unsigned char
    for (int j = 0; j < w * h; j++)
        image_data[j] = image->data[j] > 0 ?
            (unsigned char)((image->data[j] / max) * 127 + 128) :
            (unsigned char)(128 - (image->data[j] / min) * 128);
    stbi_write_png(name, w, h, 1, image_data, 0);
    free(image_data);
}


void save_reconstructed_face_to_disk(struct Dataset *dataset, struct FaceCoordinates *face, int num_eigenfaces)
{
    struct Image *image = (struct Image *)malloc(sizeof(struct Image));
    TEST_MALLOC(image);
    image->w = dataset->w;
    image->h = dataset->h;
    image->comp = 1;
    image->req_comp = 1;
    image->data = (float *)calloc(image->w * image->h, sizeof(float));
    TEST_MALLOC(image->data);

    int n = num_eigenfaces > face->num_eigenfaces ? face->num_eigenfaces : num_eigenfaces;
    for (int i = 0; i < n; i++) {
        float weight = face->coordinates[i];
        for (int j = 0; j < image->w * image->h; j++)
            image->data[j] += weight * dataset->eigenfaces[i]->data[j];
    }

    for (int j = 0; j < image->w * image->h; j++)
        image->data[j] += dataset->average->data[j];


    float min = image->data[0];
    float max = image->data[0];
    for (int j = 1; j < image->w * image->h; j++) {
        float current = image->data[j];
        if (current > max) {
            max = current;
        } else if (current < min) {
            min = current;
        }
    }
    PRINT("INFO", "Min: %f, Max: %f\n", min, max);
    for (int j = 0; j < image->w * image->h; j++)
        image->data[j] = (image->data[j] - min) / (max - min) * 255;

    sprintf(image->filename, "reconstructed/%s_with_%d.png", face->name, n);
    save_image_to_disk(image, image->filename);
    free_image(image);
}
