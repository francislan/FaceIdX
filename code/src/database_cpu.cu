#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "database_cpu.h"
#include "misc.h"
#include "eigen_cpu.h"
#include "load_save_image.h"

void free_image_cpu(struct ImageCPU *image)
{
    if (image == NULL)
	return;
    if (image->data != NULL)
	    free(image->data);
    free(image);
}

void free_face_cpu(struct FaceCoordinatesCPU *face)
{
    if (face == NULL)
	return;
    free(face->coordinates);
    free(face);
}

struct DatasetCPU * create_dataset_cpu(const char *directory, const char *name)
{
    char * line = NULL;
    size_t len = 0;
    int num_images = 0;
    int i = 0;
    int w = 0, h = 0;
    struct DatasetCPU *dataset = NULL;
    char command[200] = ""; // careful buffer overflow
    Timer timer;
    INITIALIZE_TIMER(timer);

    sprintf(command, "ls %s | grep png", directory);

    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        PRINT("BUG", "Cannot scan directory!\n");
        exit(EXIT_FAILURE);
    }

    while (getline(&line, &len, fp) != -1)
        num_images++;

    if (!num_images) {
        PRINT("WARN", "No such directory or no image in directory.\n");
        goto end;
    }

    PRINT("INFO", "%d images found in directory.\n", num_images);

    fclose(fp);
    fp = popen(command, "r"); // run the command twice, not optimal, and possible exploit

    dataset = (struct DatasetCPU *)malloc(sizeof(struct DatasetCPU));

    TEST_MALLOC(dataset);
    strcpy(dataset->name, name);
    dataset->path = "";
    dataset->num_original_images = num_images;
    dataset->original_images = (struct ImageCPU **)malloc(num_images * sizeof(struct ImageCPU *));
    TEST_MALLOC(dataset->original_images);

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, directory);
        strcat(image_name, "/");
        strcat(image_name, line);
        dataset->original_images[i] = load_image_cpu(image_name, 1);
	strcpy(dataset->original_images[i]->filename, line); // buffer overflow
        if (i == 0) {
            w = dataset->original_images[0]->w;
            h = dataset->original_images[0]->h;
        } else {
            if (w != dataset->original_images[i]->w || h != dataset->original_images[i]->h) {
                PRINT("WARN", "ImageCPUs in directory have different width and/or height. Aborting\n");
                free_dataset_cpu(dataset);
                dataset = NULL;
                goto end;
            }
        }
        i++;
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

// No input checking -> Not secure at all
struct DatasetCPU * load_dataset_cpu(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        PRINT("WARN", "Unable to open file %s!\n", path);
        return NULL;
    }
    struct DatasetCPU *dataset = (struct DatasetCPU *)malloc(sizeof(struct DatasetCPU));
    TEST_MALLOC(dataset);
    dataset->path = path;
    dataset->num_eigenfaces = 0;
    dataset->num_original_images = 0;
    dataset->num_faces = 0;
    dataset->num_new_faces = 0;
    dataset->w = 0;
    dataset->h = 0;
    dataset->original_images = NULL;
    dataset->average = NULL;
    dataset->eigenfaces = NULL;
    dataset->faces = NULL;

    for (int i = 0; i < 100; i++) {
        fread(dataset->name + i, sizeof(char), 1, f);
        if (dataset->name[i] == '\0')
            break;
    }
    fread(&(dataset->w), sizeof(int), 1, f);
    fread(&(dataset->h), sizeof(int), 1, f);
    int w = dataset->w;
    int h = dataset->h;
    fread(&(dataset->num_eigenfaces), sizeof(int), 1, f);

    dataset->average = (struct ImageCPU *)malloc(sizeof(struct ImageCPU));
    TEST_MALLOC(dataset->average);

    dataset->average->w = w;
    dataset->average->h = h;
    dataset->average->comp = 1;
    dataset->average->req_comp = 1;
    dataset->average->data = (float *)malloc(w * h * sizeof(float));
    TEST_MALLOC(dataset->average->data);
    for (int i = 0; i < 100; i++) {
        fread(dataset->average->filename + i, sizeof(char), 1, f);
        if (dataset->average->filename[i] == '\0')
            break;
    }
    fread(dataset->average->data, w * h * sizeof(float), 1, f);

    dataset->eigenfaces = (struct ImageCPU **)malloc(dataset->num_eigenfaces * sizeof(struct ImageCPU *));
    TEST_MALLOC(dataset->eigenfaces);

    for (int i = 0; i < dataset->num_eigenfaces; i++) {
        dataset->eigenfaces[i] = (struct ImageCPU *)malloc(sizeof(struct ImageCPU));
        TEST_MALLOC(dataset->eigenfaces[i]);

        dataset->eigenfaces[i]->w = w;
        dataset->eigenfaces[i]->h = h;
        dataset->eigenfaces[i]->comp = 1;
        dataset->eigenfaces[i]->req_comp = 1;
        dataset->eigenfaces[i]->data = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(dataset->eigenfaces[i]->data);
        for (int k = 0; k < 100; k++) {
            fread(dataset->eigenfaces[i]->filename + k, sizeof(char), 1, f);
            if (dataset->eigenfaces[i]->filename[k] == '\0')
               break;
        }
        fread(dataset->eigenfaces[i]->data, w * h * sizeof(float), 1, f);
    }

    int current_size = 0;
    char c;
    int num_allocated_faces = 50;
    dataset->faces = (struct FaceCoordinatesCPU **)malloc(num_allocated_faces * sizeof(struct FaceCoordinatesCPU *));
    TEST_MALLOC(dataset->faces);
    while (1) {
        size_t read = fread(&c, sizeof(char), 1, f);
        if (c != '\0' || read != 1)
            break;

        current_size++;
        if (current_size > num_allocated_faces) {
            num_allocated_faces *= 2;
            dataset->faces = (struct FaceCoordinatesCPU **)realloc(dataset->faces, num_allocated_faces * sizeof(struct FaceCoordinatesCPU *));
            TEST_MALLOC(dataset->faces);
        }

        dataset->faces[current_size - 1] = (struct FaceCoordinatesCPU *)malloc(sizeof(struct FaceCoordinatesCPU));
        TEST_MALLOC(dataset->faces[current_size - 1]);

        dataset->faces[current_size - 1]->num_eigenfaces = dataset->num_eigenfaces;
        dataset->faces[current_size - 1]->coordinates = (float *)malloc(dataset->num_eigenfaces * sizeof(float));
        TEST_MALLOC(dataset->faces[current_size - 1]->coordinates);
        for (int k = 0; k < 100; k++) {
            fread(dataset->faces[current_size - 1]->name + k, sizeof(char), 1, f);
            if (dataset->faces[current_size - 1]->name[k] == '\0')
               break;
        }
        fread(dataset->faces[current_size - 1]->coordinates, dataset->num_eigenfaces * sizeof(float), 1, f);
    }
    if (current_size < num_allocated_faces) {
        dataset->faces = (struct FaceCoordinatesCPU **)realloc(dataset->faces, current_size * sizeof(struct FaceCoordinatesCPU *));
        TEST_MALLOC(dataset->faces);
    }
    dataset->num_faces = current_size;
    fclose(f);
    return dataset;
}

// Call this only if dataset is well defined (not NULL, average computed, etc)
// Known bug: if saving to an existing file, the program doesn't know how
// many faces are already there, it will only save the ones that have been
// added since the last loading/creation of the database
int save_dataset_to_disk_cpu(struct DatasetCPU *dataset, const char *path)
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
            fwrite("\0", sizeof(char), 1, f);
            struct FaceCoordinatesCPU *face = dataset->faces[i];
            fwrite(face->name, sizeof(char), strlen(face->name), f);
            fwrite("\0", sizeof(char), 1, f);
            for (int j = 0; j < face->num_eigenfaces; j++)
                fwrite(&(face->coordinates[j]), sizeof(float), 1, f);
        }
        dataset->num_new_faces = 0;
        fclose(f);
    } else {
        FILE *f = fopen(path, "wb");
        if (f == NULL) {
            PRINT("WARNING", "Unable to create file %s!\n", path);
            return 1;
        }
        fwrite(dataset->name, sizeof(char), strlen(dataset->name), f);
        fwrite("\0", sizeof(char), 1, f);
        fwrite(&(dataset->w), sizeof(int), 1, f);
        fwrite(&(dataset->h), sizeof(int), 1, f);
        fwrite(&(dataset->num_eigenfaces), sizeof(int), 1, f);

        fwrite("Average", sizeof(char), strlen("Average"), f);
        fwrite("\0", sizeof(char), 1, f);
        fwrite(dataset->average->data, dataset->w * dataset-> h * sizeof(float), 1, f);

        char name[100] = "";

        for (int i = 0; i < dataset->num_eigenfaces; i++) {
            sprintf(name, "Eigen %d", i);
            fwrite(name, sizeof(char), strlen(name), f);
            fwrite("\0", sizeof(char), 1, f);
            fwrite(dataset->eigenfaces[i]->data, dataset->w * dataset->h * sizeof(float), 1, f);
        }

        // Do not write a '\0' at end of file
        // This is making the dataset loading function simpler
        if (dataset->num_faces > 0)
            fwrite("\0", sizeof(char), 1, f);
        for (int i = 0; i < dataset->num_faces; i++) {
            fwrite(dataset->faces[i]->name, sizeof(char), strlen(dataset->faces[i]->name), f);
            fwrite("\0", sizeof(char), 1, f);
            fwrite(dataset->faces[i]->coordinates, dataset->num_eigenfaces * sizeof(float), 1, f);
            // Do not write a '\0' at end of file
            // This is making the dataset loading function simpler
            if (i < dataset->num_faces - 1)
                fwrite("\0", sizeof(char), 1, f);
        }
        dataset->num_new_faces = 0;
        fclose(f);
    }
    return 0;
}

void free_dataset_cpu(struct DatasetCPU *dataset)
{
    if (dataset == NULL)
	return;
    for (int i = 0; i < dataset->num_original_images; i++)
	free_image_cpu(dataset->original_images[i]);
    free(dataset->original_images);

    for (int i = 0; i < dataset->num_eigenfaces; i++)
	free_image_cpu(dataset->eigenfaces[i]);
    free(dataset->eigenfaces);

    for (int i = 0; i < dataset->num_faces; i++)
	free_face_cpu(dataset->faces[i]);
    free(dataset->faces);

    free_image_cpu(dataset->average);
    free(dataset);
}


void save_reconstructed_face_to_disk_cpu(struct DatasetCPU *dataset, struct FaceCoordinatesCPU *face, int num_eigenfaces)
{
    struct ImageCPU *image = (struct ImageCPU *)malloc(sizeof(struct ImageCPU));
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
    for (int j = 0; j < image->w * image->h; j++)
        image->data[j] = (image->data[j] - min) / (max - min) * 255;

    sprintf(image->filename, "reconstructed/%s_with_%d.png", face->name, n);
    save_image_to_disk_cpu(image, image->filename);
    free_image_cpu(image);
}

// Expects dataset != NULL
// Returns the number of faces added
int add_faces_and_compute_coordinates_cpu(struct DatasetCPU *dataset, const char *path)
{
    char *line = NULL;
    size_t len = 0;
    int num_images = 0;
    int num_allocated = 0;
    int w = dataset->w;
    int h = dataset->h;
    int i = 0;
    struct ImageCPU **images = NULL;
    char command[200] = ""; // careful buffer overflow
    FILE *fp = NULL;

    if (strstr(path, ".png")) {
        if(access(path, F_OK) == -1 ) {
            PRINT("WARN", "Cannot access file %s!\n", path);
            return 0;
        }
        struct ImageCPU *image = load_image_cpu(path, 1);
        if (w != image->w || h != image->h) {
            PRINT("WARN", "Images in directory have different width and/or height. Aborting\n");
            num_images = 0;
            goto end;
        }
        for (int j = 0; j < image->w * image->h; j++)
            image->data[j] -= dataset->average->data[j];
        compute_weighs_cpu(dataset, &image, 1, 1);
        dataset->num_new_faces++;
        free_image_cpu(image);
        return 1;
    }

    // Case path is a directory
    sprintf(command, "ls %s | grep png", path);

    fp = popen(command, "r");
    if (fp == NULL) {
        PRINT("BUG", "Cannot scan directory!\n");
        exit(EXIT_FAILURE);
    }

    while (getline(&line, &len, fp) != -1)
        num_images++;

    if (!num_images) {
        PRINT("WARN", "No such directory or no image in directory.\n");
        goto end;
    }

    PRINT("INFO", "%d images found in directory.\n", num_images);

    fclose(fp);
    fp = popen(command, "r"); // run the command twice, not optimal, and possible exploit
    num_allocated = num_images;
    images = (struct ImageCPU **)malloc(num_allocated * sizeof(struct ImageCPU *));
    TEST_MALLOC(images);

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, path);
        strcat(image_name, "/");
        strcat(image_name, line);
        images[i] = load_image_cpu(image_name, 1);
	strcpy(images[i]->filename, line); // possible buffer overflow
        if (w != images[i]->w || h != images[i]->h) {
            PRINT("WARN", "Images in directory have different width and/or height. Aborting\n");
            num_images = 0;
            goto end;
        }
        i++;
        PRINT("DEBUG", "Loading file: %s\n", images[i-1]->filename);
    }
    for (int k = 0; k < num_images; k++)
        for (int j = 0; j < w * h; j++)
            images[k]->data[j] -= dataset->average->data[j];
    compute_weighs_cpu(dataset, images, num_images, 1);
    dataset->num_new_faces += num_images;

end:
    fclose(fp);
    if (line)
        free(line);
    if (images) {
        for (int i = 0; i < num_allocated; i++)
            free_image_cpu(images[i]);
        free(images);
    }
    return num_images;
}

void identify_face_cpu(struct DatasetCPU *dataset, const char *path)
{
    char *answer;
    Timer timer;
    INITIALIZE_TIMER(timer);

    if (access(path, F_OK) == -1) {
        PRINT("WARN", "Cannot access file %s!\n", path);
        return;
    }
    START_TIMER(timer);
    struct ImageCPU *image = load_image_cpu(path, 1);
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_cpu: Time for loading image: %fms\n", timer.time);

    START_TIMER(timer);
    for (int j = 0; j < image->w * image->h; j++)
        image->data[j] -=  dataset->average->data[j];
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_cpu: Time for substracting average: %fms\n", timer.time);

    START_TIMER(timer);
    struct FaceCoordinatesCPU **faces = compute_weighs_cpu(dataset, &image, 1, 0);
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_cpu: Time for computing coordinates: %fms\n", timer.time);

    struct FaceCoordinatesCPU *face = faces[0];
    START_TIMER(timer);
    struct FaceCoordinatesCPU *closest = get_closest_match_cpu(dataset, face);
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_cpu: Time for getting closest match: %fms\n", timer.time);

    if (closest == NULL) {
        printf("No match found!\n\n");
    } else {
        printf("Match found: %s\n\n", closest->name);
        strcpy(face->name, closest->name);
    }
    printf("Would you like to add the new face into the database [y/n]? ");
    get_user_string(&answer);

    if (!strcmp(answer, "y") || !strcmp(answer, "Y")) {
        printf("Enter name for the new face (leave blank to use '%s'): ", face->name);
        get_user_string(&answer);
        if (strlen(answer) != 0)
            strcpy(face->name, answer);
        dataset->faces = (struct FaceCoordinatesCPU **)realloc(dataset->faces, (dataset->num_faces + 1) * sizeof(struct FaceCoordinatesCPU *));
        TEST_MALLOC(dataset->faces);
        dataset->faces[dataset->num_faces] = face;
        dataset->num_faces++;
        dataset->num_new_faces++;
    } else {
        free_face_cpu(face);
    }
    free(answer);
    free_image_cpu(image);
    free(faces);
    FREE_TIMER(timer);
}
