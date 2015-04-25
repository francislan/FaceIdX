#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "database_gpu.h"
#include "misc.h"
#include "eigen_gpu.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// User has to call free_image
struct ImageGPU * load_image_gpu(const char *filename, int req_comp)
{
    struct ImageGPU *image = (struct ImageGPU *)malloc(sizeof(struct ImageGPU));
    TEST_MALLOC(image);
    unsigned char *image_data = stbi_load(filename, &(image->w), &(image->h), &(image->comp), req_comp);
    strcpy(image->filename, filename); // buffer overflow
    image->req_comp = req_comp;
    image->data = (float *)malloc(image->w * image->h * sizeof(float));
    TEST_MALLOC(image->data);

    for (int j = 0; j < image->w * image->h; j++)
        image->data[j] = image_data[j];

    stbi_image_free(image_data);
    return image;
}

void free_image_gpu(struct ImageGPU *image)
{
    if (image == NULL)
	return;
    if (image->data != NULL)
	    free(image->data);
    free(image);
}

void free_face_gpu(struct FaceCoordinatesGPU *face)
{
    if (face == NULL)
	return;
    free(face->coordinates);
    free(face);
}

struct DatasetGPU * create_dataset_gpu(const char *directory, const char *name)
{
    char * line = NULL;
    size_t len = 0;
    int num_images = 0;
    int i = 0;
    int w = 0, h = 0;
    struct DatasetGPU *dataset = NULL;
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
    fp = popen(command, "r"); // run the command twice, not optimal, and
                              // possible exploit if the contents of the
                              // dir is modified between the two shell commands

    dataset = (struct DatasetGPU *)malloc(sizeof(struct DatasetGPU));

    TEST_MALLOC(dataset);
    strcpy(dataset->name, name);
    dataset->path = "";
    dataset->num_original_images = num_images;

    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_original_images), num_images * w * h * sizeof(float))
    );

    dataset->original_names = (char *)malloc(num_images * sizeof(char *));
    TEST_MALLOC(dataset->original_names);

    struct ImageGPU *temp = NULL;

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, directory);
        strcat(image_name, "/");
        strcat(image_name, line);
        temp = load_image_gpu(image_name, 1);
        dataset->original_names[i] = strdup(line);
        TEST_MALLOC(dataset->original_names[i]);

        if (i == 0) {
            w = temp->w;
            h = temp->h;
        } else {
            if (w != temp->w || h != temp->h) {
                PRINT("WARN", "Images in directory have different width and/or height. Aborting\n");
                free_dataset_gpu(dataset);
                dataset = NULL;
                goto end;
            }
        }
        GPU_CHECKERROR(
        cudaMemcpy((void*)(dataset->d_original_images + i * w * h),
                   (void*)temp->data,
                   w * h * sizeof(float),
                   cudaMemcpyHostToDevice)
        );
        free_image_gpu(temp);
        i++;
        PRINT("DEBUG", "Loading file: %s\n", dataset->original_images[i-1]->filename);
    }

    dataset->w = w;
    dataset->h = h;
    dataset->num_eigenfaces = 0;
    dataset->num_faces = 0;
    dataset->num_new_faces = 0;
    dataset->d_average = NULL;
    dataset->d_eigenfaces = NULL;
    dataset->faces = NULL;

end:
    fclose(fp);
    if (line)
        free(line);
    free_image_gpu(temp);

    return dataset;
}

// No input checking -> Not secure at all
struct DatasetGPU * load_dataset_gpu(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        PRINT("WARN", "Unable to open file %s!\n", path);
        return NULL;
    }
    struct DatasetGPU *dataset = (struct DatasetGPU *)malloc(sizeof(struct DatasetGPU));
    TEST_MALLOC(dataset);
    dataset->path = path;
    dataset->num_eigenfaces = 0;
    dataset->num_original_images = 0;
    dataset->num_faces = 0;
    dataset->num_new_faces = 0;
    dataset->w = 0;
    dataset->h = 0;
    dataset->d_original_images = NULL;
    dataset->original_names = NULL;
    dataset->d_average = NULL;
    dataset->d_eigenfaces = NULL;
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

    float *temp = (float *)malloc(w * h * sizeof(float));
    TEST_MALLOC(temp);

    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_average), w * h * sizeof(float))
    );

    char c;

    for (int i = 0; i < 100; i++) {
        fread(&c, sizeof(char), 1, f);
        if (c == '\0')
            break;
    }
    fread(temp, w * h * sizeof(float), 1, f);
    GPU_CHECKERROR(
    cudaMemcpy((void*)(dataset->d_original_images + i * w * h),
               (void*)temp->data,
               w * h * sizeof(float),
               cudaMemcpyHostToDevice)
    );

    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_eigenfaces), num_eigenfaces * w * h * sizeof(float))
    );

    for (int i = 0; i < dataset->num_eigenfaces; i++) {
        for (int k = 0; k < 100; k++) {
            fread(&c, sizeof(char), 1, f);
            if (&c == '\0')
               break;
        }
        fread(temp, w * h * sizeof(float), 1, f);
        GPU_CHECKERROR(
        cudaMemcpy((void*)(dataset->d_eigenfaces + i * w * h),
                   (void*)temp->data,
                   w * h * sizeof(float),
                   cudaMemcpyHostToDevice)
        );
    }

    int current_size = 0;
    int num_allocated_faces = 50;
    dataset->faces = (struct FaceCoordinatesGPU **)malloc(num_allocated_faces * sizeof(struct FaceCoordinatesGPU *));
    TEST_MALLOC(dataset->faces);
    while (1) {
        size_t read = fread(&c, sizeof(char), 1, f);
        if (c != '\0' || read != 1)
            break;

        current_size++;
        if (current_size > num_allocated_faces) {
            num_allocated_faces *= 2;
            dataset->faces = (struct FaceCoordinatesGPU **)realloc(dataset->faces, num_allocated_faces * sizeof(struct FaceCoordinatesGPU *));
            TEST_MALLOC(dataset->faces);
        }

        dataset->faces[current_size - 1] = (struct FaceCoordinatesGPU *)malloc(sizeof(struct FaceCoordinatesGPU));
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
        dataset->faces = (struct FaceCoordinatesGPU **)realloc(dataset->faces, current_size * sizeof(struct FaceCoordinatesGPU *));
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
int save_dataset_to_disk_gpu(struct DatasetGPU *dataset, const char *path)
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
            struct FaceCoordinatesGPU *face = dataset->faces[i];
            fwrite(face->name, sizeof(char), strlen(face->name), f);
            fwrite("\0", sizeof(char), 1, f);
            for (int j = 0; j < face->num_eigenfaces; j++)
                fwrite(&(face->coordinates[j]), sizeof(float), 1, f);
        }
        dataset->num_new_faces = 0;
        fclose(f);
    } else {
        int w = dataset->w;
        int h = dataset->h;
        float *temp = (float *)malloc(w * h * sizeof(float));
        TEST_MALLOC(temp);
        FILE *f = fopen(path, "wb");
        if (f == NULL) {
            PRINT("WARNING", "Unable to create file %s!\n", path);
            return 1;
        }
        fwrite(dataset->name, sizeof(char), strlen(dataset->name), f);
        fwrite("\0", sizeof(char), 1, f);
        fwrite(&w, sizeof(int), 1, f);
        fwrite(&h, sizeof(int), 1, f);
        fwrite(&(dataset->num_eigenfaces), sizeof(int), 1, f);

        fwrite("Average", sizeof(char), strlen("Average"), f);
        fwrite("\0", sizeof(char), 1, f);
        GPU_CHECKERROR(
        cudaMemcpy((void*)temp,
                   (void*)dataset->d_average,
                   w * h * sizeof(float),
                   cudaMemcpyDeviceToHost)
        );
        fwrite(temp, w * h * sizeof(float), 1, f);

        char name[100] = "";

        for (int i = 0; i < dataset->num_eigenfaces; i++) {
            sprintf(name, "Eigen %d", i);
            fwrite(name, sizeof(char), strlen(name), f);
            fwrite("\0", sizeof(char), 1, f);
            GPU_CHECKERROR(
            cudaMemcpy((void*)temp,
                       (void*)dataset->d_eigenfaces + i * w * h,
                       w * h * sizeof(float),
                       cudaMemcpyDeviceToHost)
            );
            fwrite(temp, w * h * sizeof(float), 1, f);
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

void free_dataset_gpu(struct DatasetGPU *dataset)
{
    if (dataset == NULL)
	return;
    if (dataset->d_original_images) {
        GPU_CHECKERROR(cudaFree(dataset->d_original_images));
    }
    if (dataset->d_eigenfaces) {
        GPU_CHECKERROR(cudaFree(dataset->d_eigenfaces));
    }
    for (int i = 0; i < dataset->num_faces; i++)
	free_face_gpu(dataset->faces[i]);
    if (dataset->faces)
        free(dataset->faces);

    GPU_CHECKERROR(cudaFree(dataset->d_average));
    if (dataset->num_original_images > 0)
        for (int i = 0; i < dataset->num_original_images; i++)
            free(dataset->original_names[i]);
    if (dataset->original_names)
        free(dataset->original_names);
    free(dataset);
}

void save_image_to_disk_gpu(struct ImageGPU *image, const char *name)
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


void save_reconstructed_face_to_disk(struct DatasetGPU *dataset, struct FaceCoordinatesGPU *face, int num_eigenfaces)
{
    struct ImageGPU *image = (struct ImageGPU *)malloc(sizeof(struct ImageGPU));
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
    save_image_to_disk_gpu(image, image->filename);
    free_image_gpu(image);
}

// Expects dataset != NULL
// Returns the number of faces added
int add_faces_and_compute_coordinates(struct DatasetGPU *dataset, const char *path)
{
    char *line = NULL;
    size_t len = 0;
    int num_images = 0;
    int num_allocated = 0;
    int w = dataset->w;
    int h = dataset->h;
    int i = 0;
    struct ImageGPU **images = NULL;
    char command[200] = ""; // careful buffer overflow

    if (strstr(path, ".png")) {
        if(access(path, F_OK) == -1 ) {
            PRINT("WARN", "Cannot access file %s!\n", path);
            return 0;
        }
        struct ImageGPU *image = load_image(path, 1);
        compute_weighs_gpu(dataset, &image, 1, 1);
        dataset->num_new_faces++;
        free_image_gpu(image);
        return 1;
    }

    // Case path is a directory
    sprintf(command, "ls %s | grep png", path);

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
    num_allocated = num_images;
    images = (struct ImageGPU **)malloc(num_allocated * sizeof(struct ImageGPU *));
    TEST_MALLOC(images);

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, path);
        strcat(image_name, "/");
        strcat(image_name, line);
        images[i] = load_image(image_name, 1);
	strcpy(images[i]->filename, line); // possible buffer overflow
        if (w != images[i]->w || h != images[i]->h) {
                PRINT("WARN", "ImageGPUs in directory have different width and/or height. Aborting\n");
                num_images = 0;
                goto end;
        }
        i++;
        PRINT("DEBUG", "Loading file: %s\n", images[i-1]->filename);
    }
    compute_weighs_gpu(dataset, images, num_images, 1);
    dataset->num_new_faces += num_images;

end:
    fclose(fp);
    if (line)
        free(line);
    if (images) {
        for (int i = 0; i < num_allocated; i++)
            free_image_gpu(images[i]);
        free(images);
    }
    return num_images;
}

void identify_face_gpu(struct DatasetGPU *dataset, const char *path)
{
    char *answer;
    if (access(path, F_OK) == -1) {
        PRINT("WARN", "Cannot access file %s!\n", path);
        return;
    }
    struct ImageGPU *image = load_image_gpu(path, 1);
    struct FaceCoordinatesGPU **faces = compute_weighs_gpu(dataset, &image, 1, 0);
    struct FaceCoordinatesGPU *face = faces[0];
    struct FaceCoordinatesGPU *closest = get_closest_match_gpu(dataset, face);
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
        dataset->faces = (struct FaceCoordinatesGPU **)realloc(dataset->faces, (dataset->num_faces + 1) * sizeof(struct FaceCoordinatesGPU *));
        TEST_MALLOC(dataset->faces);
        dataset->faces[dataset->num_faces] = face;
        dataset->num_faces++;
        dataset->num_new_faces++;
    } else {
        free_face_gpu(face);
    }
    free(answer);
    free_image_gpu(image);
    free(faces);

}
