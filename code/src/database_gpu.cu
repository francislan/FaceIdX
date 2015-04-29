#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "database_gpu.h"
#include "misc.h"
#include "eigen_gpu.h"
#include "load_save_image.h"


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
    struct ImageGPU *temp = NULL;
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

    dataset->original_names = (char **)malloc(num_images * sizeof(char *));
    TEST_MALLOC(dataset->original_names);

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, directory);
        strcat(image_name, "/");
        strcat(image_name, line);
        temp = load_image_gpu(image_name, 1);
        char *temp_name = strdup(line);
        dataset->original_names[i] = temp_name;

        if (i == 0) {
            w = temp->w;
            h = temp->h;
            GPU_CHECKERROR(
            cudaMalloc((void **)&(dataset->d_original_images), num_images * w * h * sizeof(float))
            );
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
        temp = NULL;
        i++;
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
    cudaMemcpy((void*)dataset->d_average,
               (void*)temp,
               w * h * sizeof(float),
               cudaMemcpyHostToDevice)
    );

    GPU_CHECKERROR(
    cudaMalloc((void **)&(dataset->d_eigenfaces), dataset->num_eigenfaces * w * h * sizeof(float))
    );

    for (int i = 0; i < dataset->num_eigenfaces; i++) {
        for (int k = 0; k < 100; k++) {
            fread(&c, sizeof(char), 1, f);
            if (c == '\0')
               break;
        }
        fread(temp, w * h * sizeof(float), 1, f);
        GPU_CHECKERROR(
        cudaMemcpy((void*)(dataset->d_eigenfaces + i * w * h),
                   (void*)temp,
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
    free(temp);
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
                       (void*)(dataset->d_eigenfaces + i * w * h),
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
        free(temp);
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

__global__
void reconstruct_face_gpu_kernel(float *d_output, float *d_average, float *d_coordinates, float *d_eigenfaces, int num_eigens, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float pixel = d_average[i];
        for (int k = 0; k < num_eigens; k++)
            pixel += d_coordinates[k] * d_eigenfaces[k * size + i];
        d_output[i] = pixel;
    }
}

// Use one block (so that min and max are shared)
__global__
void normalize_image_to_save_gpu_kernel(float *d_image, int size)
{
    extern __shared__ float s_min_max[];
    int i = threadIdx.x;

    int max = d_image[0];
    int min = d_image[0];

    while (i < size) {
        float current = d_image[i];
        if (current > max)
            max = current;
        else if (current < min)
            min = current;
        i += blockDim.x;
    }
    i = threadIdx.x;
    s_min_max[i] = min;
    s_min_max[i + blockDim.x] = max;
    __syncthreads();

    // Reduction
    for (int stride2 = blockDim.x / 2; stride2 > 0; stride2 /= 2) {
        if (i < stride2) {
            if (s_min_max[i + stride2] < s_min_max[i])
                s_min_max[i] = s_min_max[i + stride2];
            if (s_min_max[blockDim.x + i + stride2] > s_min_max[blockDim.x + i])
                s_min_max[blockDim.x + i] = s_min_max[blockDim.x + i + stride2];
        }
        __syncthreads();
    }
    min = s_min_max[0];
    max = s_min_max[blockDim.x];
    for (i = threadIdx.x; i < size; i += blockDim.x)
        d_image[i] = (d_image[i] - min) / (max - min) * 255;
}

void save_reconstructed_face_to_disk_gpu(struct DatasetGPU *dataset, struct FaceCoordinatesGPU *face, int num_eigenfaces)
{
    int w = dataset->w;
    int h = dataset->h;
    char name[100];

    int n = num_eigenfaces > face->num_eigenfaces ? face->num_eigenfaces : num_eigenfaces;
    float *d_temp;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_temp, w * h * sizeof(float))
    );
    float *d_coordinates;
    GPU_CHECKERROR(
    cudaMalloc((void **)&d_coordinates, n * sizeof(float))
    );
    GPU_CHECKERROR(
    cudaMemcpy((void*)d_coordinates,
               (void*)face->coordinates,
               n * sizeof(float),
               cudaMemcpyHostToDevice)
    );

    int num_blocks = ceil(w * h / 1024.0);
    dim3 dimOfGrid(num_blocks, 1, 1);
    dim3 dimOfBlock(1024, 1, 1);
    if (num_blocks == 1) {
        dimOfBlock.x = 32;
        while (dimOfBlock.x < w * h)
            dimOfBlock.x *= 2;
    }
    reconstruct_face_gpu_kernel<<<dimOfGrid, dimOfBlock>>>(d_temp, dataset->d_average, d_coordinates, dataset->d_eigenfaces, n, w * h);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    dimOfGrid.x = 1;
    int size_shared_mem = 2 * dimOfBlock.x * sizeof(float);
    normalize_image_to_save_gpu_kernel<<<dimOfGrid, dimOfBlock, size_shared_mem>>>(d_temp, w * h);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        PRINT("BUG", "kernel launch failed with error \"%s\"\n",
               cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    sprintf(name, "reconstructed/%s_with_%d.png", face->name, n); //buffer overflow
    save_image_to_disk_gpu(d_temp, w, h, name);
    GPU_CHECKERROR(cudaFree(d_temp));
    GPU_CHECKERROR(cudaFree(d_coordinates));
}

// Expects dataset != NULL
// Returns the number of faces added
int add_faces_and_compute_coordinates_gpu(struct DatasetGPU *dataset, const char *path)
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
    FILE *fp = NULL;

    if (strstr(path, ".png")) {
        if(access(path, F_OK) == -1 ) {
            PRINT("WARN", "Cannot access file %s!\n", path);
            return 0;
        }
        struct ImageGPU *image = load_image_gpu(path, 1);
        if (w != image->w || h != image->h) {
            PRINT("WARN", "Images in directory have different width and/or height. Aborting\n");
            num_images = 0;
            goto end;
        }
        substract_average_gpu(&image, dataset->d_average, 1, w * h);
        compute_weighs_gpu(dataset, &image, 0, 1, 1);
        dataset->num_new_faces++;
        free_image_gpu(image);
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
    images = (struct ImageGPU **)malloc(num_allocated * sizeof(struct ImageGPU *));
    TEST_MALLOC(images);

    while (getline(&line, &len, fp) != -1) {
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1 ] = '\0';
        char image_name[100] = "";
        strcpy(image_name, path);
        strcat(image_name, "/");
        strcat(image_name, line);
        images[i] = load_image_gpu(image_name, 1);
        strcpy(images[i]->filename, line); // possible buffer overflow
        if (w != images[i]->w || h != images[i]->h) {
            PRINT("WARN", "Images in directory have different width and/or height. Aborting\n");
            num_images = 0;
            goto end;
        }
        i++;
    }
    substract_average_gpu(images, dataset->d_average, num_images, w * h);
    compute_weighs_gpu(dataset, images, 0, num_images, 1);
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
    Timer timer;
    INITIALIZE_TIMER(timer);

    if (access(path, F_OK) == -1) {
        PRINT("WARN", "Cannot access file %s!\n", path);
        return;
    }
    START_TIMER(timer);
    struct ImageGPU *image = load_image_gpu(path, 1);
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_gpu: Time for loading image: %fms\n", timer.time);

    START_TIMER(timer);
    substract_average_gpu(&image, dataset->d_average, 1, image->w * image->h);
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_gpu: Time for substracting average: %fms\n", timer.time);


    START_TIMER(timer);
    struct FaceCoordinatesGPU **faces = compute_weighs_gpu(dataset, &image, 0, 1, 0);
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_gpu: Time for computing coordinates: %fms\n", timer.time);

    struct FaceCoordinatesGPU *face = faces[0];
    START_TIMER(timer);
    struct FaceCoordinatesGPU *closest = get_closest_match_gpu(dataset, face);
    STOP_TIMER(timer);
    PRINT("DEBUG", "identify_face_gpu: Time for getting closest match: %fms\n", timer.time);

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
