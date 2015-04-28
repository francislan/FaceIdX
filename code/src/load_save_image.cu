#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "database_gpu.h"
#include "database_cpu.h"
#include "misc.h"
#include "load_save_image.h"
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

void save_image_to_disk_gpu(float *d_image, int w, int h, const char *name)
{
    float *image_data_float = (float *)malloc(w * h * sizeof(float));
    TEST_MALLOC(image_data_float);
    unsigned char *image_data = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    TEST_MALLOC(image_data);
    GPU_CHECKERROR(
    cudaMemcpy((void*)image_data_float,
               (void*)d_image,
               w * h * sizeof(float),
               cudaMemcpyDeviceToHost)
    );
    cudaDeviceSynchronize();

    // useless, already done?
    float min = image_data_float[0];
    float max = image_data_float[0];
    for (int j = 1; j < w * h; j++) {
        float current = image_data_float[j];
        if (current > max) {
            max = current;
        } else if (current < min) {
            min = current;
        }
    }
    // bad conversion from float to unsigned char
    for (int j = 0; j < w * h; j++)
        image_data[j] = image_data_float > 0 ?
            (unsigned char)((image_data_float[j] / max) * 127 + 128) :
            (unsigned char)(128 - (image_data_float[j] / min) * 128);
    stbi_write_png(name, w, h, 1, image_data, 0);
    free(image_data_float);
    free(image_data);
}

// User has to call free_image
struct ImageCPU * load_image_cpu(const char *filename, int req_comp)
{
    struct ImageCPU *image = (struct ImageCPU *)malloc(sizeof(struct ImageCPU));
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

void save_image_to_disk_cpu(struct ImageCPU *image, const char *name)
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
