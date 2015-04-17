#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>

#include "misc.h"
#include "eigen.h"
#include "database.h"

int main(int argc, char **argv)
{
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        PRINT("BUG", "Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    cudaEvent_t start_cpu, end_cpu, start_gpu, end_gpu;
    float time_for_cpu, time_for_gpu;
    FILE *f = fopen("timer_log.txt", "w");
    if(f == NULL) {
        PRINT("BUG", "Error opening file!\n");
        return EXIT_FAILURE;
    }
    GPU_CHECKERROR(cudaEventCreate(&start_cpu));
    GPU_CHECKERROR(cudaEventCreate(&end_cpu));
    GPU_CHECKERROR(cudaEventCreate(&start_gpu));
    GPU_CHECKERROR(cudaEventCreate(&end_gpu));

    struct Image *image = load_image("../../Data/nottingham/normalized/f005a.png", 1);
    if (image->data == NULL) {
        PRINT("WARN", "file could not be loaded.\n");
    } else {
        PRINT("", "Image width: %d, height: %d, comp: %d\n", image->w, image->h, image->comp);
        PRINT("", "grey: %f\n", GET_PIXEL(image, 0, 0, 0));
        PRINT("", "grey: %f\n", GET_PIXEL(image, 156, 15, 0));
    }
    free_image(image);

    struct Dataset *dataset = create_dataset("../../Data/nottingham/normalized", "./dataset.dat", "Set 1");
    if (dataset == NULL) {
        PRINT("BUG","Dataset creation failed\n");
        return EXIT_FAILURE;
    }
    PRINT("", "Dataset name: %s\n", dataset->name);
    PRINT("", "Dataset path: %s\n", dataset->path);
    PRINT("", "Dataset num_original_images: %d\n", dataset->num_original_images);
/*
    for (int i = 0; i < dataset->num_original_images; i++) {
        PRINT("", "\tImage %d: %s\n", i + 1, dataset->original_images[i]->filename);
        PRINT("", "grey 0, 0: %d\n", GET_PIXEL(dataset->original_images[i], 0, 0, 0));
        PRINT("", "grey 156, 15: %d\n", GET_PIXEL(dataset->original_images[i], 156, 15, 0));
        X
    }*/


    GPU_CHECKERROR(cudaEventRecord(start_cpu, 0));
    struct Image *average = compute_average_cpu(dataset);
    GPU_CHECKERROR(cudaEventRecord(end_cpu, 0));
    GPU_CHECKERROR(cudaEventSynchronize(end_cpu));
    GPU_CHECKERROR(cudaEventElapsedTime(&time_for_cpu, start_cpu, end_cpu));
    fprintf(f, "Time taken for computing average face on cpu: %3.1f ms\n", time_for_cpu);
    if (average == NULL) {
        PRINT("BUG","average computation failed\n");
        return EXIT_FAILURE;
    }
    //PRINT("", "grey 0, 0: %d\n", GET_PIXEL(average, 0, 0, 0));
    //PRINT("", "grey 156, 15: %d\n", GET_PIXEL(average, 156, 15, 0));

    save_image_to_disk(average, "average_cpu.png");

    // Eigenfaces
    PRINT("INFO", "Start eigenfaces computation\n");
    compute_eigenfaces_cpu(dataset, dataset->num_original_images);
    PRINT("INFO", "End eigenfaces computation\n");
    char name[100]= "";
    for (int i = 0; i < dataset->num_eigenfaces; i++) {
	sprintf(name, "eigen/Eigenface %d.png", i);
    	save_image_to_disk(dataset->eigenfaces[i], name);
    }
    PRINT("INFO", "Start coordinates computation\n");
    compute_weighs_cpu(dataset);
    PRINT("INFO", "End coordinates computation\n");
    PRINT("INFO", "Start reconstruction\n");
    for (int i = 0; i < dataset->num_original_images; i++)
        save_reconstructed_face_to_disk(dataset, dataset->faces[i], dataset->num_original_images);
    PRINT("INFO", "End reconstruction\n");
    for (int i = 0; i < dataset->num_faces; i++)
        PRINT("INFO", "The Closest match of %s is %s.\n", dataset->faces[i]->name, get_closest_match_cpu(dataset, dataset->faces[i])->name);

    save_dataset_to_disk(dataset, "dataset1.dat");



    GPU_CHECKERROR(cudaEventRecord(start_gpu, 0));
    struct Image *average_gpu = compute_average_gpu(dataset);

    GPU_CHECKERROR(cudaEventRecord(end_gpu, 0));
    GPU_CHECKERROR(cudaEventSynchronize(end_gpu));
    GPU_CHECKERROR(cudaEventElapsedTime(&time_for_gpu, start_gpu, end_gpu));
    fprintf(f, "Time taken for computing average face on gpu: %3.1f ms\n", time_for_gpu);
    // not working, has to find another way to test average
    if (average_gpu == NULL) {
        PRINT("BUG","average computation failed\n");
        return EXIT_FAILURE;
    }
    //PRINT("", "grey 0, 0: %f\n", GET_PIXEL(average_gpu, 0, 0, 0));
    //PRINT("", "grey 156, 15: %f\n", GET_PIXEL(average_gpu, 156, 15, 0));

    save_image_to_disk(average_gpu, "average_gpu.png");

    fclose(f);
    free_dataset(dataset);
    GPU_CHECKERROR(cudaEventDestroy(start_cpu));
    GPU_CHECKERROR(cudaEventDestroy(end_cpu));
    GPU_CHECKERROR(cudaEventDestroy(start_gpu));
    GPU_CHECKERROR(cudaEventDestroy(end_gpu));
    return EXIT_SUCCESS;
}
