#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>

#include "misc.h"
#include "eigen.h"
#include "database.h"

void display_menu(struct Dataset *dataset);
int get_user_choice();
void get_user_string(char **s);

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

    //struct Dataset *dataset = create_dataset("../../Data/nottingham/normalized", "Set 1");
    struct Dataset *dataset = create_dataset("../../Data/yale/normalized", "Set 2");
    if (dataset == NULL) {
        PRINT("BUG","Dataset creation failed\n");
        return EXIT_FAILURE;
    }
    PRINT("", "Dataset name: %s\n", dataset->name);
    PRINT("", "Dataset path: %s\n", dataset->path);
    PRINT("", "Dataset num_original_images: %d\n", dataset->num_original_images);

    int action = 0;
    // Do not forget to free them
    char *path = NULL;
    char *dataset_name = NULL;
    do {
        display_menu(dataset);
        action = get_user_choice();

        switch(action) {
        case 1:
            printf("\nEnter path to a repo containing images: ");
            get_user_string(&path);
            printf("\nEnter a name for the database: ");
            get_user_string(&dataset_name);
            printf("\nCreating database...\n\n");
            if (dataset != NULL)
                free_dataset(dataset);
            dataset = create_dataset(path, dataset_name);
            if (dataset)
                printf("Done!");
            break;

        case 2:
            printf("\nEnter path to a .dat file: ");
            get_user_string(&path);
            printf("\nLoading database...\n\n");
            if (dataset != NULL)
                free_dataset(dataset);
            dataset = load_dataset(path);
            if (dataset)
                printf("Done!");
            break;

        case 3:
            if (dataset == NULL) {
                PRINT("WARN", "No database is currently loaded!\n");
                break;
            }
            printf("Enter path to a repo in which %s.dat will be saved: ", dataset->name);
            get_user_string(&path);
            printf("\nSaving database...\n\n");
            path = (char *)realloc(path, (strlen(path) + strlen(dataset->name) + 6) * sizeof(char));
            TEST_MALLOC(path);
            strcat(path, "/");
            strcat(path, dataset->name);
            strcat(path, ".dat");
            save_dataset_to_disk(dataset, path);
            printf("Done!");
            break;

        case 8:
            printf("Good bye!\n");
            break;
        default:
            break;
        }
        getchar();
    } while (action != 8);
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
    //compute_eigenfaces_cpu(dataset, 50);
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
        save_reconstructed_face_to_disk(dataset, dataset->faces[i], dataset->num_eigenfaces);
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

    // Test loading dataset
    struct Dataset *dataset2 = load_dataset("dataset1.dat");
    if (dataset2 == NULL) {
        PRINT("BUG","Dataset loading failed\n");
        return EXIT_FAILURE;
    }
    PRINT("", "Dataset name: %s\n", dataset2->name);
    PRINT("", "Dataset path: %s\n", dataset2->path);
    PRINT("", "Dataset num faces: %d\n", dataset2->num_faces);
    PRINT("", "Dataset num eigenfaces: %d\n", dataset2->num_eigenfaces);
    PRINT("", "Dataset w: %d\n", dataset2->w);
    PRINT("", "Dataset h: %d\n", dataset2->h);

    free_dataset(dataset2);


    GPU_CHECKERROR(cudaEventDestroy(start_cpu));
    GPU_CHECKERROR(cudaEventDestroy(end_cpu));
    GPU_CHECKERROR(cudaEventDestroy(start_gpu));
    GPU_CHECKERROR(cudaEventDestroy(end_gpu));
    return EXIT_SUCCESS;
}


void display_menu(struct Dataset *dataset)
{
    system("clear");
    printf("////////////////////////////////////////////////////////////////////////////////\n");
    printf("///                                 FaceIdX                                  ///\n");
    printf("////////////////////////////////////////////////////////////////////////////////\n\n\n");

    printf("Current database: ");
    if (dataset == NULL) {
        printf(KRED "None");
    } else {
        printf(KGRN "%s\n\n", dataset->name);
        printf(KNRM "Number of original images: ");
        printf(KWHT "%d\n", dataset->num_original_images);
        printf(KNRM "Number of eigenfaces: ");
        printf(KWHT "%d\n", dataset->num_eigenfaces);
        printf(KNRM "Number of faces: ");
        printf(KWHT "%d\n", dataset->num_faces);
    }

    printf(KNRM "\n\n===== MENU =====\n\n");
    printf("1. Create database\n");
    printf("2. Load database\n");
    printf("3. Save database to disk\n");
    printf("4. Add face to database\n");
    printf("5. Identify face\n");
    printf("6. Export eigenfaces\n");
    printf("7. Reconstruct faces\n");
    printf("8. Exit\n");

    printf("\nYour choice: ");
}

int get_user_choice()
{
    size_t len = 0;
    int char_read;
    char *user_command;
    char_read = getline(&user_command, &len, stdin);
    if (char_read == -1) {
        PRINT("BUG", "Unexpected error.");
        return 0;
    }
    user_command[char_read - 1] = '\0';

    char *p;
    int tmp = strtol(user_command, &p, 10);

    if (*p != '\0' || (tmp == 0 && errno != 0)) {
        PRINT("WARN", "Invalid choice!\n");
        return 0;
    } else if (tmp < 1 || tmp > 8) {
            PRINT("WARN", "Invalid choice!\n");
            tmp = 0;
    }

    free(user_command);
    return tmp;
}

void get_user_string(char **s)
{
    size_t len = 0;
    int char_read;
    char_read = getline(s, &len, stdin);
    if (char_read == -1) {
        PRINT("BUG", "Unexpected error.");
        return;
    }
    (*s)[char_read - 1] = '\0';
}
