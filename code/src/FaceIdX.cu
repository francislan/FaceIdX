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

    struct Dataset *dataset = NULL;
    int action = 0;
    char name[100] = "";
    // Do not forget to free them
    char *path = NULL;
    char *dataset_name = NULL;
    int tmp;
    do {
        display_menu(dataset);
        action = get_user_choice();

        switch(action) {
        case 1:
            printf("\nEnter path to a repo containing images: ");
            get_user_string(&path);
            printf("\nEnter a name for the database: ");
            get_user_string(&dataset_name);
            if (dataset != NULL)
                free_dataset(dataset);
            dataset = create_dataset_and_compute_all(path, dataset_name);
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

        case 4:
            if (dataset == NULL) {
                PRINT("WARN", "No database is currently loaded!\n");
                break;
            }
            printf("Enter path to a repo containing new face(s) or path to a single face: ");
            get_user_string(&path);
            printf("\nAdding face(s)...\n");

            tmp = add_faces_and_compute_coordinates(dataset, path);
            if (tmp)
                printf("Adding face(s)... Done! (%d faces added)", tmp);
            break;

        case 5:
            if (dataset == NULL) {
                PRINT("WARN", "No database is currently loaded!\n");
                break;
            }
            printf("Enter path to a face to identify: ");
            get_user_string(&path);
            printf("\nIdentifying face...\n");
            identify_face(dataset, path);
            printf("\nIdentifying face... Done!");
            break;

        case 6:
            if (dataset == NULL) {
                PRINT("WARN", "No database is currently loaded!\n");
                break;
            }
            printf("\nSaving eigenfaces to ./eigen ...");
            for (int i = 0; i < dataset->num_eigenfaces; i++) {
                sprintf(name, "eigen/Eigenface %d.png", i);
                save_image_to_disk(dataset->eigenfaces[i], name);
            }
            printf("Done!");
            break;

        case 7:
            if (dataset == NULL) {
                PRINT("WARN", "No database is currently loaded!\n");
                break;
            }
            printf("Reconstructing images to ./reconstructed ...");
            for (int i = 0; i < dataset->num_faces; i++)
                save_reconstructed_face_to_disk(dataset, dataset->faces[i], dataset->num_eigenfaces);
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
        printf(KNRM "Number of new faces: ");
        printf(KWHT "%d\n", dataset->num_new_faces);
    }

    printf(KNRM "\n\n===== MENU =====\n\n");
    printf("1. Create database\n");
    printf("2. Load database\n");
    printf("3. Save database to disk\n");
    printf("4. Add face(s) to database\n");
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
