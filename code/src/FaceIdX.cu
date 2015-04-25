#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>

#include "misc.h"
#include "eigen_cpu.h"
#include "database_cpu.h"
//#include "eigen_gpu.h"
//#include "database_gpu.h"

void display_menu_cpu(struct DatasetCPU *dataset_cpu);
//void display_menu_gpu(struct DatasetGPU *dataset_gpu);
int get_user_choice();
void get_user_string(char **s);

int main(int argc, char **argv)
{

    struct DatasetCPU *dataset_cpu = NULL;
    //struct DatasetGPU *dataset_gpu = NULL;
    struct DatasetCPU *dataset_gpu = NULL;
    int use_gpu = 0;
    if (argc == 2 && !strcmp(argv[1], "-cpu"))
	use_gpu = 0;
    int action = 0;
    char name[100] = "";
    // Do not forget to free them
    char *path = NULL;
    char *dataset_name = NULL;
    int tmp;
    do {
 //       if (use_gpu)
//            display_menu_gpu(dataset_gpu);
 //       else
            display_menu_cpu(dataset_cpu);
        action = get_user_choice();

        switch(action) {
        case 1:
            printf("\nEnter path to a repo containing images: ");
            get_user_string(&path);
            printf("\nEnter a name for the database: ");
            get_user_string(&dataset_name);
            if (use_gpu) {
                if (dataset_gpu != NULL)
            //        free_dataset_gpu(dataset_gpu);
          //      dataset_gpu = create_dataset_and_compute_all_gpu(path, dataset_name);
                if (dataset_gpu)
                    printf("Done!");
            } else {
                if (dataset_cpu != NULL)
                    free_dataset_cpu(dataset_cpu);
                dataset_cpu = create_dataset_and_compute_all_cpu(path, dataset_name);
                if (dataset_cpu)
                    printf("Done!");
            }
            break;

        case 2:
            printf("\nEnter path to a .dat file: ");
            get_user_string(&path);
            printf("\nLoading database...\n\n");
            if (use_gpu) {
                if (dataset_gpu != NULL)
          //          free_dataset_gpu(dataset_gpu);
           //     dataset_gpu = load_dataset_gpu(path);
                if (dataset_gpu)
                    printf("Done!");
            } else {
                if (dataset_cpu != NULL)
                    free_dataset_cpu(dataset_cpu);
                dataset_cpu = load_dataset_cpu(path);
                if (dataset_cpu)
                    printf("Done!");
            }
            break;

        case 3:
            if (use_gpu) {
                if (dataset_gpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
                printf("Enter path to a repo in which %s.dat will be saved: ", dataset_gpu->name);
            } else {
                if (dataset_cpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
                printf("Enter path to a repo in which %s.dat will be saved: ", dataset_cpu->name);
            }
            get_user_string(&path);
            printf("\nSaving database...\n\n");
            if (use_gpu) {
                path = (char *)realloc(path, (strlen(path) + strlen(dataset_gpu->name) + 6) * sizeof(char));
                TEST_MALLOC(path);
                strcat(path, "/");
                strcat(path, dataset_gpu->name);
                strcat(path, ".dat");
      //          save_dataset_to_disk_gpu(dataset_gpu, path);
                printf("Done!");
            } else {
                path = (char *)realloc(path, (strlen(path) + strlen(dataset_cpu->name) + 6) * sizeof(char));
                TEST_MALLOC(path);
                strcat(path, "/");
                strcat(path, dataset_cpu->name);
                strcat(path, ".dat");
                save_dataset_to_disk_cpu(dataset_cpu, path);
                printf("Done!");
            }
            break;

        case 4:
            if (use_gpu) {
                if (dataset_gpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            } else {
                if (dataset_cpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            }
            printf("Enter path to a repo containing new face(s) or path to a single face: ");
            get_user_string(&path);
            printf("\nAdding face(s)...\n");

         //   if (use_gpu)
       //         tmp = add_faces_and_compute_coordinates_gpu(dataset_gpu, path);
        //    else
                tmp = add_faces_and_compute_coordinates_cpu(dataset_cpu, path);

            if (tmp)
                printf("Adding face(s)... Done! (%d faces added)", tmp);
            break;

        case 5:
            if (use_gpu) {
                if (dataset_gpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            } else {
                if (dataset_cpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            }
            printf("Enter path to a face to identify: ");
            get_user_string(&path);
            printf("\nIdentifying face...\n");
  //          if (use_gpu)
    //            identify_face_gpu(dataset_gpu, path);
      //      else
                identify_face_cpu(dataset_cpu, path);
            printf("\nIdentifying face... Done!");
            break;

        case 6:
            if (use_gpu) {
                if (dataset_gpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            } else {
                if (dataset_cpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            }
            printf("\nSaving eigenfaces to ./eigen ...");
            if (use_gpu) {
                for (int i = 0; i < dataset_gpu->num_eigenfaces; i++) {
                    sprintf(name, "eigen/Eigenface %d.png", i);
      //              save_image_to_disk_gpu(dataset_gpu->eigenfaces[i], name);
                }
            } else {
                for (int i = 0; i < dataset_cpu->num_eigenfaces; i++) {
                    sprintf(name, "eigen/Eigenface %d.png", i);
                    save_image_to_disk_cpu(dataset_cpu->eigenfaces[i], name);
                }
            }
            printf("Done!");
            break;

        case 7:
            if (use_gpu) {
                if (dataset_gpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            } else {
                if (dataset_cpu == NULL) {
                    PRINT("WARN", "No database is currently loaded!\n");
                    break;
                }
            }
            printf("Reconstructing images to ./reconstructed ...");
    //        if (use_gpu) {
     //           for (int i = 0; i < dataset_gpu->num_faces; i++)
      //              save_reconstructed_face_to_disk_gpu(dataset_gpu, dataset_gpu->faces[i], dataset_gpu->num_eigenfaces);
     //       } else {
                for (int i = 0; i < dataset_cpu->num_faces; i++)
                    save_reconstructed_face_to_disk_cpu(dataset_cpu, dataset_cpu->faces[i], dataset_cpu->num_eigenfaces);
     //       }
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


void display_menu_cpu(struct DatasetCPU *dataset_cpu)
{
    system("clear");
    printf("////////////////////////////////////////////////////////////////////////////////\n");
    printf("///                                 FaceIdX                                  ///\n");
    printf("////////////////////////////////////////////////////////////////////////////////\n\n\n");

    printf("Current database: ");
    if (dataset_cpu == NULL) {
        printf(KRED "None");
    } else {
        printf(KGRN "%s\n\n", dataset_cpu->name);
        printf(KNRM "Number of original images: ");
        printf(KWHT "%d\n", dataset_cpu->num_original_images);
        printf(KNRM "Number of eigenfaces: ");
        printf(KWHT "%d\n", dataset_cpu->num_eigenfaces);
        printf(KNRM "Number of faces: ");
        printf(KWHT "%d\n", dataset_cpu->num_faces);
        printf(KNRM "Number of new faces: ");
        printf(KWHT "%d\n", dataset_cpu->num_new_faces);
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
/*
void display_menu_gpu(struct DatasetGPU *dataset_gpu)
{
    system("clear");
    printf("////////////////////////////////////////////////////////////////////////////////\n");
    printf("///                                 FaceIdX                                  ///\n");
    printf("////////////////////////////////////////////////////////////////////////////////\n\n\n");

    printf("Current database: ");
    if (dataset_gpu == NULL) {
        printf(KRED "None");
    } else {
        printf(KGRN "%s\n\n", dataset_gpu->name);
        printf(KNRM "Number of original images: ");
        printf(KWHT "%d\n", dataset_gpu->num_original_images);
        printf(KNRM "Number of eigenfaces: ");
        printf(KWHT "%d\n", dataset_gpu->num_eigenfaces);
        printf(KNRM "Number of faces: ");
        printf(KWHT "%d\n", dataset_gpu->num_faces);
        printf(KNRM "Number of new faces: ");
        printf(KWHT "%d\n", dataset_gpu->num_new_faces);
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
*/
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
