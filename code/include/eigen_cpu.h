#ifndef EIGEN_CPU_H
#define EIGEN_CPU_H

struct DatasetCPU * create_dataset_and_compute_all_cpu(const char *path, const char *name);
void normalize_cpu(float *array, int size);
struct ImageCPU * compute_average_cpu(struct DatasetCPU * dataset);
float dot_product_cpu(float *a, float *b, int size);
void jacobi_cpu(const float *a, const int n, float *v, float *e);
int comp_eigenvalues_cpu(const void *a, const void *b);
int compute_eigenfaces_cpu(struct DatasetCPU * dataset, int num_to_keep);
struct FaceCoordinatesCPU * get_closest_match_cpu(struct DatasetCPU *dataset, struct FaceCoordinatesCPU *face);
struct FaceCoordinatesCPU ** compute_weighs_cpu(struct DatasetCPU *dataset, struct ImageCPU **images, int k, int add_to_dataset);
#endif
