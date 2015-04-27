#ifndef DATABASE_GPU_H
#define DATABASE_GPU_H

// Assumes the image is loaded and x and y are correct coordinates
#define GET_PIXEL(image, x, y, req_comp) \
	(image)->data[((y) * (image)->w + (x)) * (image)->comp + (req_comp)]

struct ImageGPU {
	float *data; //malloc
	int w;
	int h;
	int comp;
	int req_comp;
	char filename[100];
};

struct FaceCoordinatesGPU {
	char name[100];
	int num_eigenfaces;
	float *coordinates; //malloc
};

struct DatasetGPU {
	char name[100];
	const char *path;
	int num_eigenfaces;
	int num_original_images;
	int num_faces;
	// new faces added to the loaded dataset
	// when saving a dataset to disk, if the file already exists, only add
	// the last num_new_faces at the end of the file.
	int num_new_faces;
	int w;
	int h;
	struct FaceCoordinatesGPU **faces; //malloc
	char **original_names; //names of original images
	float *d_original_images; // no need to realloc
	float *d_average;
	float *d_eigenfaces;
};

void free_image_gpu(struct ImageGPU *image);
void free_face_gpu(struct FaceCoordinatesGPU *face);
struct DatasetGPU * create_dataset_gpu(const char *directory, const char *name);
struct DatasetGPU * load_dataset_gpu(const char *dataset_path);
int save_dataset_to_disk_gpu(struct DatasetGPU *dataset, const char *path);
void free_dataset_gpu(struct DatasetGPU *dataset);
__global__ void reconstruct_face_gpu_kernel(float *d_output, float *d_average, float *d_coordinates, float *d_eigenfaces, int num_eigens, int size);

// Not working for blockDim.x not a power of 2
__global__ void normalize_image_to_save_gpu_kernel(float *d_image, int size, int stride);
// TODO Try and save all faces using streams
void save_reconstructed_face_to_disk_gpu(struct DatasetGPU *dataset, struct FaceCoordinatesGPU *face, int num_eigenfaces);
int add_faces_and_compute_coordinates_gpu(struct DatasetGPU *dataset, const char *path);
void identify_face_gpu(struct DatasetGPU *dataset, const char *path);

#endif
