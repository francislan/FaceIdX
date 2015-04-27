#ifndef DATABASE_CPU_H
#define DATABASE_CPU_H

// Assumes the image is loaded and x and y are correct coordinates
#define GET_PIXEL(image, x, y, req_comp) \
	(image)->data[((y) * (image)->w + (x)) * (image)->comp + (req_comp)]

struct ImageCPU {
	float *data; //malloc
	int w;
	int h;
	int comp;
	int req_comp;
	char filename[100];
};

struct FaceCoordinatesCPU {
	char name[100];
	int num_eigenfaces;
	float *coordinates; //malloc
};

struct DatasetCPU {
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
	struct ImageCPU **original_images; //malloc
	struct ImageCPU *average; //malloc
	struct ImageCPU **eigenfaces; //malloc
	struct FaceCoordinatesCPU **faces; //malloc
};

void free_image_cpu(struct ImageCPU *image);
void free_face_cpu(struct FaceCoordinatesCPU *face);
struct DatasetCPU * create_dataset_cpu(const char *directory, const char *name);
struct DatasetCPU * load_dataset_cpu(const char *dataset_path);
int save_dataset_to_disk_cpu(struct DatasetCPU *dataset, const char *path);
void free_dataset_cpu(struct DatasetCPU *dataset);
void save_reconstructed_face_to_disk_cpu(struct DatasetCPU *dataset, struct FaceCoordinatesCPU *face, int num_eigenfaces);
int add_faces_and_compute_coordinates_cpu(struct DatasetCPU *dataset, const char *path);
void identify_face_cpu(struct DatasetCPU *dataset, const char *path);

#endif
