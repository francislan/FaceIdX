#ifndef DATABASE_H
#define DATABASE_H

// Assumes the image is loaded and x and y are correct coordinates
#define GET_PIXEL(image, x, y, req_comp) \
	(image)->data[((y) * (image)->w + (x)) * (image)->comp + (req_comp)]

struct Image {
	float *data; //malloc
	int w;
	int h;
	int comp;
	int req_comp;
	char filename[100];
};

struct FaceCoordinates {
	char name[100];
	int num_eigenfaces;
	float *coordinates; //malloc
};

struct Dataset {
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
	struct Image **original_images; //malloc
	struct Image *average; //malloc
	struct Image **eigenfaces; //malloc
	struct FaceCoordinates **faces; //malloc
	float *d_original_images; // no need to realloc
	float *d_average;
	float *d_eigenfaces;
	float **d_faces; // number of faces can be increased during exectution
};

struct Image * load_image(const char *filename, int req_comp);
void free_image(struct Image *image);
void free_face(struct FaceCoordinates *face);
struct Dataset * create_dataset(const char *directory, const char *name);
struct Dataset * load_dataset(const char *dataset_path);
int save_dataset_to_disk(struct Dataset *dataset, const char *path);
void free_dataset(struct Dataset *dataset);
void save_image_to_disk(struct Image *image, const char *name);
void save_reconstructed_face_to_disk(struct Dataset *dataset, struct FaceCoordinates *face, int num_eigenfaces);
int add_faces_and_compute_coordinates(struct Dataset *dataset, const char *path);
void identify_face(struct Dataset *dataset, const char *path);
void get_user_string(char **s);

#endif
