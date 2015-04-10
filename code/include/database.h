#ifndef DATABASE_H
#define DATABASE_H

// Assumes the image is loaded and x and y are correct coordinates
#define GET_PIXEL(image, x, y, comp) \
	(image)->data[((y) * (image)->w + (x)) * (image)->(comp) + (comp)]

struct Image {
	unsigned char *data;
	int w;
	int h;
	int comp;
	int req_comp;
	char filename[100];
};

struct FaceCoordinates {
	char *name;
	int num_eigenfaces;
	float *coordinates; //malloc
};

struct Dataset {
	const char *name;
	const char *path;
	int num_eigenfaces;
	int num_original_images;
	int num_faces;
	int w;
	int h;
	struct Image **original_images; //malloc
	struct Image *average;
	struct Image **eigenfaces; //malloc
	struct FaceCoordinates **faces; //malloc
};

struct Image * load_image(const char *filename, int req_comp);
void free_image(struct Image *image);
struct Dataset * create_dataset(const char *directory, const char *dataset_path, const char *name);
void free_dataset(struct Dataset *dataset);
void save_image_to_disk(struct Image *image);

struct Dataset * load_dataset(const char *dataset_path);
int save_average_to_dataset(struct Dataset *dataset, struct Image *average);
int save_dataset_to_disk(struct Dataset *dataset, char *path);

#endif
