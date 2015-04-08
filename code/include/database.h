#ifndef DATABASE_H
#define DATABASE_H

struct Image {
	unsigned char *data;
	int w;
	int h;
	int comp;
	int req_comp;
	const char *filename;
};

struct FaceCoordinates {
	char *name;
	int num_eigenfaces;
	float *coordinates; //malloc
};

struct Dataset {
	char *name;
	const char *path;
	int num_eigenfaces;
	int w;
	int h;
	unsigned char *average;
	unsigned char *eigenfaces; //malloc
	struct FaceCoordinates *faces; //malloc
};

struct Image load_image(const char *filename, int req_comp);
void free_image(struct Image image);
unsigned char get_pixel(struct Image image, int x, int y, int comp);

struct Dataset create_dataset(const char *directory, const char *dataset_path, char *name);
struct Dataset load_dataset(const char *dataset_path);
int save_average_to_dataset(struct Dataset dataset, struct Image average);
int save_dataset_to_disk(struct Dataset dataset, char *path);

#endif
