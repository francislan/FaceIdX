#ifndef DATABASE_H
#define DATABASE_H

struct Image {
	unsigned char *data;
	int w;
	int h;
	int comp;
	int req_comp;
	char *filename;
};

struct FaceCoordinates {
	char *name;
	int num_eigenfaces;
	float *coordinates; //malloc
};

struct Dataset {
	char *name;
	int num_eigenfaces;
	int w;
	int h;
	unsigned char *average;
	unsigned char *eigenfaces; //malloc
	struct FaceCoordinates *faces; //malloc
};

struct Image load_image(char *filename, int req_comp);
void free_image(struct Image image);
char get_pixel(struct Image image, int x, int y, int comp);

int create_dataset(char *directory, char *dataset_path, char *name);
int load_dataset(char *dataset_path);
int save_average_to_dataset(struct Dataset dataset, struct Image average);
int save_dataset_to_disk(struct Dataset dataset, char *path);

#endif
