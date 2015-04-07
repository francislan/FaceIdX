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

struct Image loadImage(char *filename, int *w, int *h, int *comp, int req_comp);
void freeImage(unsigned char *data);
int get_pixel(struct Image image, int x, int y, int comp);

int create_dataset(char *directory, char *dataset_path, char *name);
int load_dataset(char *dataset_path);
int save_average_to_dataset(struct Dataset dataset, struct Image average);
int save_dataset_to_disk(struct Dataset dataset, char *path);

#endif
