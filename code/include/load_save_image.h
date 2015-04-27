#ifndef LOAD_SAVE_IMAGE_H
#define LOAD_SAVE_IMAGE_H

struct ImageGPU * load_image_gpu(const char *filename, int req_comp);
void save_image_to_disk_gpu(float *d_image, int w, int h, const char *name);

struct ImageCPU * load_image_cpu(const char *filename, int req_comp);
void save_image_to_disk_cpu(struct ImageCPU *image, const char *name);

#endif
