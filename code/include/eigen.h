#ifndef EIGEN_H
#define EIGEN_H

struct Image * compute_average_gpu(struct Dataset * dataset);
struct Image * compute_average_cpu(struct Dataset * dataset);

#endif


// pixel_grey = get_pixel(images[0], 0, 2, 0);

/*
struct Image *images;
// load images
int num_images = 100;


struct Image average;

average = compute_average_gpu(images, num_images);

save_average_to_dataset()

*/
