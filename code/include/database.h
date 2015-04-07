#ifndef DATABASE_H
#define DATABASE_H

unsigned char * loadImage(char *filename, int *w, int *h, int *comp, int req_comp);
void freeImage(unsigned char *data);

//TODO: structure for image (with w, h, n, etc)


#endif
