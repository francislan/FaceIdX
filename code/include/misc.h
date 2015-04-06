#ifndef MISC_H
#define MISC_H

#define TEST_MALLOC(p) \
do { \
	if ((p) == NULL) { \
		printf("ERROR: malloc failed\n"); \
		exit(1); \
	} \
} while(0)

#define GET_PIXEL(data,x,y,w,r,g,b) \
do { \
	(r) = (data)[((y)*(w)+(x))* 4 + 0]; \
	(g) = (data)[((y)*(w)+(x))* 4 + 1]; \
	(b) = (data)[((y)*(w)+(x))* 4 + 2]; \
} while(0)

#endif
