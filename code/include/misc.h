#ifndef MISC_H
#define MISC_H

#define TEST_MALLOC(p) \
do { \
	if ((p) == NULL) { \
		printf("ERROR: malloc failed\n"); \
		exit(1); \
	} \
} while(0)

#endif
