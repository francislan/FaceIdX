#ifndef MISC_H
#define MISC_H

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KWHT  "\x1B[37m"

#define TEST_MALLOC(p) \
do { \
	if ((p) == NULL) { \
		printf("ERROR: malloc failed\n"); \
		exit(1); \
	} \
} while(0)

#endif
