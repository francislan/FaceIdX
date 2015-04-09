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

#define PRINT(level, fmt, ...) \
do { \
	if (!strcmp(level, "BUG")) { \
		printf(KRED, "[Error]: "); \
		printf(KRED, fmt, ##__VA_ARGS__); \
		printf(KRNM, ""); \
	} else if (!strcmp(level, "WARN")) { \
		printf(KYEL, "[Warning]: "); \
		printf(KYEL, fmt, ##__VA_ARGS__); \
		printf(KRNM, ""); \
	} else if (!strcmp(level, "INFO")) { \
		printf(KBLU, "[INFO]: "); \
		printf(KBLU, fmt, ##__VA_ARGS__); \
		printf(KRNM, ""); \
	} else { \
		printf(KNRM, fmt, ##__VA_ARGS__); \
	} \
} while(0)

#endif
