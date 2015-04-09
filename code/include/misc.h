#ifndef MISC_H
#define MISC_H

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KWHT  "\x1B[37m"

#define PRINT(level, fmt, ...) \
do { \
	if (!strcmp(level, "BUG")) { \
		printf("\x1B[31m" "[Error]: "); \
		printf("\x1B[31m" fmt, ##__VA_ARGS__); \
		printf("\x1B[0m" ""); \
	} else if (!strcmp(level, "WARN")) { \
		printf("\x1B[33m" "[Warning]: "); \
		printf("\x1B[33m" fmt, ##__VA_ARGS__); \
		printf("\x1B[0m" ""); \
	} else if (!strcmp(level, "INFO")) { \
		printf("\x1B[34m" "[INFO]: "); \
		printf("\x1B[34m" fmt, ##__VA_ARGS__); \
		printf("\x1B[0m" ""); \
	} else { \
		printf("\x1B[0m" fmt, ##__VA_ARGS__); \
	} \
} while(0)

#define TEST_MALLOC(p) \
do { \
	if ((p) == NULL) { \
		printf("ERROR: malloc failed\n"); \
		exit(1); \
	} \
} while(0)


#endif
