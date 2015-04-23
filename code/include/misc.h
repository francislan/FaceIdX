#ifndef MISC_H
#define MISC_H

// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__ ))
static void gpuCheckError(cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
                file, line );
        exit(EXIT_FAILURE);
    }
}

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
	} else if (!strcmp(level, "DEBUG")) { \
		printf("\x1B[32m" "[DEBUG]: "); \
		printf("\x1B[32m" fmt, ##__VA_ARGS__); \
		printf("\x1B[0m" ""); \
	} else { \
		printf("\x1B[0m" fmt, ##__VA_ARGS__); \
	} \
} while(0)

#define TEST_MALLOC(p) \
do { \
	if ((p) == NULL) { \
		printf("\x1B[31m" "[Error]: malloc failed\n"); \
		exit(1); \
	} \
} while(0)

struct Timer
{
	cudaEvent_t start;
	cudaEvent_t end;
	float time;
};

#define START_TIMER(t) GPU_CHECKERROR(cudaEventRecord(t.start, 0))

#define STOP_TIMER(t) do { \
    GPU_CHECKERROR(cudaEventRecord(t.end, 0)); \
    GPU_CHECKERROR(cudaEventSynchronize(t.end)); \
    GPU_CHECKERROR(cudaEventElapsedTime(&(t.time), t.start, t.end)); \
} while(0)

#define INITIALIZE_TIMER(t) do { \
	GPU_CHECKERROR(cudaEventCreate(&(t.start))); \
	GPU_CHECKERROR(cudaEventCreate(&(t.end))); \
} while(0)

#define FREE_TIMER(t) do { \
	GPU_CHECKERROR(cudaEventDestroy(t.start)); \
	GPU_CHECKERROR(cudaEventDestroy(t.end)); \
} while(0)

#endif
