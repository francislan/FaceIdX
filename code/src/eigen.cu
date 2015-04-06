#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <errno.h>
#include <math.h>

#define INITIAL_SIZE 100000
#define PARTIAL_SIZE 10000
#define THREADS_PER_BLOCK 256
 
// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// GPU version
__global__ void isPrime_gpu (unsigned int max, unsigned int first_index,
                             unsigned int *array, unsigned int *count)
{
    // For reduction
    __shared__ int cache[THREADS_PER_BLOCK];
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    // index for cache
    int t = threadIdx.x;
    int isPrime = 1;
    if (n >= PARTIAL_SIZE || n + first_index >= max) {
        isPrime = 0;
    } else {
        unsigned int a = array[n];
        if (a < 2)
            isPrime = 0;
        if (a % 2 == 0 && a > 2)
            isPrime = 0;

        if (isPrime == 1) {
            unsigned int b = 3;
            unsigned int b_max = floor(sqrt((float)a));

            while (b <= b_max && isPrime == 1) {
                if (a % b == 0)
                    isPrime = 0;
                b += 2;
            }
        }
    }
    cache[t] = isPrime;

    __syncthreads();

    // Reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (t < i)
            cache[t] += cache[t + i];
        __syncthreads();
        i /= 2;
    }
    if (t == 0)
        count[blockIdx.x] = cache[0];
}

// Returns 0 if success
int fill_inputs_from_file(char *filename, unsigned int **inputs, unsigned int *n)
{
    int current_size = INITIAL_SIZE;
    *n = 0;
    char *line = NULL;
    size_t len = 0;
    int read;
    char *p;
    int tmp;

    FILE* file = fopen(filename, "r");
    if (file == NULL)
        return 1;

    *inputs = (unsigned int *)malloc(INITIAL_SIZE * sizeof(unsigned int));
    if (*inputs == NULL)
        return 1;

    // safe parsing
    while ((read = getline(&line, &len, file)) != -1) {
        if (line[read - 1] == '\n')
            line[read - 1] = '\0';
        tmp = strtol(line, &p, 10);

        if (*p != '\0' || (tmp == 0 && errno != 0)) {
            return 2;
        } else {
            // array size not enough, reallocation necessary
            if (*n >= current_size) {
                current_size *= 2;
                *inputs = (unsigned int *)realloc(*inputs, current_size * sizeof(unsigned int));
                if (*inputs == NULL)
                    return 1;
            }
            (*inputs)[*n] = tmp;
        }
        (*n)++;
    }

    fclose(file);
    if (line)
        free(line);

    return 0;
}


int count_prime(int argc, char **argv)
{
    cudaDeviceProp prop;
    int whichDevice;
    GPU_CHECKERROR(cudaGetDevice(&whichDevice));
    GPU_CHECKERROR(cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }

    cudaEvent_t start, stop;
    GPU_CHECKERROR(
    cudaEventCreate(&start)
    );
    GPU_CHECKERROR(
    cudaEventCreate(&stop)
    );
    float elapsedTime;

    unsigned int *h_inputs;
    unsigned int nr_inputs;

    if (argc != 2) {
        printf("usage: hw4 filename");
        return EXIT_FAILURE;
    }

    switch(fill_inputs_from_file(argv[1], &h_inputs, &nr_inputs)) {
        case 1:
            perror("error");
            return EXIT_FAILURE;

        case 2:
            printf("error reading file: not a number.\n");
            return EXIT_FAILURE;

        default:
            break;
    }

    // Allocate host locked
    int size_fill_up = ((nr_inputs + PARTIAL_SIZE - 1) / PARTIAL_SIZE) * PARTIAL_SIZE;
    unsigned int *h_inputs_array;
    GPU_CHECKERROR(cudaHostAlloc((void**) &h_inputs_array,
                                 size_fill_up * sizeof(unsigned int),
                                 cudaHostAllocDefault));

    GPU_CHECKERROR(
    cudaMemcpy((void *) h_inputs_array,
                (void *) h_inputs,
                nr_inputs * sizeof (unsigned int),
                cudaMemcpyHostToHost)
    );
    unsigned int num_blocks = (PARTIAL_SIZE + THREADS_PER_BLOCK - 1) /
                                THREADS_PER_BLOCK;
    unsigned int num_chunks = (nr_inputs + PARTIAL_SIZE - 1) / PARTIAL_SIZE;
    unsigned int *d_inputs_array_0, *d_inputs_array_1;

    // array that stores the partial count of # of primes
    // no use of atomic operation
    unsigned int *d_partial_count_0, *d_partial_count_1;
    unsigned int *h_partial_count;

    GPU_CHECKERROR(
    cudaHostAlloc((void**) &h_partial_count,
                  num_chunks * num_blocks * sizeof(unsigned int),
                  cudaHostAllocDefault)
    );

    GPU_CHECKERROR(
    cudaMalloc((void **) &d_inputs_array_0, PARTIAL_SIZE * sizeof(unsigned int))
    );
    GPU_CHECKERROR(
    cudaMalloc((void **) &d_inputs_array_1, PARTIAL_SIZE * sizeof(unsigned int))
    );

    GPU_CHECKERROR(
    cudaMalloc((void **) &d_partial_count_0, num_blocks * sizeof(unsigned int))
    );
    GPU_CHECKERROR(
    cudaMalloc((void **) &d_partial_count_1, num_blocks * sizeof(unsigned int))
    );

    GPU_CHECKERROR(cudaEventRecord(start, 0));

    cudaStream_t stream0, stream1;
    GPU_CHECKERROR(cudaStreamCreate(&stream0));
    GPU_CHECKERROR(cudaStreamCreate(&stream1));

    for (int k = 0; k < 100; k++) {
        for (int i = 0; i < num_chunks; i += 2) {
            GPU_CHECKERROR(
            cudaMemcpyAsync(d_inputs_array_0,
                            h_inputs_array + i * PARTIAL_SIZE,
                            PARTIAL_SIZE * sizeof (unsigned int),
                            cudaMemcpyHostToDevice,
                            stream0)
            );

            isPrime_gpu<<<num_blocks, THREADS_PER_BLOCK, 0, stream0>>>
                                                (nr_inputs,
                                                 i * PARTIAL_SIZE,
                                                 d_inputs_array_0,
                                                 d_partial_count_0);

            if (i + 1 < num_chunks) {
                GPU_CHECKERROR(
                cudaMemcpyAsync(d_inputs_array_1,
                                h_inputs_array + (i+1) * PARTIAL_SIZE,
                                PARTIAL_SIZE * sizeof (unsigned int),
                                cudaMemcpyHostToDevice,
                                stream1)
                );

                isPrime_gpu<<<num_blocks, THREADS_PER_BLOCK, 0, stream1>>>
                                                    (nr_inputs,
                                                     (i+1) * PARTIAL_SIZE,
                                                     d_inputs_array_1,
                                                     d_partial_count_1);
            }

            GPU_CHECKERROR(
            cudaMemcpy(h_partial_count + i * num_blocks,
                       d_partial_count_0,
                       num_blocks * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost)
            );

            if (i + 1 < num_chunks) {
                GPU_CHECKERROR(
                cudaMemcpy(h_partial_count + (i+1) * num_blocks,
                           d_partial_count_1,
                           num_blocks * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost)
                );
            }
        }
    }
    GPU_CHECKERROR(cudaStreamSynchronize(stream0));
    GPU_CHECKERROR(cudaStreamSynchronize(stream1));

    // Finish the sum on the CPU side
    unsigned int h_nr_prime_gpu = 0;
    for (int i = 0; i < num_blocks * num_chunks; i++)
        h_nr_prime_gpu += h_partial_count[i];

    GPU_CHECKERROR(cudaEventRecord(stop, 0));
    GPU_CHECKERROR(cudaEventSynchronize(stop));
    GPU_CHECKERROR(cudaEventElapsedTime(&elapsedTime,
                                         start, stop));
    printf("%d %3.1f\n", h_nr_prime_gpu, elapsedTime / 1000.0);

    // free up the memory:
    GPU_CHECKERROR(cudaFreeHost(h_inputs_array));
    GPU_CHECKERROR(cudaFreeHost(h_partial_count));
    GPU_CHECKERROR(cudaFree(d_partial_count_0));
    GPU_CHECKERROR(cudaFree(d_partial_count_1));
    GPU_CHECKERROR(cudaFree(d_inputs_array_0));
    GPU_CHECKERROR(cudaFree(d_inputs_array_1));
    GPU_CHECKERROR(cudaStreamDestroy(stream0));
    GPU_CHECKERROR(cudaStreamDestroy(stream1));
    GPU_CHECKERROR(cudaEventDestroy(start));
    GPU_CHECKERROR(cudaEventDestroy(stop));
    free(h_inputs);

    return EXIT_SUCCESS;
}
