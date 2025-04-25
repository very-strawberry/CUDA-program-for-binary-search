#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <ctime>

#define ARRAY_SIZE 10000000  // 10 million elements
#define NUM_SEARCHES 1000000  // 1 million searches
#define THREADS_PER_BLOCK 256
#define ITERATIONS 5

__global__ void binarySearchKernel(const int* __restrict__ array,
                                  const int* __restrict__ search_values,
                                  int* __restrict__ results,
                                  const int array_size,
                                  const int num_searches) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    // Each thread handles multiple searches
    for (int i = tid; i < num_searches; i += stride) {
        int target = search_values[i];
        int left = 0;
        int right = array_size - 1;
        int result = -1;




        while (left <= right) {
            int mid = left + (right - left) / 2;
            int mid_val = array[mid];


            if (mid_val == target) {
                result = mid;
                break;
            }


            if (mid_val < target)
                left = mid + 1;
            else
                right = mid - 1;
        }


        results[i] = result;
    }
}


// CPU version of binary search for comparison
int binarySearchCPU(const int *array, int target, int array_size) {
    int left = 0;
    int right = array_size - 1;


    while (left <= right) {
        int mid = left + (right - left) / 2;


        if (array[mid] == target)
            return mid;


        if (array[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }


    return -1;
}


// Utility function to perform GPU test and measure time
double runGPUTest(int *d_array, int *d_search_values, int *d_results, int *h_results_gpu, int num_blocks) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Start timing
    cudaEventRecord(start);


    // Launch kernel
    binarySearchKernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_array, d_search_values, d_results, ARRAY_SIZE, NUM_SEARCHES);


    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // Copy results back
    cudaMemcpy(h_results_gpu, d_results, NUM_SEARCHES * sizeof(int), cudaMemcpyDeviceToHost);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return milliseconds / 1000.0; // Convert to seconds
}


int main() {
    printf("Initializing arrays... (Array size: %d, Searches: %d)\n", ARRAY_SIZE, NUM_SEARCHES);


    // Allocate memory for arrays on host
    int *h_array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int *h_search_values = (int*)malloc(NUM_SEARCHES * sizeof(int));
    int *h_results_gpu = (int*)malloc(NUM_SEARCHES * sizeof(int));
    int *h_results_cpu = (int*)malloc(NUM_SEARCHES * sizeof(int));


    // Initialize a sorted array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = i * 2;  // Even numbers: 0, 2, 4, 6, ...
    }


    // Generate search values with various patterns to create more diverse workload
    srand(time(NULL));
    for (int i = 0; i < NUM_SEARCHES; i++) {
        int pattern = i % 4;
        switch (pattern) {
            case 0:  // Existing value (best case)
                h_search_values[i] = h_array[rand() % ARRAY_SIZE];
                break;
            case 1:  // Non-existing value (worst case)
                h_search_values[i] = rand() % (ARRAY_SIZE * 2) * 2 + 1;  // Odd numbers
                break;
            case 2:  // Search in lower half
                h_search_values[i] = h_array[rand() % (ARRAY_SIZE / 2)];
                break;
            case 3:  // Search in upper half
                h_search_values[i] = h_array[ARRAY_SIZE / 2 + rand() % (ARRAY_SIZE / 2)];
                break;
        }
    }


    // Allocate memory on the device
    int *d_array, *d_search_values, *d_results;
    cudaMalloc(&d_array, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&d_search_values, NUM_SEARCHES * sizeof(int));
    cudaMalloc(&d_results, NUM_SEARCHES * sizeof(int));


    // Copy data to device
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_array, h_array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_search_values, h_search_values, NUM_SEARCHES * sizeof(int), cudaMemcpyHostToDevice);


    // Set device to max performance
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaSetDeviceFlags(cudaDeviceScheduleAuto);


    // Calculate optimal number of blocks
    int max_blocks = prop.multiProcessorCount * 32;  // Typically good load
    int num_blocks = (NUM_SEARCHES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    num_blocks = (num_blocks > max_blocks) ? max_blocks : num_blocks;


    printf("Running tests with %d blocks of %d threads each\n", num_blocks, THREADS_PER_BLOCK);


    // --- GPU Binary Search ---
    // Run multiple times and take average for more stable measurement
    double gpu_time_total = 0.0;
    printf("Running GPU tests (%d iterations)...\n", ITERATIONS);


    for (int iter = 0; iter < ITERATIONS; iter++) {
        double gpu_time = runGPUTest(d_array, d_search_values, d_results, h_results_gpu, num_blocks);
        gpu_time_total += gpu_time;
        printf("  Iteration %d: %.6f seconds\n", iter+1, gpu_time);
    }


    double gpu_time_avg = gpu_time_total / ITERATIONS;


    // --- CPU Binary Search ---
    printf("Running CPU test...\n");
    clock_t cpu_start = clock();


    // Perform search on CPU
    for (int i = 0; i < NUM_SEARCHES; i++) {
        h_results_cpu[i] = binarySearchCPU(h_array, h_search_values[i], ARRAY_SIZE);
    }


    clock_t cpu_end = clock();
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC;


    // Display results for first 5 searches
    printf("\nResults for first 5 searches:\n");
    printf("%-10s %-15s %-15s %-15s %-15s\n", "Index", "Search Value", "Expected Index", "GPU Result", "CPU Result");
    printf("--------------------------------------------------------------\n");
    for (int i = 0; i < 5 && i < NUM_SEARCHES; i++) {
        int search_val = h_search_values[i];
        int expected_idx = (search_val % 2 == 0) ? search_val / 2 : -1;
        printf("%-10d %-15d %-15d %-15d %-15d\n",
               i, search_val, expected_idx, h_results_gpu[i], h_results_cpu[i]);
    }
    printf("\n");


    // Verify results
    printf("Verifying all results...\n");
    int errors = 0;
    for (int i = 0; i < NUM_SEARCHES; i++) {
        if (h_results_gpu[i] != h_results_cpu[i]) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at index %d: GPU = %d, CPU = %d, Search value = %d\n",
                       i, h_results_gpu[i], h_results_cpu[i], h_search_values[i]);
            }
        }
    }


    // Print performance results
    printf("\nBinary Search Performance:\n");
    printf("Array size: %d, Number of searches: %d\n", ARRAY_SIZE, NUM_SEARCHES);
    printf("GPU time (avg): %.6f seconds\n", gpu_time_avg);
    printf("CPU time: %.6f seconds\n", cpu_time);
    printf("Speedup: %.2f times faster\n", cpu_time / gpu_time_avg);


    if (errors == 0) {
        printf("All results match between CPU and GPU versions!\n");
    } else {
        printf("Found %d mismatches between CPU and GPU results.\n", errors);
    }


    // Free allocated memory
    free(h_array);
    free(h_search_values);
    free(h_results_gpu);
    free(h_results_cpu);
    cudaFree(d_array);
    cudaFree(d_search_values);
    cudaFree(d_results);


    return 0;
}

