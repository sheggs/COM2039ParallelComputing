#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#define BLOCK_SIZE 32
#define N 1024
// This is the kernal
__global__ void scanKernal(float* d_I, float* d_O) {
    // The ID for the thread selected (Including block dimensions).
    int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    // The thread ID
    int threadid = threadIdx.x;
    // Creating a shared memory space
    int block = blockDim.x * 1;
    __shared__ float* temp;
   // printf("wow %i", block);
    __syncthreads();

    temp = new float[block];
   // printf("BSize: %i", block);
    __syncthreads();

    // Store the thread id and its value in shared memory. This is because accessing shared memory has a lower latency than global memory.
    temp[threadid] = d_I[globalid];
    // Waiting for threads to complete. [Thread guard]
    __syncthreads();
   printf("BLOCK DIM: %i", blockDim.x);

    for (int offset = 1; offset < N; offset *= 2) {
        if (threadid >= offset)
            temp[threadid] += temp[threadid - offset];
        __syncthreads();
    }
    d_O[globalid] = temp[threadid];
}

__global__ void finalScan(float* d_firstScan, float* d_secondScan, float* d_O) {
    // The ID for the thread selected (Including block dimensions).
    int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    // The thread ID
    int threadid = threadIdx.x;
    // Creating a shared memory space
    int block = blockDim.x * 1;
    // This shared array stores the second scan. This is will be used to be added onto the final output.
    __shared__ float* addifier;
    // The size of the array is N/BLOCK_SIZE as it is the number of blocks in the grid.
    addifier = new float[N / BLOCK_SIZE];
    // Synchronising the threads.
    __syncthreads();
    // Checking if the block id is greater than 0 because we don't want to change the first block values. Also, we don't want to go into negative values if we subtract one from the index.
    if (blockIdx.x > 0) addifier[blockIdx.x - 1] = d_secondScan[blockIdx.x - 1];
    // Synchronising the threads.
    __syncthreads();
    // Shared array that stores the first scan results.
    __shared__ float* temp;
    temp = new float[block];
    // Store the thread id and its value in shared memory. This is because accessing shared memory has a lower latency than global memory.
    temp[threadid] = d_firstScan[globalid];
    // Waiting for threads to complete. [Thread guard]
    __syncthreads();
    printf("BLOCK DIM: %i", blockDim.x);
    // Checking if the block ID is greater than zero as block ID zero must not be modified.
    if (blockIdx.x > 0) {
        // Add the second scan result onto the first scan result.
        temp[threadid] = temp[threadid] + addifier[blockIdx.x - 1];
        // Synchornise the threads
        __syncthreads();
    }
    // Write back into global memory
    d_O[globalid] = temp[threadid];
}

void scanMiddle(float* h_input, float* h_output) {
    // d_input stores the first input
    // d_output now stores the second scan results.
    // d_first_output stores the first scan results.
    // d_final_output stores the final output where the scan is completly done.


    // Setting up pointers for the device output and input arrays.
    float* d_input, * d_output, * d_first_output, * d_final_output;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Creating an array to store the first scan results.
    float* firstScanResults = new float[N];
    // Allocating space for the input array in the GPU
    cudaError_t err = cudaMalloc((void**)&d_input, sizeof(float) * N);
    // Printing the error for allocating space.
    printf("CUDA malloc d_input: %s\n", cudaGetErrorString(err));
    // Allocating space for the output array in the GPU
    err = cudaMalloc((void**)&d_final_output, sizeof(float) * N);
    // Printing the error for allocating space.
    printf("CUDA malloc d_output: %s\n", cudaGetErrorString(err));
    // Allocating space for the output array in the GPU
    err = cudaMalloc((void**)&d_first_output, sizeof(float) * N);
    // Printing the error for allocating space.
    printf("CUDA malloc d_output: %s\n", cudaGetErrorString(err));
    // Copying the input data (h_input) to the GPU (d_input)
    err = cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);
    // Any copy errors from CUDA will be displayed
    printf("Copy Error CPU -> GPU : %s\n", cudaGetErrorString(err));
    // Creating dimensions for the threads and blocks
    dim3 noThreads(BLOCK_SIZE);
    dim3 noBlocks((N / BLOCK_SIZE));
    cudaEventRecord(start);
    // Calling the GPU kernal with N threads and N/BLOCK_SIZE blocks. Each block has 32 threads. The parameters are the input and output values.
    scanKernal << < noBlocks, noThreads >> > (d_input, d_first_output);
    // Synchrnoizing the GPU (Waiting for all threads to finish).
    err = cudaDeviceSynchronize();
    // Any kernal running errors to be printed.
    printf("RUN kernal %s\n", cudaGetErrorString(err));
    // Copying the first Scan results to the first output
    err = cudaMemcpy(firstScanResults, d_first_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // The number of elements in the output array for the second scan. Should be the number of blocks in the grid
    int output_items_count = (N/BLOCK_SIZE);
    // Intalise an array that will store the total of each block.
    float* outputFiltered = new float[output_items_count];
    // This is a counter that will be used to reference the outputFiltered array.
    int counter = 0;
    // Here we are looping through all the values at the end of each block and copying it over the outputFiltered array.
    for (int i = 1; i <= output_items_count; i++) {
        // x is the index for the last value in the block.
        int x = BLOCK_SIZE * i;
        // Copying the value.
        outputFiltered[counter] = firstScanResults[x - 1];
        // Incrementing the counter
        counter++;
    }
    // Printing errors.
    printf("\nCOPY FROM GPU %s :\n", cudaGetErrorString(err));
    cudaFree(d_input);    
    float* secondScanResults = new float[N / BLOCK_SIZE];
    // Allocating space for the input array in the GPU
    err = cudaMalloc((void**)&d_input, sizeof(float) * output_items_count);
    // Printing the error for allocating space.
    printf("CUDA malloc d_input: %s\n", cudaGetErrorString(err));
    // Allocating space for the output array in the GPU
    err = cudaMalloc((void**)&d_output, sizeof(float) * output_items_count);
    // Printing the error for allocating space.
    printf("CUDA malloc d_output: %s\n", cudaGetErrorString(err));
    // Copying the input data (outputFiltered) to the GPU (d_input)
    err = cudaMemcpy(d_input, outputFiltered, sizeof(float) * output_items_count, cudaMemcpyHostToDevice);
    // Any copy errors from CUDA will be displayed
    printf("Copy Error CPU -> GPU : %s\n", cudaGetErrorString(err));
    // Calling the scan kernal again for the second scan.
    scanKernal << < noBlocks, noThreads >> > (d_input, d_output);
    // Copying the second scan back to the Host.
    err = cudaMemcpy(secondScanResults, d_output, sizeof(float) * output_items_count, cudaMemcpyDeviceToHost);
    // Freeing the d_input we don't need this again.
    cudaFree(d_input);
    // This kernal completes the final mapping
    finalScan << < noBlocks, noThreads >> > (d_first_output, d_output,d_final_output);
    // Synchornising.
    err = cudaDeviceSynchronize();
    // Copying the final output
    err = cudaMemcpy(h_output, d_final_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // Printing the output
    printf("\nFinal OUTPUT\n");
    for (int i = 0; i < N; i++) {
        printf("\nFinal OUTPUT [%i] = %f\n", i, h_output[i]);
    }



    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Scan N=%i Elapsed time was: %f\n milliseconds", N, milliseconds);
    //printf("Scan Completed: Element 0: %f", h_output[32]);
    //printf("\n Reduced Answer: %f\n Real answer: %i", output, N);

    // Freeing GPU memory that was allocated to d_input and d_output
    cudaFree(d_first_output);
    cudaFree(d_output);
    cudaFree(d_final_output);


}

int main() {
    // Creating an output and input array. Output array is the size of the grid. Input array is the number of elements we want to reduce.
    float* output = new float[N];
    float* input = new float[N];
    // This variable is just to check the real final answer
    float x = 0;
    // Looping through int he input array and placing the values we want to reduce.
    for (int i = 0; i < N; i++) {
        input[i] = 1.0f;
        // Adding the values to the variable x to see the real final answer computed by the CPU
        x = x + 1.0f;
    }
    // Outputting the real final answer.
    printf("\nReal Answer: %f\n", x);
    // Calling the middle function that calls the kernal.
    scanMiddle(input, output);


    return 0;

}
