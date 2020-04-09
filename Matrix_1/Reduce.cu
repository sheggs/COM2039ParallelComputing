#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#define BLOCK_SIZE 32
#define N 512
// This is the kernal
__global__ void reduceKernal(float* d_I, float*d_O) {
    // The ID for the thread selected (Including block dimensions).
    int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    // The thread ID
    int threadid = threadIdx.x;
    // Creating a shared memory space
    __shared__ float temp[BLOCK_SIZE];
    // Store the thread id and its value in shared memory. This is because accessing shared memory has a lower latency than global memory.
    temp[threadid] = d_I[globalid];
    // Waiting for threads to complete. [Thread guard]
    __syncthreads();
    // Looping through from half the block dimension. We will be doing reducing by 'Sequential Addressing'.
    for (unsigned int i = blockDim.x / 2; i >= 1; i >>= 1) {
        // Having a check to ensure the thread id is smaller. This is because we want the thread id to always be less than the value of 'i'.
        if (threadid < i) {
            // Adding the the values together.
            temp[threadid] += temp[threadid + i];
        }
        // Waiting for threads to synchrnoize
        __syncthreads();
    }
    // All values in the temporary array should be the total value of the block. So when everything is finished we write back to global memory.
    if (threadid == 0) d_O[blockIdx.x] = temp[threadid];
}

void ReduceMiddle(float* h_input, float * h_output) {
    // Setting up pointers for the device output and input arrays.
    float* d_input, * d_output;
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    
    // Allocating space for the input array in the GPU
    cudaError_t err = cudaMalloc((void**)&d_input, sizeof(float)*N);
    // Printing the error for allocating space.
    printf("CUDA malloc d_input: %s\n", cudaGetErrorString(err));
    // Allocating space for the output array in the GPU
    err = cudaMalloc((void**)&d_output, sizeof(float) * (N/BLOCK_SIZE));
    // Printing the error for allocating space.
    printf("CUDA malloc d_output: %s\n", cudaGetErrorString(err));
    // Copying the input data (h_input) to the GPU (d_input)
    err = cudaMemcpy(d_input, h_input, sizeof(float)*N, cudaMemcpyHostToDevice);
    // Any copy errors from CUDA will be displayed
    printf("Copy Error CPU -> GPU : %s\n", cudaGetErrorString(err));

    // Creating dimensions for the threads and blocks
    dim3 noThreads(BLOCK_SIZE);
    dim3 noBlocks((N / BLOCK_SIZE));

    //cudaEventRecord(start);
    // Calling the GPU kernal with 512 threads and 16 blocks. Each block has 32 threads. The parameters are the input and output values.
    reduceKernal << < noBlocks, noThreads >> > (d_input, d_output)
    // Synchrnoizing the GPU (Waiting for all threads to finish).
    err = cudaDeviceSynchronize();
    // Any kernal running errors to be printed.
    printf("RUN kernal %s\n", cudaGetErrorString(err));
    // Any errors from copying the output from the GPU to CPU
    err = cudaMemcpy(h_output, d_output, sizeof(float)*(N/BLOCK_SIZE), cudaMemcpyDeviceToHost);
    printf("COPY FROM GPU %s :", cudaGetErrorString(err));
    // Variable to compute the total.
    float y = 0;
    printf("\n OUTPUT \n");
    // Looping through the output
    for (int i = 0; i < N/BLOCK_SIZE;i++) {
        // Printing each value in the output
        printf("%f", h_output[i]);
        // Adding the output value to the total.
        y = y + h_output[i];
    }
    // Printing the output value.
    printf("\n And the final reduction is: %f\n", y);
    printf("\n INPUT \n");
    // Looping through the input
    for (int i = 0; i < N; i++) {
        // Printing the input values
        printf("%f", h_input[i]);
    }

    // Freeing GPU memory that was allocated to d_input and d_output
    cudaFree(d_input);
    cudaFree(d_output);

}

int main() {
    // Creating an output and input array. Output array is the size of the grid. Input array is the number of elements we want to reduce.
    float* output = new float[N / BLOCK_SIZE];
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
    ReduceMiddle(input, output);
    return 0;

}
