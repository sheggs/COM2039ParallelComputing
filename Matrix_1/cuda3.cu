#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#define BLOCK_SIZE 32
#define N 43008
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
    printf("KOUT: %f", temp[threadid]);
}

__global__ void finalScan(float* d_firstScan, float* d_secondScan) {
    //printf("\n FINAL SCAN");
    // The ID for the thread selected (Including block dimensions).
    int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    // The thread ID
    int threadid = threadIdx.x;
    // Creating a shared memory space
    int block = blockDim.x * 1;
    int block_id = blockIdx.x * 1;
    // This shared array stores the second scan. This is will be used to be added onto the final output.
    //__shared__ float* addifier;
    // The size of the array is N/BLOCK_SIZE as it is the number of blocks in the grid.
    //addifier = new float[N / BLOCK_SIZE];
    // Synchronising the threads.
    //__syncthreads();
    // Checking if the block id is greater than 0 because we don't want to change the first block values. Also, we don't want to go into negative values if we subtract one from the index.
    //if (blockIdx.x > 0) addifier[blockIdx.x - 1] = d_secondScan[blockIdx.x - 1];
    // Synchronising the threads.
    //__syncthreads();
    // Shared array that stores the first scan results.
    //__shared__ float* temp;
    //temp = new float[512];
    //__syncthreads();

    // Store the thread id and its value in shared memory. This is because accessing shared memory has a lower latency than global memory.
    //temp[threadid] = d_firstScan[globalid];
    // Waiting for threads to complete. [Thread guard]
    //__syncthreads();
    printf("BLOCK DIM: %i", blockDim.x);
    // Checking if the block ID is greater than zero as block ID zero must not be modified.

    if (blockIdx.x > 0) {
        // Add the second scan result onto the first scan result.

        d_firstScan[globalid] = d_firstScan[globalid] + d_secondScan[blockIdx.x - 1];;
        printf("\n Hmm %f \n", d_secondScan[blockIdx.x - 1]);

        // Synchornise the threads
        __syncthreads();
    }
    // Write back into global memory
    d_firstScan[globalid] = d_firstScan[globalid];
    // printf("\n KERN: %f \n ", temp[threadid]);
    __syncthreads();

}

float* addScan(float* x, float* y, int n, float * d_input, float * d_output) {
    float* g = new float[N];
    dim3 noBlocks;
    dim3 noThreads;
    noBlocks = dim3(n / BLOCK_SIZE);
    noThreads = dim3(BLOCK_SIZE);
    cudaError_t err;
    //float* d_input, * d_aux;
    //err = cudaMalloc((void**)&d_input, sizeof(float) * n);
    //printf("\nError:d_input %s\n", cudaGetErrorString(err));
    //err = cudaMalloc((void**)&d_output, sizeof(float) * n / BLOCK_SIZE);
    //printf("\nError:d_aux %s\n", cudaGetErrorString(err));


    err = cudaMemcpy(d_input, x, sizeof(float) * n, cudaMemcpyHostToDevice);
    printf("\nError:cudaMemcpy %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_output, y, sizeof(float) * n / BLOCK_SIZE, cudaMemcpyHostToDevice);
    printf("\nError:d_aux %s\n", cudaGetErrorString(err));

    finalScan << < noBlocks, noThreads >> > (d_input, d_output);
    err = cudaDeviceSynchronize();
    printf("\nError: cudaDeviceSynchronize%s\n", cudaGetErrorString(err));
    err = cudaMemcpy(g, d_input, sizeof(float) * n, cudaMemcpyDeviceToHost);
    printf("\nError: cudaMemcpy %s\n", cudaGetErrorString(err));

    //cudaFree(d_input);
    //cudaFree(d_output);

    return g;
}
float* getEnds(float* x, int n) {
    int n2 = n / BLOCK_SIZE;
    float* output = new float[n2];
    for (int i = 0; i < n / BLOCK_SIZE; i++) {
        output[i] = x[((i + 1) * BLOCK_SIZE) - 1];
    }
    return output;
}
float* runScan(float* x, int n, float * d_input, float * d_output) {
    dim3 noBlocks;
    dim3 noThreads;
    noBlocks = dim3(n / BLOCK_SIZE);
    noThreads = dim3(BLOCK_SIZE);
    //float* d_input, float* d_output;
    //cudaMalloc((void**)&d_input, sizeof(float) * n);
    //cudaMalloc((void**)&d_output, sizeof(float) * n);
    cudaMemcpy(d_input, x, sizeof(float) * n, cudaMemcpyHostToDevice);
    scanKernal << < noBlocks, noThreads >> > (d_input, d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(x, d_output, sizeof(float) * n, cudaMemcpyDeviceToHost);
    //cudaFree(d_input);
    //cudaFree(d_output);
    return x;

}
float* recursion(int n, float* aux) {
    //float* res = runScan(aux, n);
    return nullptr;
}

float* f(float* a, int n) {
    float placeholder = 0;
    for (int i = 0; i < n; i++) {
        a[i] = a[i] + placeholder;
        if ((i + 1) % BLOCK_SIZE == 0) {
            placeholder = a[i];
            //printf("F %f \n", a[i]);
        }
    }
    return a;
}
void scanMiddle(float* h_input, float* h_output) {
    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, sizeof(float) * N);
    cudaMalloc((void**)&d_output, sizeof(float) * N);

    float* firstScan = runScan(h_input, N, d_input, d_output);
    float* ends = getEnds(firstScan, N);
    float* scanEnds = runScan(ends, N / BLOCK_SIZE, d_input, d_output);
    float* g = f(scanEnds, N / BLOCK_SIZE);
    float* addScans = addScan(firstScan, g, N, d_input, d_output);


    for (int i = 0; i < N; i++) {
        printf("\n %i END_SCAN %f", i, addScans[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

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
    cudaDeviceReset();


    return 0;

}