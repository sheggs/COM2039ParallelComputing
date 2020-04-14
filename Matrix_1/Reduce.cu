#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#define BLOCK_SIZE 32
#define N 8192
// This is the kernal
__global__ void reduceKernal(float* d_I, float*d_O) {
    // The ID for the thread selected (Including block dimensions).
    int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    // The thread ID
    int threadid = threadIdx.x;
    // Creating a shared memory space
    int block = blockDim.x * 1;
    __shared__ float* temp;
    printf("wow %i", block);
    temp = new float[block];
    printf("BSize: %i",block);
    // Store the thread id and its value in shared memory. This is because accessing shared memory has a lower latency than global memory.
    temp[threadid] = d_I[globalid];
    // Waiting for threads to complete. [Thread guard]
    __syncthreads();
    printf("BLOCK DIM: %i", blockDim.x);
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
 
    printf("\n Kernal Done %f \n", temp[0]);
}

// Uses atomic add.
__global__ void reduceGPUKernal(float* d_I, float* d_O) {
    // The ID for the thread selected (Including block dimensions).
    int globalid = blockDim.x * blockIdx.x + threadIdx.x;
    // The thread ID
    int threadid = threadIdx.x;
    // Creating a shared memory space
    int block = blockDim.x * 1;
    __shared__ float* temp;
    printf("wow %i", block);
    temp = new float[block];
    printf("BSize: %i", block);
    // Store the thread id and its value in shared memory. This is because accessing shared memory has a lower latency than global memory.
    temp[threadid] = d_I[globalid];
    // Waiting for threads to complete. [Thread guard]
    __syncthreads();
    printf("BLOCK DIM: %i", blockDim.x);
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
    if (threadid == 0) 
        /*d_O[blockIdx.x] = temp[threadid];*/
        atomicAdd(d_O, temp[0]);
    //printf("\n Kernal Done %f \n", temp[0]);
}


// Best way to reduce using GPU is by using the reduce kernal recurivly.
float gpuOldReduction(float * input) {
    printf("GPU REDUCTION STARTED");

    int noThreads = (N / BLOCK_SIZE);
    float* h_output = new float[1];
    float* h_input = input;
    // Getting the number of blocks. Each block has 2 elements.
    int numberofBlocks = ceil(noThreads / BLOCK_SIZE);
    if (numberofBlocks == 0) {
        numberofBlocks = 1;
    }
    float* h_block = new float[1];
    h_block[0] = (float)numberofBlocks;
    printf("I: %i F: %f", numberofBlocks, h_block[0]);
    float* d_input, * d_output, * d_block;
    // Allocating space for the input array in the GPU
    cudaError_t err = cudaMalloc((void**)&d_input, sizeof(float) * noThreads);
    // Printing the error for allocating space.
    printf("CUDA malloc d_final_input: %s\n", cudaGetErrorString(err));
    // Allocating space for the output array in the GPU
    err = cudaMalloc((void**)&d_output, sizeof(float) * numberofBlocks);
    // Printing the error for allocating space.
    printf("CUDA malloc d_final_output: %s\n", cudaGetErrorString(err));
    // Copying the input data (input) to the GPU (d_final_input)
    err = cudaMemcpy(d_input, h_input, sizeof(float) * noThreads, cudaMemcpyHostToDevice);
    // Any copy errors from CUDA will be displayed
    printf("Copy Error CPU -> GPU : %s\n", cudaGetErrorString(err));

    printf("BLOCK: %i\n", numberofBlocks);
    int no1 = BLOCK_SIZE / 2;

    dim3 threadPerBlock;
    dim3 noBlocks;
    bool ifSwitch = true;
    threadPerBlock = dim3(32);
    noBlocks = dim3(numberofBlocks);
    while (noThreads > 0) {

        numberofBlocks = ceil(noThreads / no1);
        if (numberofBlocks == 0) {
            numberofBlocks = 1;
        }
        printf("\nNoThreads %i\n", noThreads);
        if (ifSwitch) {
            printf("\nSwitched Once\n");
            reduceKernal << <noBlocks, threadPerBlock >> > (d_input, d_output);
            ifSwitch = false;
            if (noThreads == 1) break;
        }
        else {
            printf("\nSwitched Twice\n");
            reduceKernal << <noBlocks, threadPerBlock >> > (d_output, d_input);
            ifSwitch = true;
            if (noThreads == 1) break;
        }
        noThreads = noThreads / 32;
        if (noThreads == 0) {
            noThreads = 1;
        }
        no1 = no1 / 2;
    }
    //reduceKernal << <noBlocks, threadPerBlock >> > (d_input, d_output);
    err = cudaDeviceSynchronize();
    // Any kernal running errors to be printed.
    printf("RUN kernal %s\n", cudaGetErrorString(err));
    // Any errors from copying the output from the GPU to CPU
    if (ifSwitch) {
        printf("From h_ouptut");
        err = cudaMemcpy(h_output, d_output, sizeof(float) * numberofBlocks, cudaMemcpyDeviceToHost);
    }
    else {
        printf("From input");

        err = cudaMemcpy(h_output, d_input, sizeof(float) * numberofBlocks, cudaMemcpyDeviceToHost);
    }
    printf("\nTesting 1: %f %f\n", h_output[0], h_output[1]);

    printf("\nFinal %f\n:", h_output[0]);
    cudaFree(d_input);
    cudaFree(d_output);

    return h_output[0];
}

// Best way to reduce using GPU is by using the reduce kernal recurivly.
float gpuReduction(float* d_input, float * d_output) {
    printf("GPU REDUCTION STARTED");
    int block_number = (N / BLOCK_SIZE) / BLOCK_SIZE;
    float* h_output = new float[block_number];
    int noThreads = (N / BLOCK_SIZE);
    // Getting the number of blocks. Each block has 2 elements.
    int numberofBlocks = ceil(noThreads / BLOCK_SIZE);
    if (numberofBlocks == 0) {
        numberofBlocks = 1;
    }
    int no1 = BLOCK_SIZE;
    dim3 threadPerBlock;
    dim3 noBlocks;
    bool ifSwitch = true;
    threadPerBlock = dim3(BLOCK_SIZE);
    noBlocks = dim3(numberofBlocks);
    while (noThreads > 0) {
        printf("\nBLOCK: %i\n", numberofBlocks);
        numberofBlocks = noThreads / no1;
        if (numberofBlocks == 0) {
            numberofBlocks = 1;
        }
        printf("\nNoThreads %i\n", noThreads);
        if (ifSwitch) {
           // printf("\nSwitched Once\n");
            reduceKernal << <noBlocks, threadPerBlock >> > (d_input, d_output);
            ifSwitch = false;
            if (noThreads == 1) break;
        }
        else {
            printf("\nSwitched Twice\n");
            reduceKernal << <noBlocks, threadPerBlock >> > (d_output, d_input);
            ifSwitch = true;
            if (noThreads == 1) break;
        }
        noThreads = noThreads / 32;
        if (noThreads == 0) {
            noThreads = 1;
        }
        no1 = no1 / 2;
    }

    //reduceKernal << <noBlocks, threadPerBlock >> > (d_input, d_output);
    cudaError_t err = cudaDeviceSynchronize();
    // Any kernal running errors to be printed.
    printf("RUN kernal %s\n", cudaGetErrorString(err));
    // Any errors from copying the output from the GPU to CPU
    if (ifSwitch) {
        printf("From h_ouptut");
        err = cudaMemcpy(h_output, d_output, sizeof(float) * numberofBlocks, cudaMemcpyDeviceToHost);
    }
    else {
        printf("From input");

        err = cudaMemcpy(h_output, d_input, sizeof(float) * numberofBlocks, cudaMemcpyDeviceToHost);
    }
   // printf("\nTesting 1: %f %f\n", h_output[0], h_output[1]);

    printf("\nFinal %f\n:", h_output[0]);


    cudaFree(d_input);
    cudaFree(d_output);
    return h_output[0];

}
void ReduceMiddleGPU(float* h_input, float* h_output) {
    // Setting up pointers for the device output and input arrays.
    float* d_input, * d_output, * d_gpu_output;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Allocating space for the input array in the GPU
    cudaError_t err = cudaMalloc((void**)&d_input, sizeof(float) * N);
    // Printing the error for allocating space.
    printf("CUDA malloc d_input: %s\n", cudaGetErrorString(err));
    // Allocating space for the output array in the GPU
    err = cudaMalloc((void**)&d_output, sizeof(float) * (N / BLOCK_SIZE));
    // Printing the error for allocating space.
    printf("CUDA malloc d_output: %s\n", cudaGetErrorString(err));
    // Copying the input data (h_input) to the GPU (d_input)
    err = cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);
    // Any copy errors from CUDA will be displayed
    printf("Copy Error CPU -> GPU : %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_gpu_output, sizeof(float) * (N / BLOCK_SIZE));
    printf("CUDA malloc d_gpu_output: %s\n", cudaGetErrorString(err));

    // Creating dimensions for the threads and blocks
    dim3 noThreads(BLOCK_SIZE);
    dim3 noBlocks((N / BLOCK_SIZE));
    cudaEventRecord(start);

    // Calling the GPU kernal with 512 threads and 16 blocks. Each block has 32 threads. The parameters are the input and output values.
    reduceKernal << < noBlocks, noThreads >> > (d_input, d_output);
    // Synchrnoizing the GPU (Waiting for all threads to finish).
    err = cudaDeviceSynchronize();
    // Any kernal running errors to be printed.
    printf("RUN kernal %s\n", cudaGetErrorString(err));
    float output = gpuReduction(d_output, d_gpu_output);
    
   // err = cudaMemcpy(h_output, d_output, sizeof(float) * (N / BLOCK_SIZE), cudaMemcpyDeviceToHost);
    //float output = gpuOldReduction(h_output);
    // Variable to compute the total.
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gpu_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nGPU N=%i Elapsed time was: %f\n milliseconds",N, milliseconds);
    printf("\n GPU Reduced Answer: %f\n Real answer: %i", output, N);



}




float cpuReduction(float* input) {
    float reduction = 0;
    // Looping through the output
    for (int i = 0; i < N / BLOCK_SIZE; i++) {
        // Adding the output value to the total.
        reduction = reduction + input[i];
    }
    return reduction;

}

void ReduceMiddleCPU(float* h_input, float * h_output) {
    // Setting up pointers for the device output and input arrays.
    float* d_input, * d_output;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

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
    cudaEventRecord(start);

    // Calling the GPU kernal with 512 threads and 16 blocks. Each block has 32 threads. The parameters are the input and output values.
    reduceKernal << < noBlocks, noThreads >> > (d_input, d_output);
    // Synchrnoizing the GPU (Waiting for all threads to finish).
    err = cudaDeviceSynchronize();
    // Any kernal running errors to be printed.
    printf("RUN kernal %s\n", cudaGetErrorString(err));
    // Any errors from copying the output from the GPU to CPU
    err = cudaMemcpy(h_output, d_output, sizeof(float)*(N/BLOCK_SIZE), cudaMemcpyDeviceToHost);
    printf("COPY FROM GPU %s :\n", cudaGetErrorString(err));

    // Variable to compute the total.
    float y = 0;

    cudaFree(d_input);
    cudaFree(d_output);
    float output = cpuReduction(h_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CPU N=%i Elapsed time was: %f\n milliseconds",N, milliseconds);
    printf("\n Reduced Answer: %f\n Real answer: %i", output,N);

    // Freeing GPU memory that was allocated to d_input and d_output


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
    ReduceMiddleGPU(input, output);


    return 0;

}
