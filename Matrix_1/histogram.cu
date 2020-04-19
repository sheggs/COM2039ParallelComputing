#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <iostream>
//#define BIN_COUNT 8
#define BLOCK_SIZE 64
#define N 512

__global__ void histogram(int* d_in, int* d_bins, const int BIN_COUNT)
{
	// Storing the global thread id
	int gid = threadIdx.x + (blockDim.x * blockIdx.x);
	// Storing the bin number
	int whatBin = d_in[gid] % BIN_COUNT;
	// Atomic add on the bin in global memory (incrementing by one).
	atomicAdd(&(d_bins[whatBin]), 1);

}
// This funciton sets up the histogram kernal

void histogramMiddle(int* h_input, int* h_bins) {
	// Creating pointers for the device input and bin storage
	int* d_input, * d_bins;
	// Initializing the error variable
	cudaError_t err;
	// Intalizing variables.
	int noThreads = BLOCK_SIZE;
	int noBlocks = N / BLOCK_SIZE;
	int BIN_COUNT = 8;
	// Allocating the input and bin on the device.
	
	err = cudaMalloc((void**)&d_input, sizeof(int) * N);
	printf("\n Allocating d_input error %s \n", cudaGetErrorString(err));
	err = cudaMalloc((void**)&d_bins, sizeof(int) * (N / BLOCK_SIZE));
	printf("\n Allocating d_bins error %s \n", cudaGetErrorString(err));

	// Copying the data to the GPU;
	err = cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);
	printf("\n Copying input data from CPU -> GPU error: %s \n", cudaGetErrorString(err));
	
	// Now call the Kernal
	histogram<<< noBlocks, noThreads >>>(d_input, d_bins,BIN_COUNT);
	err = cudaDeviceSynchronize();
	printf("\n Kernel error: %s \n", cudaGetErrorString(err));

	// Time to copy the bins back into the CPU
	err = cudaMemcpy(h_bins, d_bins, sizeof(int) * (N / BLOCK_SIZE), cudaMemcpyDeviceToHost);
	printf("\n Copying bins from GPU -> CPU error: %s \n", cudaGetErrorString(err));
	for (int i = 0; i < (N / BLOCK_SIZE); i++) {
		printf("\n bin_id %i %i \n ",i%BLOCK_SIZE,h_bins[i]);
	}
	return;
}
int main(void)
{
	// Initalizing the arrays
	int* input = new int[N];
	int* bins = new int[N % BLOCK_SIZE];
	// Putting in values
	// The way this is being set up [0] = 0; [255] = 255; [N] = N;
	for (int i = 0; i < N; i++) {
		input[i] = i;
	}
	// Calling the function that prepares the kernal.
	histogramMiddle(input, bins);
}
