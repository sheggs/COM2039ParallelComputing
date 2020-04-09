//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//
//#define BLOCK_SIZE 16
//
//typedef struct { int width; int height; float* elements; } Matrix;
//
//__global__ void MatrixMultKern(const Matrix A, const Matrix B, const Matrix C) {
//    //  Calculate the column index of C and B
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    //  Calculate the row index of C and of A
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    if ((row < A.height) && (col < B.width)) {
//        float Cvalue = 0;
//        //  each thread computes one element of the block sub-matrix
//        for (int k = 0; k < A.width; ++k) {
//            Cvalue += A.elements[row * A.width + k] * B.elements[k * B.width + col];
//        }
//        C.elements[row * C.width + col] = Cvalue;
//    }
//}
//
////Matrix multiplication - Host Code
//// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
//void MatrixMult(const Matrix h_A, const Matrix h_B, Matrix h_C)
//{
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    // Load A and B into device memory
//    Matrix d_A;
//    d_A.width = h_A.width; d_A.height = h_A.height;
//    size_t size = h_A.width * h_A.height * sizeof(float);
//    cudaError_t err = cudaMalloc(&d_A.elements, size);
//    printf("CUDA malloc h_A: %s\n", cudaGetErrorString(err));
//    cudaMemcpy(d_A.elements, h_A.elements, size, cudaMemcpyHostToDevice);
//    Matrix d_B;
//    d_B.width = h_B.width; d_B.height = h_B.height;
//    size = h_B.width * h_B.height * sizeof(float);
//    err = cudaMalloc(&d_B.elements, size);
//    printf("CUDA malloc h_B: %s\n", cudaGetErrorString(err));
//    cudaMemcpy(d_B.elements, h_B.elements, size, cudaMemcpyHostToDevice);
//    // Allocate C in Device memory
//    Matrix d_C;
//    d_C.width = h_C.width; d_C.height = h_C.height;
//    size = h_C.width * h_C.height * sizeof(float);
//    err = cudaMalloc(&d_C.elements, size);
//    printf("CUDA malloc h_C: %s\n", cudaGetErrorString(err));
//
//    // Invoke Kernel
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//    dim3 dimGrid(d_B.width / dimBlock.x, d_A.height / dimBlock.y);
//    cudaEventRecord(start);
//
//    MatrixMultKern << < dimGrid, dimBlock >> > (d_A, d_B, d_C);
//    err = cudaThreadSynchronize();
//    cudaEventRecord(stop);
//
//    printf("Run kernel: %s\n", cudaGetErrorString(err));
//
//    //  Read C from Device to Host
//    err = cudaMemcpy(h_C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
//    printf("Copy h_C off device: %s\n", cudaGetErrorString(err));
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    printf("Matrix 1: Elapsed time was: %i %f\n milliseconds", h_A.width, milliseconds);
//    //  Free Device Memory
//    cudaFree(d_A.elements);
//    cudaFree(d_B.elements);
//    cudaFree(d_C.elements);
//}
//
//int pow(int y) {
//    int test = 16; 
//    for (int i = 0; i < (y-1); i++) {
//        test = test * 2;
//    }
//    return test;
//}
//int main(int argc, char* argv[]) {
//    Matrix a, b, c;
//    int size = 8192;
//    // read dimensions of a and b
//    a.height = size;
//    a.width = size;
//    b.height = a.width;
//    b.width = size;
//    a.elements = (float*)malloc(a.width * a.height * sizeof(float));
//    b.elements = (float*)malloc(b.width * b.height * sizeof(float));
//    c.height = a.height;
//    c.width = b.width;
//    c.elements = (float*)malloc(c.width * c.height * sizeof(float));
//    for (int i = 0; i < a.height; i++)
//        for (int j = 0; j < a.width; j++)
//            a.elements[i * a.width + j] = (float)(rand() % 3);
//    for (int i = 0; i < b.height; i++)
//        for (int j = 0; j < b.width; j++)
//            b.elements[i * b.width + j] = (float)(rand() % 2);
//    MatrixMult(a, b, c);
//    //for (int i = 0; i < a.height; i++) {
//    //    for (int j = 0; j < a.width; j++)
//    //        printf("%f ", a.elements[i * a.width + j]);
//    //    printf("\n");
//    //}
//    //printf("\n");
//    //for (int i = 0; i < b.height; i++) {
//    //    for (int j = 0; j < b.width; j++)
//    //        printf("%f ", b.elements[i * b.width + j]);
//    //    printf("\n");
//    //}
//    //printf("\n");
//    //for (int i = 0; i < c.height; i++) {
//    //    for (int j = 0; j < c.width; j++)
//    //        printf("%f ", c.elements[i * c.width + j]);
//    //    printf("\n");
//    //}
//    //printf("\n");
//    return 0;
//}