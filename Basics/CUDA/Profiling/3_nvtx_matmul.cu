#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>



__global__ void gpu_matmul (float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        for (int i=0; i<k; i++)
        {
            *(C + (row*n) + col) += (*(A + (row*k) + i)) * (*(B + (i*n) + col));
        }
    }
}

void matrixMul(float *A, float *B, float *C, int m, int k, int n)
{
    nvtxRangePush("Matrix Multiplication");
    float *d_A, *d_B, *d_C;

    nvtxRangePush("Memory Allocation");
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(float));
    nvtxRangePop();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    nvtxRangePush("Kernel Execution");
    gpu_matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();
}


void initialize_matrix(float *mat, int rows, int cols, float value)
{
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            *(mat + i*cols + j) = value;
        }
    }
}

int main()
{
    float *h_A, *h_B, *h_C;
    int m = 4000, k = 4000, n = 4000;

    h_A = (float*)malloc(m * k * sizeof(float));
    h_B = (float*)malloc(k * n * sizeof(float));
    h_C = (float*)malloc(m * n * sizeof(float));

    initialize_matrix(h_A, m, k, 1.01f);
    initialize_matrix(h_B, k, n, 2.02f);

    // GPU Part
    matrixMul(h_A, h_B, h_C, m, k, n);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Matrix multiplication completed successfully.\n");

    return 0;
    
}