#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void vector_addition(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void cpu_vector_addition(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

void initialize_vector(float* vec, int N, float value)
{
    for (int i = 0; i < N; ++i)
    {
        vec[i] = value;
    }
}

void vector_addition_3D(const float* A, const float* B, float* C, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
}

int main ()
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float); // Size in bytes

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);

    // Initialize host vectors
    initialize_vector(h_A, N, 1.0f);
    initialize_vector(h_B, N, 2.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Ceiling division



    // GPU vector addition
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_addition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU vector addition
    clock_t cpu_start = clock();
    cpu_vector_addition(h_A, h_B, h_C_ref, N);
    clock_t cpu_end = clock();
    float cpu_time = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;

    printf("GPU time: %.3f ms\n", gpu_time);
    printf("CPU time: %.3f ms\n", cpu_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);



    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}