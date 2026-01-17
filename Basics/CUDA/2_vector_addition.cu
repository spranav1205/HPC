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

__global__ void vector_addition_3D(const float* A, const float* B, float* C, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny; 
        if(idx < nx*ny*nz)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
}

void generate_3D_grid(int N, dim3 &gridDim, dim3 &blockDim)
{
    int threadsPerBlock = 8; // 8x8x8 = 512 threads per block
    blockDim = dim3(threadsPerBlock, threadsPerBlock, threadsPerBlock);

    int blocksX = (int)ceil(cbrt((float)N) / threadsPerBlock);
    int blocksY = blocksX;
    int blocksZ = blocksX;

    gridDim = dim3(blocksX, blocksY, blocksZ);
}

void generate_vector_3D(int N, float* h_A, float* h_B)
{
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
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

    // Test 3D vector addition
    const int N3D = 1 << 15; // 32K elements
    size_t size3D = N3D * sizeof(float);
    float *h_A3D = (float*)malloc(size3D);
    float *h_B3D = (float*)malloc(size3D);
    float *h_C3D = (float*)malloc(size3D);  

    generate_vector_3D(N3D, h_A3D, h_B3D);

    float *d_A3D, *d_B3D, *d_C3D;
    cudaMalloc((void**)&d_A3D, size3D);
    cudaMalloc((void**)&d_B3D, size3D);
    cudaMalloc((void**)&d_C3D, size3D);

    cudaMemcpy(d_A3D, h_A3D, size3D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B3D, h_B3D, size3D, cudaMemcpyHostToDevice);

    dim3 gridDim3D, blockDim3D;
    generate_3D_grid(N3D, gridDim3D, blockDim3D);

    // Time

    cudaEvent_t start3D, stop3D;
    cudaEventCreate(&start3D);
    cudaEventCreate(&stop3D);

    cudaEventRecord(start3D);   
    vector_addition_3D<<<gridDim3D, blockDim3D>>>(d_A3D, d_B3D, d_C3D, blockDim3D.x * gridDim3D.x, blockDim3D.y * gridDim3D.y, blockDim3D.z * gridDim3D.z);
    cudaEventRecord(stop3D);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C3D, d_C3D, size3D, cudaMemcpyDeviceToHost);

    float gpu_time3D = 0;
    cudaEventElapsedTime(&gpu_time3D, start3D, stop3D);
    printf("3D GPU time: %.3f ms\n", gpu_time3D);

    // Free device memory for 3D vectors
    cudaFree(d_A3D);
    cudaFree(d_B3D);
    cudaFree(d_C3D);

    // Free host memory for 3D vectors
    free(h_A3D);
    free(h_B3D);
    free(h_C3D);

    return 0;
}