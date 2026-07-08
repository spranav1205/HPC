#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>


// __global__ void saxpy(int n, float a, const float * __restrict__ x, float * __restrict__ y)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalThreads = gridDim.x * blockDim.x;

//     float4 *x4 = (float4*)x;
//     float4 *y4 = (float4*)y;
    
//     int elements4 = n / 4;

//     for(int idx = i; idx < elements4; idx += totalThreads)
//     {
//         float4 reg_x = x4[idx];
//         float4 reg_y = y4[idx];

//         reg_y.x = a * reg_x.x + reg_y.x;
//         reg_y.y = a * reg_x.y + reg_y.y;
//         reg_y.z = a * reg_x.z + reg_y.z;
//         reg_y.w = a * reg_x.w + reg_y.w;

//         y4[idx] = reg_y;
//     }
// }

__global__ void saxpy(int n, float a, const float * __restrict__ x, float * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    float4 *x4 = (float4*)x;
    float4 *y4 = (float4*)y;
    
    int elements4 = n / 4;

    for(int idx = i; idx < elements4; idx += totalThreads)
    {
        float4 reg_x = x4[idx];
        float4 reg_y = y4[idx];

        reg_y.x = a * reg_x.x + reg_y.x;
        reg_y.y = a * reg_x.y + reg_y.y;
        reg_y.z = a * reg_x.z + reg_y.z;
        reg_y.w = a * reg_x.w + reg_y.w;

        y4[idx] = reg_y;
    }
}

void cpu_saxpy(int n, float a, float *x, float *y, float *output)
{
    for (int i = 0; i < n; i++)
    {
        output[i] = a * x[i] + y[i];
    }
}

void initialize_random(float *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = rand() / (float)RAND_MAX;
}

int main()
{
    // 1<<20 elements = 1,048,576 elements. 
    // Must be divisible by 4 for proper float4 casting without complex bounds checking.
    int N = 1 << 25; 

    // Host allocations
    float *h_x      = (float*)malloc(N * sizeof(float));
    float *h_y      = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float *cpu_out  = (float*)malloc(N * sizeof(float));

    float a = static_cast<float>(M_PI);

    // 1. Initialize data on the Host first
    initialize_random(h_x, N);
    initialize_random(h_y, N);

    // 2. Compute golden baseline on CPU
    cpu_saxpy(N, a, h_x, h_y, cpu_out);

    // Device allocations
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // 3. Copy initialized data from Host to Device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Running SAXPY kernel with N = " << N << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 4. Execution configuration optimized for float4 usage
    int blockSize = 128;
    int numElements4 = N / 4; 
    int numBlocks = (numElements4 + blockSize - 1) / blockSize;

    saxpy<<<numBlocks, blockSize>>>(N, a, d_x, d_y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    // 5. Copy results back to Host
    cudaMemcpy(h_output, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(cpu_out[i] - h_output[i]) > 1e-4)
        {
            std::cerr << "Mismatch at index " << i << ": CPU = " << cpu_out[i] << ", GPU = " << h_output[i] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "SUCCESS: Verification passed!" << std::endl;
    }


    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_output);
    free(cpu_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}