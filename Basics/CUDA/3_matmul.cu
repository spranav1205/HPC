#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                    \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error: %s (err_num=%d)\n",       \
                    cudaGetErrorString(err), err);                  \
            exit(err);                                              \
        }                                                           \
    }

void print_peak_memory() {
    size_t free_mem, total_mem, used_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    used_mem = total_mem - free_mem;
    printf("Peak GPU Memory Used: %.2f MB / %.2f MB\n", 
           used_mem / (1024.0 * 1024.0), 
           total_mem / (1024.0 * 1024.0));
}

void cpu_matmul(float *A, float *B, float *C, int m, int k, int n)
{
    // Dimensions of A are (m,k), B are (k,n) and thus C are (m,n)

    for (int a=0; a<m; a++)
    {
        for (int b=0; b<n; b++)
        {
            for (int i=0; i<k; i++)
            {
                *(C + (a*n) + b) += (*(A + (a*k) + i)) * (*(B + (i*n) + b));
            }
        }
    }
}

__global__ void gpu_matmul (float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        for (int i=0; i<k; i++)
        {
            *(C + (row*n) + col) += (*(A + (row*k) + i)) * (*(B + (i*n) + col));

            // atomicAdd(&C[row * n + col], A[row * k + i] * B[i * n + col]);
        }
    }
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
    int m = 4000, k = 400, n = 4000;

    // Use pinned (page-locked) host memory so cudaMemcpyAsync is truly asynchronous
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_A, m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_B, k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_C, m * n * sizeof(float)));

    initialize_matrix(h_A, m, k, 1.0f);
    initialize_matrix(h_B, k, n, 2.0f);
    // Ensure output starts from zero before CPU and GPU paths
    initialize_matrix(h_C, m, n, 0.0f);

    // Time

    clock_t start, end;
    double cpu_time_used, gpu_time_used;

    start = clock();

    cpu_matmul(h_A, h_B, h_C, m, k, n);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time taken: %f seconds\n", cpu_time_used);

    // printf("Result matrix C:\n");
    // for (int i=0; i<m; i++)
    // {
    //     for (int j=0; j<n; j++)
    //     {
    //         printf("%f ", *(h_C + i*n + j));
    //     }
    //     printf("\n");
    // }

    // GPU Part

    // We'll use two copy streams and one compute stream.
    // Copy streams will "notify" compute stream via events when H2D copies finish.
    cudaStream_t sCopyA, sCopyB, sCompute;
    cudaEvent_t aDone, bDone;
    CHECK_CUDA_ERROR(cudaStreamCreate(&sCopyA));
    CHECK_CUDA_ERROR(cudaStreamCreate(&sCopyB));
    CHECK_CUDA_ERROR(cudaStreamCreate(&sCompute));
    CHECK_CUDA_ERROR(cudaEventCreate(&aDone));
    CHECK_CUDA_ERROR(cudaEventCreate(&bDone));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, m * n * sizeof(float)));

    // Start H2D copies in separate streams
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice, sCopyA));
    // Notify: when A's H2D copy in sCopyA reaches this point, event aDone is recorded
    CHECK_CUDA_ERROR(cudaEventRecord(aDone, sCopyA));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice, sCopyB));
    // Notify: when B's H2D copy in sCopyB reaches this point, event bDone is recorded
    CHECK_CUDA_ERROR(cudaEventRecord(bDone, sCopyB));

    // Compute stream waits for both copies to finish before proceeding
    // Wait: sCompute will not execute subsequent ops until aDone has occurred in sCopyA
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(sCompute, aDone, 0));
    // Wait: sCompute will not execute subsequent ops until bDone has occurred in sCopyB
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(sCompute, bDone, 0));

    // Zero C in the compute stream to keep operations ordered within sCompute
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_C, 0, m * n * sizeof(float), sCompute));


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Ceiling division

    // Optional: time only the kernel using CUDA events on the compute stream
    cudaEvent_t kernelStart, kernelStop;
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&kernelStop));
    CHECK_CUDA_ERROR(cudaEventRecord(kernelStart, sCompute));

    // Kernel runs in sCompute and will start only after the two waits above
    gpu_matmul<<<numBlocks, threadsPerBlock, 0, sCompute>>>(d_A, d_B, d_C, m, k, n);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventRecord(kernelStop, sCompute));

    // Copy result back on sCompute; runs after the kernel by stream order
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost, sCompute));

    // Synchronize the compute stream to finish memset -> kernel -> D2H
    CHECK_CUDA_ERROR(cudaStreamSynchronize(sCompute));

    float ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, kernelStart, kernelStop));
    printf("GPU Kernel Time: %.3f ms\n", ms);
    
    print_peak_memory();

    // At this point, sCompute is synchronized; results in h_C are ready.
    

    // printf("Result matrix C from GPU:\n");
    // for (int i=0; i<m; i++)
    // {
    //     for (int j=0; j<n; j++)
    //     {
    //         printf("%f ", *(h_C + i*n + j));   
    //     }
    //     printf("\n");
    // }
    // Cleanup
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    cudaEventDestroy(aDone);
    cudaEventDestroy(bDone);
    cudaStreamDestroy(sCopyA);
    cudaStreamDestroy(sCopyB);
    cudaStreamDestroy(sCompute);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    printf("Performance Summary:\n");
    printf("CPU Time taken: %f seconds\n", cpu_time_used);
    printf("GPU Kernel Time: %.3f ms\n", ms);
    printf("Speedup (CPU time / GPU kernel time): %.2f x\n", cpu_time_used / (ms / 1000.0));

    return 0;
    
}