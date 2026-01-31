#include "6_faster_matmul.cuh"
#include <cstring>
#include <stdio.h>
#include <math.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

struct BenchResult {
    float time_ms;
    double gflops;
};

BenchResult benchmark_kernel(const char *name, int M, int N, int K, double flops,
                             float *d_A, float *d_B, float *d_C, float *h_C,
                             const float *h_C_ref, const float *h_C_naive,
                             dim3 gridDim, dim3 blockDim, int kernel_id) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    memcpy(h_C, (void*)h_C_ref, M * N * sizeof(float));
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float time_ms = 0.0f;
    
    cudaEventRecord(start, 0);
    
    if (kernel_id == 0) {
        naive_matmul<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if (kernel_id == 1) {
        coalescing_matmul<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if (kernel_id == 2) {
        cacheblocking_matmul<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if (kernel_id == 3) {
        const int BM = 64, BN = 64, BK = 8, TM = 4;
        oneD_blockTiling_matmul<BM,BN,BK,TM><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if (kernel_id == 4) {
        const int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
        twoD_blockTiling_matmul<BM,BN,BK,TM,TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    double gflops = (flops / 1e9) / (time_ms / 1000.0);
    
    printf("[%s] Time: %.3f ms | GFLOPS: %.1f", name, time_ms, gflops);
    compare_matrices(h_C_naive, h_C, M, N);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    return {time_ms, gflops};
}

int main() {
    const int M = 4096, N = 4096, K = 4096;
    double flops = 2.0 * M * N * K;
    
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *h_C_ref = (float*)malloc(M * N * sizeof(float));
    float *h_C_naive = (float*)malloc(M * N * sizeof(float));
    
    generate_matrix(h_A, M, K);
    generate_matrix(h_B, K, N);
    generate_matrix(h_C, M, N);
    memcpy(h_C_ref, h_C, M * N * sizeof(float));
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("\n====== Matrix Mult Benchmark (M=%d, N=%d, K=%d) ======\n", M, N, K);
    printf("Total FLOPs: %.2e\n\n", flops);
    
    // Naive (reference)
    dim3 blockDim1(32, 32);
    dim3 gridDim1(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    benchmark_kernel("Naive", M, N, K, flops, d_A, d_B, d_C, h_C, h_C_ref, h_C, gridDim1, blockDim1, 0);
    memcpy(h_C_naive, h_C, M * N * sizeof(float));
    
    // Others
    dim3 blockDim2(1024);
    dim3 gridDim2(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    
    benchmark_kernel("Coalescing", M, N, K, flops, d_A, d_B, d_C, h_C, h_C_ref, h_C_naive, gridDim2, blockDim2, 1);
    benchmark_kernel("CacheBlock", M, N, K, flops, d_A, d_B, d_C, h_C, h_C_ref, h_C_naive, gridDim2, blockDim2, 2);
    
    const int BM = 64, BN = 64, BK = 8, TM = 4;
    dim3 gridDim3(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim3((BM / TM) * BN);
    benchmark_kernel("1D Tiling", M, N, K, flops, d_A, d_B, d_C, h_C, h_C_ref, h_C_naive, gridDim3, blockDim3, 3);
    
    const int TN = 4;
    dim3 gridDim4(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    dim3 blockDim4((BM * BN) / (TM * TN));
    benchmark_kernel("2D Tiling", M, N, K, flops, d_A, d_B, d_C, h_C, h_C_ref, h_C_naive, gridDim4, blockDim4, 4);
    
    printf("\n====== Done ======\n\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref); free(h_C_naive);
    
    return 0;
}
