/*
Concepts:
- Coalescing
- Shared Memory
- Warp Primitives

Questions:
- Is this memory-bound or compute-bound?
- What is the arithmetic intensity?
- Which Nsight metrics matter?
*/

#include "common.h"
#include <vector>
#include <random>
#include <algorithm>

// CPU reference (skeleton)
void cpu_gemm(const float* A, const float* B, float* C, int N, int M, int K) {
    // TODO: implement CPU matmul (I x J x K loops)
    // Placeholder: zero the output
    std::fill(C, C + size_t(N)*M, 0.0f);
}

// CUDA kernel skeleton (naive)
__global__ void gemm_kernel(const float* A, const float* B, float* C, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= M) return;
    // TODO: implement dot-product accumulation
    float val = 0.0f; // placeholder
    C[row*M + col] = val;
}

// GPU launcher
void gpu_gemm(const float* h_A, const float* h_B, float* h_C, int N, int M, int K) {
    float *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    size_t bytesA = size_t(N)*K*sizeof(float);
    size_t bytesB = size_t(K)*M*sizeof(float);
    size_t bytesC = size_t(N)*M*sizeof(float);
    checkCuda(cudaMalloc(&d_A, bytesA), "malloc A");
    checkCuda(cudaMalloc(&d_B, bytesB), "malloc B");
    checkCuda(cudaMalloc(&d_C, bytesC), "malloc C");
    checkCuda(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice), "h2d A");
    checkCuda(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice), "h2d B");

    dim3 threads(16,16);
    dim3 blocks((M + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);
    gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost), "d2h C");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Validation
bool validate_gemm(const float* ref, const float* got, int N, int M, float eps=1e-3f) {
    size_t NM = size_t(N)*M; float max_err=0.0f; size_t idx=0;
    for (size_t i=0;i<NM;++i){ float e = std::fabs(ref[i]-got[i]); if (e>max_err){max_err=e; idx=i;} if (!(e<=eps)){ std::cerr<<"Mismatch at "<<i; return false;} }
    std::cout<<"Max err="<<max_err<<" at "<<idx<<std::endl; return true;
}

// Benchmark (GFLOPS)
void benchmark_gemm(const float* h_A, const float* h_B, float* h_C, int N, int M, int K, int warmups=3, int iterations=20) {
    float *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    size_t bytesA = size_t(N)*K*sizeof(float);
    size_t bytesB = size_t(K)*M*sizeof(float);
    size_t bytesC = size_t(N)*M*sizeof(float);
    checkCuda(cudaMalloc(&d_A, bytesA), "malloc A");
    checkCuda(cudaMalloc(&d_B, bytesB), "malloc B");
    checkCuda(cudaMalloc(&d_C, bytesC), "malloc C");
    checkCuda(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice), "h2d A");
    checkCuda(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice), "h2d B");

    dim3 threads(16,16);
    dim3 blocks((M + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i=0;i<warmups;++i) gemm_kernel<<<blocks,threads>>>(d_A,d_B,d_C,N,M,K);
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    float ms; double total_ms=0.0,min_ms=1e9,max_ms=0.0;
    for (int it=0; it<iterations; ++it){
        cudaEventRecord(start);
        gemm_kernel<<<blocks,threads>>>(d_A,d_B,d_C,N,M,K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);
        total_ms+=ms; min_ms=std::min(min_ms,(double)ms); max_ms=std::max(max_ms,(double)ms);
    }
    double avg_ms = total_ms/iterations; double s = avg_ms/1000.0;
    // FLOPS for GEMM ~ 2*N*M*K
    double flops = 2.0 * double(N) * double(M) * double(K);
    double gflops = (flops / 1e9) / s;
    std::cout<<"GEMM: avg_ms="<<avg_ms<<" ms, GFLOPS="<<gflops<<"\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(){
    int N=512, K=512, M=512; // moderate sizes
    std::vector<float> A(size_t(N)*K), B(size_t(K)*M), C(size_t(N)*M), cref(size_t(N)*M);
    std::mt19937 rng(123); std::uniform_real_distribution<float> dist(0.0f,1.0f);
    for (size_t i=0;i<A.size();++i) A[i]=dist(rng);
    for (size_t i=0;i<B.size();++i) B[i]=dist(rng);

    cpu_gemm(A.data(), B.data(), cref.data(), N, M, K);
    gpu_gemm(A.data(), B.data(), C.data(), N, M, K);
    if (!validate_gemm(cref.data(), C.data(), N, M)) return 1;
    benchmark_gemm(A.data(), B.data(), C.data(), N, M, K);
    return 0;
}
