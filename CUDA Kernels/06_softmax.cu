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
#include <cmath>

// CPU reference (skeleton)
void cpu_softmax(const float* in, float* out, int N) {
    // TODO: implement stable softmax
    // Placeholder: copy input to output
    for (int i=0;i<N;++i) out[i] = in[i];
}

// CUDA kernel skeleton (per-element exponentials)
__global__ void softmax_kernel(const float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // TODO: compute exp and normalization
    out[idx] = in[idx]; // placeholder
}

// GPU launcher
void gpu_softmax(const float* h_in, float* h_out, int N) {
    float *d_in=nullptr, *d_out=nullptr;
    size_t bytes = size_t(N)*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    int threads = 256; int blocks = (N + threads - 1)/threads;
    softmax_kernel<<<blocks,threads>>>(d_in, d_out, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "d2h out");

    cudaFree(d_in); cudaFree(d_out);
}

// Validation
bool validate_softmax(const float* ref, const float* got, int N, float eps=1e-4f) {
    float max_err=0.0f; int idx=0;
    for (int i=0;i<N;++i){ float e = std::fabs(ref[i]-got[i]); if (e>max_err){max_err=e; idx=i;} if (!(e<=eps)){ std::cerr<<"Mismatch at "<<i; return false;} }
    std::cout<<"Max err="<<max_err<<" at "<<idx<<std::endl; return true;
}

// Benchmark (GFLOPS placeholder)
void benchmark_softmax(const float* h_in, float* h_out, int N, int warmups=5, int iterations=50) {
    float *d_in=nullptr, *d_out=nullptr; size_t bytes = size_t(N)*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    int threads=256; int blocks=(N+threads-1)/threads;
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i=0;i<warmups;++i) softmax_kernel<<<blocks,threads>>>(d_in,d_out,N);
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    float ms; double total_ms=0.0,min_ms=1e9,max_ms=0.0;
    for (int it=0; it<iterations; ++it){
        cudaEventRecord(start);
        softmax_kernel<<<blocks,threads>>>(d_in,d_out,N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);
        total_ms+=ms; min_ms=std::min(min_ms,(double)ms); max_ms=std::max(max_ms,(double)ms);
    }
    double avg_ms = total_ms/iterations; double s = avg_ms/1000.0;
    // Placeholder FLOP count per element (e.g., exp + add = ~20 flops) -- TODO set accurate count
    double flops = double(N) * 20.0;
    double gflops = (flops/1e9)/s;
    std::cout<<"Softmax: avg_ms="<<avg_ms<<" ms, GFLOPS(estimate)="<<gflops<<"\n";

    cudaFree(d_in); cudaFree(d_out); cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(){
    int N = 1<<20;
    std::vector<float> in(N), ref(N), out(N);
    std::mt19937 rng(123); std::uniform_real_distribution<float> dist(0.0f,1.0f);
    for (int i=0;i<N;++i) in[i]=dist(rng);

    cpu_softmax(in.data(), ref.data(), N);
    gpu_softmax(in.data(), out.data(), N);
    if (!validate_softmax(ref.data(), out.data(), N)) return 1;
    benchmark_softmax(in.data(), out.data(), N);
    return 0;
}
