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
float cpu_reduction(const float* a, size_t N) {
    // TODO: implement a correct CPU reduction (sum)
    // Placeholder: return first element
    return (N>0) ? a[0] : 0.0f;
}

// CUDA kernel skeleton (per-block partial reduce)
__global__ void reduction_kernel(const float* a, float* partial, size_t N) {

}

// GPU launcher
float gpu_reduction(const float* h_a, size_t N) {
    float *d_a=nullptr, *d_partial=nullptr;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    checkCuda(cudaMalloc(&d_a, N * sizeof(float)), "malloc d_a");
    checkCuda(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice), "h2d a");
    checkCuda(cudaMalloc(&d_partial, blocks * sizeof(float)), "malloc d_partial");

    size_t shared = threads * sizeof(float);
    reduction_kernel<<<blocks, threads, shared>>>(d_a, d_partial, N);
    checkCuda(cudaGetLastError(), "kernel launch");

    std::vector<float> h_partial(blocks);
    checkCuda(cudaMemcpy(h_partial.data(), d_partial, blocks*sizeof(float), cudaMemcpyDeviceToHost), "d2h partial");
    // Finalize on CPU (placeholder)
    float result = (blocks>0) ? h_partial[0] : 0.0f;

    cudaFree(d_a); cudaFree(d_partial);
    return result;
}

// Validation
bool validate_reduction(float ref, float got, float eps=1e-4f) {
    float e = std::fabs(ref-got);
    if (e > eps) {
        std::cerr << "Reduction validation failed: ref="<<ref<<" got="<<got<<" err="<<e<<std::endl;
        return false;
    }
    std::cout << "Reduction max err="<<e<<std::endl;
    return true;
}

// Benchmark
void benchmark_reduction(const float* h_a, size_t N, int warmups=5, int iterations=50) {
    float *d_a=nullptr, *d_partial=nullptr;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    checkCuda(cudaMalloc(&d_a, N * sizeof(float)), "malloc d_a");
    checkCuda(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice), "h2d a");
    checkCuda(cudaMalloc(&d_partial, blocks * sizeof(float)), "malloc d_partial");

    size_t shared = threads * sizeof(float);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    for (int i=0;i<warmups;++i) reduction_kernel<<<blocks,threads,shared>>>(d_a,d_partial,N);
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    float ms; double total_ms=0.0,min_ms=1e9,max_ms=0.0;
    for (int it=0; it<iterations; ++it) {
        cudaEventRecord(start);
        reduction_kernel<<<blocks,threads,shared>>>(d_a,d_partial,N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);
        total_ms += ms; min_ms = std::min(min_ms,(double)ms); max_ms = std::max(max_ms,(double)ms);
    }

    double avg_ms = total_ms/iterations; double seconds = avg_ms/1000.0;
    double gb = (double(N)*sizeof(float))/1e9; // one read
    std::cout<<"Reduction: avg_ms="<<avg_ms<<" ms, BW(GB/s)="<<(gb/seconds)<<"\n";

    cudaFree(d_a); cudaFree(d_partial);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(){
    size_t N = 1<<20;
    std::vector<float> a(N);
    std::mt19937 rng(123); std::uniform_real_distribution<float> d(0.0f,1.0f);
    for (size_t i=0;i<N;++i) a[i]=d(rng);

    float cref = cpu_reduction(a.data(), N);
    float gref = gpu_reduction(a.data(), N);
    if (!validate_reduction(cref, gref)) return 1;
    benchmark_reduction(a.data(), N);
    return 0;
}
