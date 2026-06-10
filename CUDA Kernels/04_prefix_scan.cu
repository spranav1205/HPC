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
void cpu_prefix_scan(const float* in, float* out, size_t N) {
    // TODO: implement correct inclusive/exclusive scan
    // Placeholder: output = input (no scan)
    for (size_t i=0;i<N;++i) out[i] = in[i];
}

// CUDA kernel skeleton for block-wise scan
__global__ void scan_kernel(const float* in, float* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // TODO: implement scan logic
    out[idx] = in[idx]; // placeholder
}

// GPU launcher
void gpu_prefix_scan(const float* h_in, float* h_out, size_t N) {
    float *d_in=nullptr, *d_out=nullptr;
    size_t bytes = N*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    scan_kernel<<<blocks, threads>>>(d_in, d_out, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "d2h out");

    cudaFree(d_in); cudaFree(d_out);
}

// Validation
bool validate_scan(const float* ref, const float* got, size_t N, float eps=1e-5f) {
    float max_err=0.0f; size_t idx=0;
    for (size_t i=0;i<N;++i){ float e = std::fabs(ref[i]-got[i]); if (e>max_err){max_err=e; idx=i;} if (!(e<=eps)){ std::cerr<<"Mismatch at "<<i; return false;} }
    std::cout<<"Max err="<<max_err<<" at "<<idx<<std::endl; return true;
}

// Benchmark (elements/sec)
void benchmark_scan(const float* h_in, float* h_out, size_t N, int warmups=5, int iterations=50) {
    float *d_in=nullptr, *d_out=nullptr; size_t bytes = N*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    int threads = 256; int blocks = (N + threads - 1)/threads;
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    for (int i=0;i<warmups;++i) scan_kernel<<<blocks,threads>>>(d_in,d_out,N);
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    float ms; double total_ms=0.0,min_ms=1e9,max_ms=0.0;
    for (int it=0; it<iterations; ++it){
        cudaEventRecord(start);
        scan_kernel<<<blocks,threads>>>(d_in,d_out,N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);
        total_ms+=ms; min_ms=std::min(min_ms,(double)ms); max_ms=std::max(max_ms,(double)ms);
    }
    double avg_ms = total_ms/iterations; double s = avg_ms/1000.0;
    double elems_per_sec = double(N)/s;
    std::cout<<"PrefixScan: avg_ms="<<avg_ms<<" ms, elems/s="<<elems_per_sec<<"\n";

    cudaFree(d_in); cudaFree(d_out); cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(){
    size_t N = 1<<20;
    std::vector<float> in(N), ref(N), out(N);
    std::mt19937 rng(123); std::uniform_real_distribution<float> d(0.0f,1.0f);
    for (size_t i=0;i<N;++i) in[i]=d(rng);

    cpu_prefix_scan(in.data(), ref.data(), N);
    gpu_prefix_scan(in.data(), out.data(), N);
    if (!validate_scan(ref.data(), out.data(), N)) return 1;
    benchmark_scan(in.data(), out.data(), N);
    return 0;
}
