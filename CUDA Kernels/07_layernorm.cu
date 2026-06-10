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
void cpu_layernorm(const float* in, float* out, int N, int D) {
    // TODO: implement mean/variance and normalization over D dimension
    // Placeholder: copy input to output (no normalization)
    for (int i=0;i<N*D;++i) out[i] = in[i];
}

// CUDA kernel skeleton (per-row layernorm)
__global__ void layernorm_kernel(const float* in, float* out, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    // TODO: compute mean/variance and normalize across D
    for (int j=0;j<D;++j) {
        int idx = row * D + j;
        out[idx] = in[idx]; // placeholder
    }
}

// GPU launcher
void gpu_layernorm(const float* h_in, float* h_out, int N, int D) {
    float *d_in=nullptr, *d_out=nullptr;
    size_t bytes = size_t(N)*D*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    layernorm_kernel<<<blocks,threads>>>(d_in,d_out,N,D);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "d2h out");

    cudaFree(d_in); cudaFree(d_out);
}

// Validation
bool validate_layernorm(const float* ref, const float* got, int N, int D, float eps=1e-5f) {
    int ND = N*D; float max_err=0.0f; int idx=0;
    for (int i=0;i<ND;++i){ float e = std::fabs(ref[i]-got[i]); if (e>max_err){max_err=e; idx=i;} if (!(e<=eps)){ std::cerr<<"Mismatch at "<<i; return false;} }
    std::cout<<"Max err="<<max_err<<" at "<<idx<<std::endl; return true;
}

// Benchmark (Memory-bound: GB/s)
void benchmark_layernorm(const float* h_in, float* h_out, int N, int D, int warmups=5, int iterations=50) {
    float *d_in=nullptr, *d_out=nullptr; size_t bytes = size_t(N)*D*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    int threads=256; int blocks=(N+threads-1)/threads;
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i=0;i<warmups;++i) layernorm_kernel<<<blocks,threads>>>(d_in,d_out,N,D);
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    float ms; double total_ms=0.0,min_ms=1e9,max_ms=0.0;
    for (int it=0; it<iterations; ++it){
        cudaEventRecord(start);
        layernorm_kernel<<<blocks,threads>>>(d_in,d_out,N,D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);
        total_ms+=ms; min_ms=std::min(min_ms,(double)ms); max_ms=std::max(max_ms,(double)ms);
    }
    double avg_ms = total_ms/iterations; double s = avg_ms/1000.0;
    // LayerNorm typically reads input and writes output => ~2 * N*D * sizeof(float)
    double gb = (2.0 * double(N) * double(D) * sizeof(float)) / 1e9;
    std::cout<<"LayerNorm: avg_ms="<<avg_ms<<" ms, BW(GB/s)="<<(gb/s)<<"\n";

    cudaFree(d_in); cudaFree(d_out); cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(){
    int N=1024, D=512; // rows x features
    std::vector<float> in(size_t(N)*D), ref(size_t(N)*D), out(size_t(N)*D);
    std::mt19937 rng(123); std::uniform_real_distribution<float> dist(0.0f,1.0f);
    for (size_t i=0;i<in.size();++i) in[i]=dist(rng);

    cpu_layernorm(in.data(), ref.data(), N, D);
    gpu_layernorm(in.data(), out.data(), N, D);
    if (!validate_layernorm(ref.data(), out.data(), N, D)) return 1;
    benchmark_layernorm(in.data(), out.data(), N, D);
    return 0;
}
