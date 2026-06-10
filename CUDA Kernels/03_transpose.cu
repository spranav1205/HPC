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
void cpu_transpose(const float* in, float* out, int W, int H) {
    // TODO: implement correct CPU transpose
    // Placeholder: copy input to output (no transpose)
    int N = W*H;
    for (int i=0;i<N;++i) out[i] = in[i];
}

// CUDA kernel skeleton (naive transpose)
__global__ void transpose_kernel(const float* in, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    // TODO: write transposed element
    out[idx] = in[idx]; // placeholder (no-op)
}

// GPU launcher
void gpu_transpose(const float* h_in, float* h_out, int W, int H) {
    float *d_in=nullptr, *d_out=nullptr;
    size_t bytes = size_t(W)*H*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    dim3 threads(16,16);
    dim3 blocks((W + threads.x - 1)/threads.x, (H + threads.y - 1)/threads.y);
    transpose_kernel<<<blocks, threads>>>(d_in, d_out, W, H);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "d2h out");

    cudaFree(d_in); cudaFree(d_out);
}

// Validation
bool validate_transpose(const float* ref, const float* got, int W, int H, float eps=1e-5f) {
    int N = W*H; float max_err=0.0f; int max_i=0;
    for (int i=0;i<N;++i){ float e = std::fabs(ref[i]-got[i]); if (e>max_err){max_err=e;max_i=i;} if (!(e<=eps)){ std::cerr<<"Mismatch at "<<i; return false;} }
    std::cout<<"Max error="<<max_err<<" at "<<max_i<<std::endl; return true;
}

// Benchmark
void benchmark_transpose(const float* h_in, float* h_out, int W, int H, int warmups=5, int iterations=50) {
    float *d_in=nullptr, *d_out=nullptr;
    size_t bytes = size_t(W)*H*sizeof(float);
    checkCuda(cudaMalloc(&d_in, bytes), "malloc d_in");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "h2d in");

    dim3 threads(16,16);
    dim3 blocks((W + threads.x - 1)/threads.x, (H + threads.y - 1)/threads.y);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i=0;i<warmups;++i) transpose_kernel<<<blocks,threads>>>(d_in,d_out,W,H);
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    float ms; double total_ms=0.0,min_ms=1e9,max_ms=0.0;
    for (int it=0; it<iterations; ++it){
        cudaEventRecord(start);
        transpose_kernel<<<blocks,threads>>>(d_in,d_out,W,H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);
        total_ms+=ms; min_ms=std::min(min_ms,(double)ms); max_ms=std::max(max_ms,(double)ms);
    }
    double avg_ms = total_ms/iterations; double s = avg_ms/1000.0;
    // transpose reads and writes entire matrix => 2 * W*H * sizeof(float)
    double gb = (2.0 * double(W)*H * sizeof(float))/1e9;
    std::cout<<"Transpose: avg_ms="<<avg_ms<<" ms, BW(GB/s)="<<(gb/s)<<"\n";

    cudaFree(d_in); cudaFree(d_out); cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(){
    int W = 2048, H = 1024; // example
    std::vector<float> in(W*H), ref(W*H), out(W*H);
    std::mt19937 rng(123); std::uniform_real_distribution<float> dist(0.0f,1.0f);
    for (int i=0;i<W*H;++i) in[i]=dist(rng);

    cpu_transpose(in.data(), ref.data(), W, H);
    gpu_transpose(in.data(), out.data(), W, H);
    if (!validate_transpose(ref.data(), out.data(), W, H)) return 1;
    benchmark_transpose(in.data(), out.data(), W, H);
    return 0;
}
