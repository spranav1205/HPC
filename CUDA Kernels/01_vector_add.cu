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
void cpu_vector_add(const float* a, const float* b, float* out, size_t N) {
    for (size_t i = 0; i < N; ++i) out[i] = a[i] + b[i];
}

// CUDA kernel skeleton
__global__ void vector_add_kernel(const float* a, const float* b, float* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    out[idx] = a[idx] + b[idx];
}

// GPU launcher
void gpu_vector_add(const float* h_a, const float* h_b, float* h_out, size_t N) {
    float *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    size_t bytes = N * sizeof(float);
    checkCuda(cudaMalloc(&d_a, bytes), "malloc d_a");
    checkCuda(cudaMalloc(&d_b, bytes), "malloc d_b");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "memcpy h2d a");
    checkCuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "memcpy h2d b");

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "memcpy d2h out");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

// Validation
bool validate_vector_add(const float* ref, const float* got, size_t N, float eps=1e-5f) {
    float max_err = 0.0f;
    size_t max_idx = 0;
    for (size_t i=0;i<N;++i) {
        float e = std::fabs(ref[i]-got[i]);
        if (e > max_err) { max_err = e; max_idx = i; }
        if (!(e <= eps)) {
            std::cerr << "Validation failed at " << i << ": ref=" << ref[i] << " got=" << got[i] << " err=" << e << std::endl;
            return false;
        }
    }
    std::cout << "Max error: " << max_err << " at " << max_idx << std::endl;
    return true;
}

// Benchmarking (CUDA events)
void benchmark_vector_add(const float* h_a, const float* h_b, float* h_out, size_t N,
                          int warmups=5, int iterations=50) {
    float *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    size_t bytes = N * sizeof(float);
    checkCuda(cudaMalloc(&d_a, bytes), "malloc d_a");
    checkCuda(cudaMalloc(&d_b, bytes), "malloc d_b");
    checkCuda(cudaMalloc(&d_out, bytes), "malloc d_out");
    checkCuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "h2d a");
    checkCuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "h2d b");

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmups
    for (int i=0;i<warmups;++i) {
        vector_add_kernel<<<blocks,threads>>>(d_a,d_b,d_out,N);
    }
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    // Timed runs
    float ms; double total_ms=0.0; double min_ms=1e9; double max_ms=0.0;
    for (int it=0; it<iterations; ++it) {
        cudaEventRecord(start);
        vector_add_kernel<<<blocks,threads>>>(d_a,d_b,d_out,N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms; min_ms = std::min(min_ms, (double)ms); max_ms = std::max(max_ms, (double)ms);
    }

    double avg_ms = total_ms / iterations;
    double seconds = avg_ms / 1000.0;
    // Typical vector-add reads 2 arrays and writes 1 array -> 3 * N * sizeof(float)
    double gb = (3.0 * double(N) * sizeof(float)) / 1e9;
    std::cout << "VectorAdd: avg_ms=" << avg_ms << " ms, min=" << min_ms << " ms, max=" << max_ms << " ms\n";
    std::cout << "Memory BW (GB/s): " << (gb / seconds) << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    size_t N = 1<<25; // 1M elements
    std::vector<float> a(N), b(N), ref(N), out(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(0.0f,1.0f);
    for (size_t i=0;i<N;++i) { a[i]=d(rng); b[i]=d(rng); }

    cpu_vector_add(a.data(), b.data(), ref.data(), N);
    gpu_vector_add(a.data(), b.data(), out.data(), N);

    if (!validate_vector_add(ref.data(), out.data(), N)) {
        std::cerr << "Validation failed" << std::endl; return 1;
    }

    benchmark_vector_add(a.data(), b.data(), out.data(), N);

    return 0;
}
