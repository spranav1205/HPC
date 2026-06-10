// common.h - shared utilities for cuda-learning examples
#pragma once

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

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>

inline void checkCuda(cudaError_t err, const char* ctx = nullptr) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error" << (ctx ? " [" : "")
                  << (ctx ? ctx : "") << (ctx ? "]" : "")
                  << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Simple timer helper (host)
struct HostTimer {
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};
