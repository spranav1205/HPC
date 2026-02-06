#include <torch/extension.h>
#include <cuda_runtime.h>

// A simple vector addition kernel
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// Wrapper function to interface with PyTorch
void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int size = a.size(0);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        size
    );
}

// Binding the C++ function to a Python name
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
}