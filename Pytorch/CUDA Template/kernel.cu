#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// A simple vector addition kernel
template<typename scalar_t>
__global__ void activation_kernel(const scalar_t*__restrict__ x, scalar_t* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = x[idx]*x[idx] + 2*x[idx]; // Example activation: square function
    }
}

// Wrapper function to interface with PyTorch
torch::Tensor activation_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    const int size = x.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    // Use .scalar_type() instead of .type()
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "activation_cuda", ([&] {
        activation_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

// Binding the C++ function to a Python name
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("activation", &activation_cuda, "Activation function (CUDA)");
}