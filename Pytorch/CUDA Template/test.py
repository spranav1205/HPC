import torch
import time
import activation_cuda_backend as activation_cuda

class fn(torch.autograd.Function):
    @staticmethod
    def forward(x):
        x = activation_cuda.activation(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class torchVersion:
    # Same functionality implemented in PyTorch
    @staticmethod
    def forward(x):
        return x * x + 2 * x
    
def benchmark(fn, x, name, num_iters=100):
    for _ in range(10):
        y = fn.forward(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        y = fn.forward(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_iters
    print(f"{name} average time over {num_iters} iterations: {avg_time*1000:.4f} ms")
    return y

def main():
    torch.manual_seed(0)
    x = torch.randn(1000, 1000, device='cuda')
    c = benchmark(fn, x, "Custom CUDA Extension")
    c_torch = benchmark(torchVersion, x, "PyTorch Implementation")


if __name__ == "__main__":
    main()