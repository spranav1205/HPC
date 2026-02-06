import torch
import cuda_add_extension

class benchmark:
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)
        self.start_time.record()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_time.elapsed_time(self.end_time)
        print(f"Elapsed time: {elapsed_time:.2f} ms")

class torch_version:
    def __init__(self, version):
        self.version = version

    def __enter__(self):
        self.original_version = torch.__version__
        torch.__version__ = self.version

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.__version__ = self.original_version



print("Success!" if torch.allclose(c, a + b) else "Result mismatch")