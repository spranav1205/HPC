from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# python setup.py install
setup(
    name='activation_cuda',
    ext_modules=[
        CUDAExtension('activation_cuda_backend', [
            'kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)