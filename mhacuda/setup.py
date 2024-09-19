from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mha_cuda',
    ext_modules=[
        CUDAExtension(
            'mha_cuda', 
            ['mha_cuda_kernel.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-arch=sm_70']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

