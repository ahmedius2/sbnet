from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_path = './sbnet/ops/'
setup(
    name='sbnet',
    version='0.0.1',
    packages=['sbnet'],
    cmdclass={
        'build_ext': BuildExtension
    },
    ext_modules=[CUDAExtension('sbnet.ops', [
            ext_path + 'reduce_mask_cuda.cu',
            ext_path + 'reduce_mask.cpp',
            ext_path + 'sparse_gather_cuda.cu',
            ext_path + 'sparse_gather.cpp',
            ext_path + 'pybind_modules.cpp',
    ])]
)
