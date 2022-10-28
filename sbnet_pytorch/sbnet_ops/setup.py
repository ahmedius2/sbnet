from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='sbnet',
		ext_modules=[CUDAExtension('sbnet', [
			'/reduce_mask_cuda.cu',
			'/reduce_mask.cpp',
			'/sparse_gather.cu',
			'/sparse_gather.cpp',
			'/pybind_modules.cpp',
		])],
		cmdclass={
			'build_ext': BuildExtension
	})
