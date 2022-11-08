from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='sbnet_module',
		ext_modules=[CUDAExtension('sbnet_module', [
			'reduce_mask_cuda.cu',
			'reduce_mask.cpp',
			'sparse_gather_cuda.cu',
			'sparse_gather.cpp',
			'pybind_modules.cpp',
		])],
		cmdclass={
			'build_ext': BuildExtension
	})
