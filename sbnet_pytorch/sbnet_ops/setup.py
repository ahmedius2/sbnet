from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sbnet',
      ext_modules=[cpp_extension.CppExtension('sbnet', ['reduce_mask.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
