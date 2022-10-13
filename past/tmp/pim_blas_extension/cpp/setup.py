from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pim_cpp',
      ext_modules=[cpp_extension.CppExtension('pim_cpp', ['pim.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      package_data={'pim_cpp': [
          'PIM_App/transaction_generator.cc',
          'PIM_App/transaction_generator.h',
          'pim_config.h']},
      )


