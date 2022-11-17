from setuptools import find_packages, setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

flush_utils = CppExtension(
    name="flush_cpp",
    sources=[
        "flush.cpp",
        "mm_flush.cpp"
    ]
)

setup(
    name='flush_cpp',
    ext_modules=[flush_utils],
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
    )
