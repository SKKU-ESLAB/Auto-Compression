from setuptools import find_packages, setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

pim_utils = CppExtension(
    name="pim_cpp",
    sources=[
        "pim.cpp",
        "pim_blas.cpp",
        "pim_runtime.cpp",
        "pim_config.cpp",
        "pim_func_sim/pim_func_sim.cc",
        "pim_func_sim/pim_unit.cc",
        "pim_func_sim/pim_utils.cc",
        "../fpga_pim.c",
    ]
)

setup(
    name='pim_cpp',
    ext_modules=[pim_utils],
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages(),
    )
