from setuptools import setup, Extension
import numpy as np
import os


MKL_ROOT = os.getenv("MKLROOT", "/opt/intel/oneapi/mkl/latest")
MKL_LIB_DIR = f"{MKL_ROOT}/lib/intel64"
MKL_INCLUDE_DIR = f"{MKL_ROOT}/include"

os.environ["LD_LIBRARY_PATH"] = f"{MKL_LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"

module = Extension(
    'extreme_matmul',
    sources=['extreme_matmul.c'],
    include_dirs=[np.get_include(), MKL_INCLUDE_DIR],
    libraries=['mkl_rt'],
    library_dirs=[MKL_LIB_DIR],
    extra_compile_args=['-O3', '-march=native', '-fopenmp'],
    extra_link_args=['-Wl,--no-as-needed', '-lmkl_rt', '-liomp5', '-lpthread', '-lm', '-ldl'],
)

setup(
    name='extreme_matmul',
    version='1.0',
    description='MKL-accelerated matrix multiplication',
    ext_modules=[module],
)