from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "src.halko.shared",
        ["src/halko/shared.pyx"],
        extra_compile_args=["-fopenmp", "-O3", "-g0", '-Wno-unreachable-code'],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
