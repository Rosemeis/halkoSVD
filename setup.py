from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "halko.shared",
        ["halko/shared.pyx"],
        extra_compile_args=["-fopenmp", "-O3", "-g0", '-Wno-unreachable-code'],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

setup(
    name="halko",
    version="0.2",
    description="Fast Python/Cython implementation of the PCAone Halko algorithm",
    author="Jonas Meisner",
    packages=["halko"],
    entry_points={
        "console_scripts": ["halko=halko.main:main"]
    },
    python_requires=">=3.7",
	install_requires=[
		"cython",
		"numpy"
	],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
