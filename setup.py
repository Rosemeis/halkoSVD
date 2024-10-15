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
    name="halkoSVD",
    version="0.2.1",
    description="Fast Python/Cython implementation of the PCAone Halko algorithm",
    author="Jonas Meisner",
    packages=["halko"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
		"Development Status :: 3 - Alpha"
    ],
    entry_points={
        "console_scripts": ["halkoSVD=halko.main:main"]
    },
    python_requires=">=3.7",
	install_requires=[
		"cython>=3.0.0",
		"numpy>=1.26.0"
	],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
