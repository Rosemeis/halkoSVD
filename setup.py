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
	version="0.2.4",
	author="Jonas Meisner",
	author_email="meisnerucph@gmail.com",
	description="Fast Python/Cython implementation of the PCAone Halko algorithm",
	long_description_content_type="text/markdown",
	long_description=open("README.md").read(),
	url="https://github.com/Rosemeis/halkoSVD",
	classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.7",
	install_requires=[
		"cython>3.0.0",
		"numpy>2.0.0",
	],
	packages=["halko"],
	entry_points={
		"console_scripts": ["halkoSVD=halko.main:main"]
	},
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
