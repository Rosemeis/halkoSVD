from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "halko.shared_cy",
        ["halko/shared_cy.pyx"],
        extra_compile_args=["-fopenmp", "-g0"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="Halko CPU/GPU",
    version="0.1",
    description="CPU/GPU implementations of the PCAone Halko algorithm",
    author="Jonas Meisner",
    packages=["halko"],
    entry_points={
        "console_scripts": ["halko=halko.halkoMain:main"]
    },
    python_requires=">=3.6",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
