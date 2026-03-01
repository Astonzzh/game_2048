from setuptools import setup
from Cython.Build import cythonize

setup(
    name="Fast Bitboard",
    ext_modules=cythonize("bitboard.pyx", compiler_directives={'language_level': "3"})
)