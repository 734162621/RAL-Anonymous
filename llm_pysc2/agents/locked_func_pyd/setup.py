# import distutils.core
# from Cython.Build import cythonize
# from Cython.Distutils import build_ext
#
# a = cythonize("locked_func.py")[0]
# distutils.core.setup(
#     name='locked_func',
#     version="1.0",
#     ext_modules=[a],
#     author="",
# )

from distutils.core import setup
# from setuptools import setup
from Cython.Build import cythonize
# import os

setup(name='text-pysc2 locked functions', ext_modules=cythonize("main_agent_funcs.pyx", language_level="3"))


# if __name__ == "__main__":
#     os.system("python setup.py install build_ext --inplace")
