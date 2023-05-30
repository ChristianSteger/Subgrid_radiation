# Description: Setup file
#
# Installation of package: python -m pip install .
# Installation in development mode: python -m pip install -e .
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import sys
from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np

# -----------------------------------------------------------------------------
# Manual settings
# -----------------------------------------------------------------------------

# NetCDF 3/4 library
lib_netcdf = ["libnetcdf", "libnetcdf_c++4"]  # NetCDF4
# lib_netcdf = ["libnetcdf_c++"]  # NetCDF3

# -----------------------------------------------------------------------------
# Operating system dependent settings
# -----------------------------------------------------------------------------

path_lib_conda = os.environ["CONDA_PREFIX"] + "/lib/"
if sys.platform in ["linux", "linux2"]:
    print("Operating system: Linux")
    lib_end = ".so"
    compiler = "gcc"
    extra_compile_args_cython = ["-O3", "-ffast-math", "-fopenmp"]
elif sys.platform in ["darwin"]:
    print("Operating system: Mac OS X")
    lib_end = ".dylib"
    compiler = "clang"
    extra_compile_args_cython = ["-O3", "-ffast-math",
                                 "-Wl,-rpath," + path_lib_conda,
                                 "-L" + path_lib_conda, "-fopenmp"]
elif sys.platform in ["win32"]:
    print("Operating system: Windows")
    print("Warning: Package not yet tested for Windows")
else:
    raise ValueError("Unsupported operating system")
libraries_cython = ["m", "pthread"]
include_dirs_cpp = [np.get_include()]
extra_objects_cpp = [path_lib_conda + i + lib_end for i in ["libembree3"]
                     + lib_netcdf]

# -----------------------------------------------------------------------------
# Compile Cython/C++ code
# -----------------------------------------------------------------------------

os.environ["CC"] = compiler

ext_modules = [
    Extension("shortwave_subgrid.transform",
              ["shortwave_subgrid/transform.pyx"],
              libraries=libraries_cython,
              extra_compile_args=extra_compile_args_cython,
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
    Extension("shortwave_subgrid.subsolar_lookup",
              sources=["shortwave_subgrid/subsolar_lookup.pyx", "shortwave_subgrid/subsolar_lookup_comp.cpp"],
              include_dirs=include_dirs_cpp,
              extra_objects=extra_objects_cpp,
              extra_compile_args=["-O3"],
              language="c++"),
    Extension("shortwave_subgrid.sun_position",
              sources=["shortwave_subgrid/sun_position.pyx", "shortwave_subgrid/sun_position_comp.cpp"],
              include_dirs=include_dirs_cpp,
              extra_objects=extra_objects_cpp,
              extra_compile_args=["-O3"],
              language="c++")
    ]

setup(name="shortwave_subgrid",
      version="0.1",
      packages=["shortwave_subgrid"],
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules)
