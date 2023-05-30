# Shortwave_subgrid
Package to efficiently compute sub-grid correction factors for downwelling direct shortwave radiation.

# Installation

Shortwave_subgrid has been tested with **Python 3.9** (Mac OS X).
It is recommended to install dependencies via [Conda](https://docs.conda.io/en/latest/#).
Installation via **Conda** can be accomplished as follows for different platforms:

## Linux

Installation requires the [GNU Compiler Collection (GCC)](https://gcc.gnu.org). Create an appropriate Conda environment

```bash
conda create -n raytracing -c conda-forge embree3 tbb-devel cython numpy scipy xarray matplotlib cartopy netcdf4 cmcrameri skyfield
```

and **activate this environment**.
Then install the package [Utilities](https://github.com/ChristianSteger/Utilities) according to the provided instructions.
The Shortwave_subgrid package can then be installed with:

```bash
git clone https://github.com/ChristianSteger/Shortwave_subgrid.git
cd Shortwave_subgrid
python -m pip install .
```

## Mac OS X

Shortwave_subgrid is compiled with **Clang** under Mac OS X. As the Apple-provided **Clang** does not support OpenMP, an alternative **Clang** with OpenMP support has to be installed.
This can be done via Conda. Create an appropriate Conda environment

```bash
conda create -n raytracing -c conda-forge embree3 tbb-devel cython numpy scipy xarray matplotlib cartopy netcdf4 cmcrameri skyfield c-compiler openmp python=3.9
```

and **activate this environment**.
Then install the package [Utilities](https://github.com/ChristianSteger/Utilities) according to the provided instructions.
The Shortwave_subgrid package can then be installed with:

```bash
git clone https://github.com/ChristianSteger/Shortwave_subgrid.git
cd Shortwave_subgrid
python -m pip install .
```

# Usage