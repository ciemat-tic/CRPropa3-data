#!/bin/bash

export ARCH=linux-rocky9-cascadelake

# Cargar dependencias
spack load cmake%gcc@11.3.1 arch=$ARCH
spack load python@3.11.9%gcc@11.3.1 arch=$ARCH
spack load py-numpy@2.1.1%gcc@11.3.1 arch=$ARCH
spack load py-scipy@1.14.1%gcc@11.3.1 arch=$ARCH
spack load py-pandas%gcc@11.3.1 arch=$ARCH
spack load fftw@3.3.10%gcc@11.3.1 arch=$ARCH
spack load swig@4.1.1%gcc@11.3.1 arch=$ARCH
spack load hdf5@1.14.3%gcc@11.3.1 arch=$ARCH
spack load muparser@2.3.4%gcc@11.3.1 arch=$ARCH

# Usar el gcc del sistema, que ya es 11.3.1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Directorio build de CRPropa
export CRPROPA_BUILD=/home/u7644/CRPropa_versions/CRPropa_linux-rocky9-cascadelake/CRPropa3/build

# Variables para Python y librerías
export PYTHONPATH=$CRPROPA_BUILD:$PYTHONPATH
export LD_LIBRARY_PATH=$CRPROPA_BUILD:$LD_LIBRARY_PATH

echo "ARCH=$ARCH"
echo "CC=$CC"
echo "CXX=$CXX"
echo "which gcc: $(which gcc)"
echo "which g++: $(which g++)"
echo "which python: $(which python)"
echo "which cmake: $(which cmake)"
$CC --version | head -1
$CXX --version | head -1
python --version
cmake --version | head -1
