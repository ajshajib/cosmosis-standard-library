# - GSL_INC the path to GSL header files
# - GSL_LIB the path to GSL library files
# - CFITSIO_INC the path to CFTSIO header files
# - CFITSIO_LIB the path to CFTSIO library files
# - FFTW_LIBRARY the path to FFTW header files
# - FFTW_INCLUDE_DIR the path to FFTW library files
# - LAPACK_LINK whatever command line you need to link to LAPACK
# - CXX Command for your C++ compiler
# - CC Command for your C compiler
# - FC Command for your Fortran compiler
# - MPIFC Command for your MPI Fortran compiler
# - COSMOSIS_ALT_COMPILERS=1

module load gsl/2.7
export GSL_INC=$GSL_DIR/include
export GSL_LIB=$GSL_DIR/lib

module load cfitsio/3.490
export CFITSIO_INC=$CFITSIO_DIR/include
export CFITSIO_LIB=$CFITSIO_DIR/lib

module load fftw3/3.3.9
export FFTW_LIBRARY=$FFTW3_DIR/lib
export FFTW_INCLUDE_DIR=$FFTW3_DIR/include

module load lapack/3.10.0
export LAPACK_LINK="-llapack -lblas"

export CXX=g++
export CC=gcc
export FC=gfortran
export MPIFC=mpif90

export COSMOSIS_ALT_COMPILERS=1





