#!/bin/sh

CONFIG_NAME=$(basename $0 .sh)

# This is the path to the root of the git repo; i.e. the directory
# that has src/, examples/, cmake/, Documentation/, etc.
ParELAGMC_BASE_DIR=${HOME}/Code/parelagmc

# THIS SHOULD *NOT* BE ${ParELAGMC_BASE_DIR}!
ParELAGMC_BUILD_DIR=${HOME}/build/parelagmc-release 

EXTRA_ARGS=$@

if [ ! -e $ParELAGMC_BUILD_DIR ]; then
    mkdir -p $ParELAGMC_BUILD_DIR
fi
cd $ParELAGMC_BUILD_DIR

# Force a complete reconfigure; if this is not the desired behavior,
# comment out these two lines.
rm CMakeCache.txt
rm -rf CMakeFiles


cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_CXX_COMPILER=/home/osborn9/.sarah/local/gcc/7.2.0/bin/g++ \
    -DCMAKE_C_COMPILER=/home/osborn9/.sarah/local/gcc/7.2.0/bin/gcc \
    -DMPI_C_COMPILER=/home/osborn9/.sarah/local/gcc/7.2.0/mpich/3.2/bin/mpicc \
    -DMPI_CXX_COMPILER=/home/osborn9/.sarah/local/gcc/7.2.0/mpich/3.2/bin/mpicxx \
    -DParELAGMC_ENABLE_ParMoonolith=YES \
    -DParELAG_DIR=${HOME}/build/parelag \
    -DTRNG_DIR=/optional/path/to/my/trng/otherwise/automatically/installed \
    -DParMoonolith_DIR=/optional/path/to/my/moonolith/otherwise/automatically/installed \
    ${EXTRA_ARGS} \
    ${ParELAGMC_BASE_DIR}
