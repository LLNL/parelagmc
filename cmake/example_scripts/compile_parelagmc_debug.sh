#!/bin/sh

CONFIG_NAME=$(basename $0 .sh)

# This is the path to the root of the git repo; i.e. the directory
# that has src/, examples/, cmake/, Documentation/, etc.
ParELAGMC_BASE_DIR=/path/to/my/parelagmc

# THIS SHOULD *NOT* BE ${ParELAGMC_BASE_DIR}!
ParELAGMC_BUILD_DIR=/path/to/my/build/dir

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
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DParELAG_DIR=/path/to/my/parelag/build/or/install \
    -DTRNG_DIR=/optional/path/to/my/trng/otherwise/automatically/installed \
    -DParELAGMC_ENABLE_ParMoonolith=/YES/NO \
    -DParMoonolith_DIR=/optional/path/to/my/moonolith/otherwise/automatically/installed \
    ${EXTRA_ARGS} \
    ${ParELAGMC_BASE_DIR}
