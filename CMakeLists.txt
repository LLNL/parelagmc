# Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at  
# Lawrence Livermore National Laboratory. LLNL-CODE-747639. All rights        
# reserved. Please see COPYRIGHT and LICENSE for details.               
#                                                                             
# This file is part of the ParELAGMC library. For more information and source 
# code availability see https://github.com/LLNL/parelagmc.                    
#                                                                             
# ParELAGMC is free software; you can redistribute it and/or modify it        
# under the terms of the GNU General Public License (as published by the      
# Free Software Foundation) version 2, dated June 1991. 

message("")
message("Welcome to ParELAGMC!")
message("")


# Some features here won't work if version < 3.1.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Project name
set(PROJECT_NAME ParELAGMC)
project(${PROJECT_NAME})

# RPATH on MACOSX
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

# Set version
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 1)

string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UC)

#
# Important paths/variables
#
set(HOME_DIR $ENV{HOME})
set(${PROJECT_NAME}_CMAKE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${${PROJECT_NAME}_CMAKE_PATH}/modules)

# Enforce C11
if ((NOT CMAKE_C_STANDARD) OR (CMAKE_C_STANDARD LESS 11))
  set(CMAKE_C_STANDARD 11)
endif()
set(CMAKE_C_STANDARD_REQUIRED ON)

message(STATUS "Using C standard: c${CMAKE_C_STANDARD}")

# Enforce C++11
if ((NOT CMAKE_CXX_STANDARD) OR (CMAKE_CXX_STANDARD LESS 11))
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using CXX standard: c++${CMAKE_CXX_STANDARD}")

#
# Development flags
#
option(${PROJECT_NAME}_DEVELOPER_BUILD
  "Enable extra compiler flags for development."
  OFF)
if (${PROJECT_NAME}_DEVELOPER_BUILD)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR
      ${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang" OR
      ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
  endif ()
endif()

# Whether to build the examples
option(${PROJECT_NAME}_BUILD_EXAMPLES
  "Build the examples/ directory"
  ON)

# Whether to build the SPE10 examples
option(${PROJECT_NAME}_BUILD_SPE10_EXAMPLES
  "Build the examples/SPE10 directory"
  ON)

# Whether to enable the testing framework
option(${PROJECT_NAME}_ENABLE_TESTING
  "Enable the Ctest framework"
  ON)

if (${PROJECT_NAME}_ENABLE_TESTING)
  # Need the examples/testsuite if tests are enabled
  set(${PROJECT_NAME}_BUILD_EXAMPLES ON)
endif()


#
# Options for third-party libraries
#
option(${PROJECT_NAME}_ENABLE_ParMoonolith
  "Should ParMoonolith library be enabled for parallel non-matching mesh transfer"
  ON)


# On massively distributed systems on which it may be best/required to
# use static executables, this option should be enabled to "force"
# compilers to link everything as statically as possible.
option(${PROJECT_NAME}_FORCE_STATIC_LINKAGE
  "Whether to (attempt to) build static executables for examples and tests"
  OFF)

if (${PROJECT_NAME}_FORCE_STATIC_LINKAGE)
  # Find .a before .so. This matters A LOT for Vulcan
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)

  # Disables trying to build any shared library
  set(BUILD_SHARED_LIBS OFF)

  # At least strongly suggest to a compiler to use static linkage by
  # default. This works, for example, for getting GNU to link libc
  # statically.
  set(CMAKE_LINK_SEARCH_START_STATIC ON)
endif (${PROJECT_NAME}_FORCE_STATIC_LINKAGE)

#
# Initial configuration of linked libraries and include directories
#

# This is a list of TPLs that are used by all targets
set(TPL_LIBRARIES "")

# This is a list of linker flags to be used with TPLs for all targets
set(TPL_LINKER_FLAGS "")

include_directories("${PROJECT_SOURCE_DIR}/src")

# Configure file gets written to the binary directory
include_directories("${PROJECT_BINARY_DIR}/src")

#
# THIRD-PARTY LIBRARIES (TPLs)
#
# Required third-party libraries:
#   1. ParELAG (MPI, BLAS/LAPACK, HYPRE, MFEM, METIS, SUITESPARSE) 
#   2. TRNG 
# Optional (but encouraged) third-party library:
#   3. ParMoonolith

#
# REQUIRED LIBRARIES
#

find_package(ParELAG REQUIRED NO_MODULE)
include_directories(${ParELAG_INCLUDE_DIRS})

find_package(TRNG REQUIRED)
include_directories(${TRNG_INCLUDE_DIRS})

#
# Build the TPL_LIBRARIES LIST
#
list(APPEND TPL_LIBRARIES ${ParELAG_LIBRARIES})
list(APPEND TPL_LIBRARIES ${TRNG_LIBRARIES})

#
# OPTIONAL LIBRARIES
#
if(${PROJECT_NAME}_ENABLE_ParMoonolith)
  find_package(ParMoonolith REQUIRED)
  if (ParMoonolith_FOUND)
    include_directories(${MOONOLITH_INCLUDES})
    list(APPEND TPL_LIBRARIES ${MOONOLITH_LIBRARIES})
  else()
    set(${PROJECT_NAME}_ENABLE_ParMoonolith OFF)
    message(STATUS "** ParMoonolith NOT FOUND (OR NOT SUCCESSFULLY INSTALLED); ParMoonolith will be disabled **")
  endif()
else()
  message(STATUS "** ParMoonolith is NOT Enabled! **")
endif()

# Cleanup the TPL list
list(REMOVE_DUPLICATES TPL_LIBRARIES)

#
# Add the subdirectories
#

include(ParELAGMC_CMakeUtilities)
if (${PROJECT_NAME}_ENABLE_TESTING)
  enable_testing()

  function(add_mpi_test name procs exe_target)
    add_test(
      NAME ${name}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${procs} ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${exe_target}> ${MPIEXEC_POSTFLAGS} ${ARGN})
  endfunction(add_mpi_test)
 
  # style test - no tabs
  add_test(notabs
    grep -rP --include={*.cpp,*.hpp,*.c,*.h} '\t' ${PROJECT_SOURCE_DIR})
  set_tests_properties(notabs PROPERTIES WILL_FAIL "TRUE")

endif()

# The primary source directory for the library.
add_subdirectory(src)

# Build the examples directory, if requested
if (${PROJECT_NAME}_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#
# Documentation target
#
add_subdirectory(doc)

#
# CONFIGURATION FILE
#
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}_Config.h.in"
  "${PROJECT_BINARY_DIR}/src/${PROJECT_NAME}_Config.h"
  )

#
# INSTALL TARGET
#

# Library install command in src/CMakeLists.txt
# All the export stuff is either in src/CMakeLists.txt or cmake/CMakeLists.txt

# Build directory export
export(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE parelagmc::
  FILE ${PROJECT_NAME}Targets.cmake)

# Add the build tree to package registry
export(PACKAGE ${PROJECT_NAME})

# Handle the rest of the export stuff
add_subdirectory(cmake)

# Install the headers
install(DIRECTORY src/ DESTINATION include/parelagmc
  FILES_MATCHING PATTERN "*.hpp")
install(FILES "${PROJECT_BINARY_DIR}/src/${PROJECT_NAME}_Config.h"
  DESTINATION include/parelagmc)

