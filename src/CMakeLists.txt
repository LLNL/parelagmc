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
  
#
# Add all sources
#
list(APPEND SRCS
    AnalyticExponentialCovariance.cpp
    BayesianInverseProblem.cpp
    DarcySolver.cpp
    DarcySolver_Legacy.cpp
    EmbeddedPDESampler.cpp
    EmbeddedPDESampler_Legacy.cpp
    KLSampler.cpp
    MaternCovariance.cpp
    MC_Manager.cpp
    MeshUtilities.cpp
    MLMC_Manager.cpp
    NormalDistributionSampler.cpp
    PDESampler.cpp
    PDESampler_Legacy.cpp
    UniformDistributionSampler.cpp
    Utilities.cpp
)

list(APPEND HDRS
    AnalyticExponentialCovariance.hpp
    BayesianInverseProblem.hpp
    CovarianceFunction.hpp
    DarcySolver.hpp
    DarcySolver_Legacy.hpp
    EmbeddedPDESampler.hpp
    EmbeddedPDESampler_Legacy.hpp
    KLSampler.hpp
    MaternCovariance.hpp
    MC_Manager.hpp
    MeshUtilities.hpp
    ML_BayesRatio_Manager.hpp
    ML_BayesRatio_Splitting_Manager.hpp
    MLMC_Manager.hpp
    MLSampler.hpp
    NormalDistributionSampler.hpp
    PDESampler.hpp
    PDESampler_Legacy.hpp
    PhysicalMLSolver.hpp
    Sampling_Method_Manager.hpp
    SL_BayesRatio_Manager.hpp
    SL_BayesRatio_Splitting_Manager.hpp
    UniformDistributionSampler.hpp
    Utilities.hpp
)

if (ParELAGMC_ENABLE_ParMoonolith)
    add_subdirectory(transfer)
    list(APPEND SRCS
        L2ProjectionPDESampler.cpp
        L2ProjectionPDESampler_Legacy.cpp
    )
    list(APPEND HDRS
        L2ProjectionPDESampler.hpp
        L2ProjectionPDESampler_Legacy.hpp
    )
endif() #ParELAG_ENABLE_ParMoonolith

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS})
set(HEADERS ${HEADERS} ${HDRS})

# Create the ParELAGMC library from the collected sources
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Add the TPL_LIBRARIES dependency
target_link_libraries(${PROJECT_NAME} PUBLIC ${TPL_LIBRARIES} MPI::MPI_CXX)

# This seems like the best, albeit funky to read, way to ensure safely
# adding a compiler option...
target_compile_options(${PROJECT_NAME} PUBLIC
  $<$<CONFIG:Debug>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-O0>>)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/parelagmc>
  )

#
# Install target for the library
#
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include/parelagmc
  )

install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE parelagmc::
  DESTINATION lib/cmake/parelagmc
  )



