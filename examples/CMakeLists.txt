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

include(ParELAGMC_CMakeUtilities)

list(APPEND EXAMPLE_SRCS
  AnalyticCovarianceTest.cpp
  DarcyTest.cpp
  DarcyTest_Legacy.cpp
  EmbeddedPDESamplerTest.cpp
  EmbeddedPDESamplerTest_Legacy.cpp
  KLSampler.cpp
  MLMC.cpp
  MLMC_EmbeddedPDESampler.cpp
  PDESamplerTest.cpp
  PDESamplerTest_Legacy.cpp
)

if (ParELAGMC_ENABLE_ParMoonolith)
  list(APPEND EXAMPLE_SRCS
    ComputeReferenceObservationalData.cpp
    DarcyTest_Legacy_RandomInput.cpp
    DarcyTest_RandomInput.cpp
    LikelihoodExample.cpp
    MLMC_Manual.cpp
    MLMC_ProjectionPDESampler.cpp
    ProjectionPDESamplerTest.cpp
    ProjectionPDESamplerTest_Legacy.cpp
    RatioEstimator_MC.cpp
    RatioEstimator_MC_Manager.cpp
    RatioEstimator_MC_Sampling_Error.cpp
    RatioEstimator_MC_Spatial_Error.cpp
    RatioEstimator_MLMC.cpp
    RatioEstimator_MLMC_Manager.cpp
    RealizationTest.cpp
    SamplerTest.cpp
    SLMC.cpp
    SLMC_ProjectionPDESampler.cpp
  )
endif() #ParELAGMC_ENABLE_ParMoonolith

add_parelagmc_executables(EXAMPLE_SRCS)

file(COPY example_parameterlists/example_parameters.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY example_parameterlists/bayes_square_parameters.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (ParELAGMC_BUILD_SPE10_EXAMPLES)
    add_subdirectory(SPE10)
endif()

if (${PROJECT_NAME}_ENABLE_TESTING)
  message(STATUS "Adding ctest examples")
  # Mixed Darcy test with deterministic coefficient
  add_test(DarcyDeterministicTest DarcyTest.exe)
  set_tests_properties(DarcyDeterministicTest
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "0  2         17152 ; 1  2         2240; 2  2         304")

  # SPDE sampler with matching mesh embedding 
  add_test(PDESamplerTest_MatchingMeshEmbedding EmbeddedPDESamplerTest.exe)
  set_tests_properties(PDESamplerTest_MatchingMeshEmbedding
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "1.1226[0-9]*; 9.0325[0-9]*; 5.1372[0-9]*")
  
  # MLMC simulation with SPDE sampler without mesh embedding 
  add_test(MLMC_PDESampler MLMC.exe)
  set_tests_properties(MLMC_PDESampler
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "FINAL MLMC ERRORS; Estimate                                  2.5599[0-9]*")

  # SPDE sampler without mesh embedding 
  add_test(PDESamplerTest PDESamplerTest.exe)
  set_tests_properties(PDESamplerTest
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "1.2593[0-9]*; 9.3103[0-9]*; 6.3853[0-9]* ")
  
  if (ParELAGMC_ENABLE_ParMoonolith)
    # Mixed Darcy test with random input coefficient
    add_test(DarcyRandomInputTest DarcyTest_RandomInput.exe)
    set_tests_properties(DarcyRandomInputTest
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "0  2.391[0-9]*   17152; 1  2.103[0-9]*   2240; 2  1.998[0-9]*   304")

    # BIP likelihood evaluation 
    add_test(BayesianInverseProblem_LikelihoodEvaluation LikelihoodExample.exe)
    set_tests_properties(BayesianInverseProblem_LikelihoodEvaluation
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "L = 0 : 0.9279[0-9]*; L = 1 : 0.9578[0-9]*; L = 2 : 0.9269[0-9]*")
    
    # SPDE sampler with non-matching mesh embedding 
    add_test(PDESamplerTest_NonMatchingMeshEmbedding ProjectionPDESamplerTest.exe)
    set_tests_properties(PDESamplerTest_NonMatchingMeshEmbedding
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "1.1226[0-9]*; 9.0325[0-9]*; 5.1372[0-9]*")
    
    #RatioEstimator_MC.cpp
    # BIP ratio estimator with MC 
    add_test(BayesianInverseProblem_MC_RatioEstimator RatioEstimator_MC.exe)
    set_tests_properties(BayesianInverseProblem_MC_RatioEstimator
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "0 [ ]*  1.987[0-9 ]*  0.07749[0-9 ]* 0.8569[0-9 ]*  0.009691[0-9 ]* 2.319[0-9 ]*  2.332[0-9 ]*")
  endif() #ParELAGMC_ENABLE_ParMoonolith

endif (${PROJECT_NAME}_ENABLE_TESTING)
