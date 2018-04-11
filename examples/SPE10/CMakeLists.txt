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

list(APPEND SPE10_EXAMPLE_SRCS
  SPE10_DarcyTest.cpp
  SPE10_EmbeddedPDESampler.cpp
  SPE10_EmbeddedPDESampler_Performance.cpp
  SPE10_MLMC.cpp
  SPE10_MLMC_EmbeddedPDESampler.cpp
  SPE10_PDESampler.cpp
  SPE10_PDESampler_Performance.cpp
  SPE10_SamplerTest.cpp
)    

if (ParELAGMC_ENABLE_ParMoonolith)
  list(APPEND SPE10_EXAMPLE_SRCS
    SPE10_DarcyTest_RandomInput.cpp
    SPE10_ProjectionPDESampler.cpp
    SPE10_ProjectionPDESampler_Performance.cpp
    SPE10_MLMC_ProjectionPDESampler.cpp
  )
endif() #ParELAGMC_ENABLE_ParMoonolith

add_parelagmc_executables(SPE10_EXAMPLE_SRCS)

file(COPY ../example_parameterlists/spe10_3D_parameters.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


