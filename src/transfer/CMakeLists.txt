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
  
list(APPEND TRANSFER_SRCS
  Box.cpp
  HashGrid.cpp
  MeshUtils.cpp
  MortarAssemble.cpp
  MortarAssembler.cpp
  MortarIntegrator.cpp
  ParMortarAssembler.cpp
  )

list(APPEND TRANSFER_HDRS
  Box.hpp
  HashGrid.hpp
  MeshUtils.hpp
  MortarAssemble.hpp
  MortarAssembler.hpp
  MortarIntegrator.hpp
  ParMortarAssembler.hpp
  )

convert_filenames_to_full_paths(TRANSFER_SRCS)
convert_filenames_to_full_paths(TRANSFER_HDRS)

set(SOURCES ${SRCS} ${TRANSFER_SRCS} PARENT_SCOPE)
set(HEADERS ${HDRS} ${TRANSFER_HDRS} PARENT_SCOPE)

