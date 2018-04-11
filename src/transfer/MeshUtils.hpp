// Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at  
// Lawrence Livermore National Laboratory. LLNL-CODE-747639. All rights        
// reserved. Please see COPYRIGHT and LICENSE for details.                     
//                                                                             
// This file is part of the ParELAGMC library. For more information and source 
// code availability see https://github.com/LLNL/parelagmc.                    
//                                                                             
// ParELAGMC is free software; you can redistribute it and/or modify it        
// under the terms of the GNU General Public License (as published by the      
// Free Software Foundation) version 2, dated June 1991.                       
 
#ifndef MESH_UTILS_HPP_
#define MESH_UTILS_HPP_ 

#include <mfem.hpp>

namespace parelagmc 
{
mfem::Element * NewElem(const int type, const int *cells_data, const int attr);
void Finalize(mfem::Mesh &mesh, const bool generate_edges);
}

#endif /* MESH_UTILS_HPP_ */
