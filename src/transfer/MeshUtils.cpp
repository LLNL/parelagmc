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
 
#include "MeshUtils.hpp"
#include <assert.h>

namespace parelagmc 
{
using namespace mfem;
Element * NewElem(const int type, const int *cells_data, const int attr)
{
    switch (type)
    {
        case Geometry::TRIANGLE:      return new Triangle(cells_data, attr);
        case Geometry::SQUARE:        return new Quadrilateral(cells_data, attr);
        case Geometry::CUBE:          return new Hexahedron(cells_data, attr);
        case Geometry::TETRAHEDRON:   return new Tetrahedron(cells_data, attr);
        
        default: 
        {
            assert(false && "unknown type");
            return nullptr;
        }
    }
}

void Finalize(Mesh &mesh, const bool generate_edges)
{
    //based on the first element
    int type = mesh.GetElement(0)->GetGeometryType();

    switch (type)
    {
        case Geometry::TRIANGLE:      return mesh.FinalizeTriMesh(generate_edges);
        case Geometry::SQUARE:        return mesh.FinalizeQuadMesh(generate_edges);
        case Geometry::CUBE:          return mesh.FinalizeHexMesh(generate_edges);
        case Geometry::TETRAHEDRON:   return mesh.FinalizeTetMesh(generate_edges);

        default: 
        {
            assert(false && "unknown type");
            return;
        }
    }
}
   
} /* namespace parelagmc */
