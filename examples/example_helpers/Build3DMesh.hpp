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
 
#ifndef BUILD3DMESH_HPP__
#define BUILD3DMESH_HPP__

#include <elag.hpp>

#include "MeshUtilities.hpp"

namespace parelagmc
{
namespace examplehelpers
{

std::unique_ptr<mfem::Mesh> Build3DHexMesh()
{
    return parelag::make_unique<mfem::Mesh>(4,4,4,mfem::Element::HEXAHEDRON,
        1, 2.0, 2.0, 2.0);
}

std::unique_ptr<mfem::Mesh> Build3DHexEnlargedMesh()
{
    auto mesh = parelag::make_unique<mfem::Mesh>(6,6,6,
        mfem::Element::HEXAHEDRON,1, 3.0, 3.0, 3.0);
    ShiftMesh(mesh, -0.5, -0.5, -0.5);
    return mesh;
}

std::unique_ptr<mfem::Mesh> Build3DHexEmbeddedMesh()
{
    auto mesh = parelag::make_unique<mfem::Mesh>(6,6,6,
        mfem::Element::HEXAHEDRON,1, 3.0, 3.0, 3.0);
    ShiftMesh(mesh, -0.5, -0.5, -0.5);

    // Alter mesh attributes so that original mesh 
    // has attribute=2
    mfem::Array<int> vertex;
    double * coords;
    // lower 
    int count;
    for (int i = 0; i < mesh->GetNE(); i++)
    {
        count = 0;
        mesh->GetElement(i)->GetVertices(vertex);
        // loop over each dimension
        for (int j = 0; j < 3; j++)
        {
            for (int v = 0; v < vertex.Size(); v++)
            {
                coords = mesh->GetVertex(vertex[v]);
                if (coords[j] == -0.5)
                    count += 1;
            }
            if (count > 2)
                mesh->GetElement(i)->SetAttribute(2);
        }
    }    
    // upper 
    for (int i = 0; i < mesh->GetNE(); i++)
    {
        count = 0;
        mesh->GetElement(i)->GetVertices(vertex);
        // loop over each dimension
        for (int j = 0; j < 3; j++)
        {
            for (int v = 0; v < vertex.Size(); v++)
            {
                coords = mesh->GetVertex(vertex[v]);
                if (coords[j] == 2.5)
                    count += 1;
            }
            if (count > 2)
                mesh->GetElement(i)->SetAttribute(2);
        }
    }
    return mesh;
}
}// namespace examplehelpers
}// namespace parelagmc
#endif /* BUILD3DHEXMESH_HPP__ */
