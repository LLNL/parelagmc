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
 
#include "MeshUtilities.hpp"
#include "Utilities.hpp"

namespace parelagmc
{

using namespace mfem;
using namespace parelag;

std::unique_ptr<Mesh> Create_SPE10_Mesh(
        const int nDimensions, 
        const std::vector<int> & N,
        const std::vector<double> & h)
{
    std::unique_ptr<Mesh> mesh;
    if (nDimensions == 3)
        mesh = make_unique<Mesh>(N[0], N[1], N[2], Element::HEXAHEDRON, 1,
                                     N[0]*h[0],N[1]*h[1],
                                     N[2]*h[2]);
    else
        mesh = make_unique<Mesh>(N[0],
                                 N[1],
                                 Element::QUADRILATERAL,
                                 1, N[0]*h[0], N[1]*h[1]);
    return mesh;
}

std::unique_ptr<Mesh> Create_Embedded_SPE10_Mesh(
        const int nDimensions,    
        const std::vector<int> & N,
        const std::vector<double> & h,
        const std::vector<int> & n)
{
    const int embed_nx = N[0] + 2*n[0];
    const int embed_ny = N[1] + 2*n[1];
    int embed_nz;
    if (nDimensions == 3) embed_nz = N[2] + 2*n[2];
    
    std::unique_ptr<Mesh> embedmesh;
    // Create extended mesh
    if (nDimensions == 3)
        embedmesh = make_unique<Mesh>(embed_nx, embed_ny, embed_nz,
            Element::HEXAHEDRON, 1,
            embed_nx*h[0],embed_ny*h[1],embed_nz*h[2]);
    else
        embedmesh = make_unique<Mesh>(embed_nx, embed_ny,
                                 Element::QUADRILATERAL,
                                 1, embed_nx*h[0], embed_ny*h[1]);
    // Shift mesh
    Vector displacement;
    const int nv = embedmesh->GetNV();
    if (nDimensions == 3)
        displacement.SetSize(3*nv);
    else
        displacement.SetSize(2*nv);
    for (int i = 0; i < nv; i++)
    {
        displacement(i) = -n[0]*h[0];
        displacement(i + nv) = -n[1]*h[1];
        if (nDimensions == 3)
            displacement(i + 2*nv) = -n[2]*h[2];
    }
    embedmesh->MoveVertices(displacement);

    // Set "active" cells to have attr=1, else attr=2 in original mesh
    const int embed_num_xy = embed_nx*embed_ny;
    if (nDimensions == 3)
    {
       
        for (int i_z = 0; i_z < embed_nz; i_z++)
        {
            // lower z embedded layers
            if ( i_z < n[2])
            {
                for (int j = 0; j < embed_num_xy; j++)
                    embedmesh->GetElement( j + i_z*embed_num_xy )->SetAttribute(2);
            }

            else if ( i_z >= N[2] + n[2])
            {
                for (int j = 0; j < embed_num_xy; j++)
                    embedmesh->GetElement( j + i_z*embed_num_xy )->SetAttribute(2);
            }
            else // interior z
            {
                // loop over y 
                for (int i_y = 0; i_y < embed_ny; i_y++)
                {
                    if (i_y < n[1] || i_y >= N[1] + n[1])
                    {
                        for (int i_x = 0; i_x < embed_nx; i_x++)
                        {
                            embedmesh->GetElement(i_x + embed_nx*i_y + embed_num_xy*i_z)
                                    ->SetAttribute(2);
                        }
                    }
                    else // interior y  
                    {
                        int num_active = N[0];
                        for (int i_x = 0; i_x < embed_nx; i_x++)
                        {
                            if (i_x < n[0] || i_x >= num_active + n[0])
                            {
                                embedmesh->GetElement(i_x + embed_nx*i_y + embed_num_xy*i_z)->SetAttribute(2);
                            }
                            else
                                embedmesh->GetElement( i_x + embed_nx*i_y + embed_num_xy*i_z)->SetAttribute(1);
                        }
                    }
                }
            }
        }
    } 
    else // dim == 2
    { 
        // loop over y 
            for (int i_y = 0; i_y < embed_ny; i_y++)
            {
                if (i_y < n[1] || i_y >= N[1] + n[1])
                {
                    for (int i_x = 0; i_x < embed_nx; i_x++)
                    {
                        embedmesh->GetElement(i_x + embed_nx*i_y )
                                ->SetAttribute(2);
                    }
                }
                else // interior y  
                {
                    int num_active = N[0];
                    for (int i_x = 0; i_x < embed_nx; i_x++)
                    {
                        if (i_x < n[0] || i_x >= num_active + n[0])
                        {
                            embedmesh->GetElement(i_x + embed_nx*i_y )->SetAttribute(2);
                        }
                        else
                            embedmesh->GetElement( i_x + embed_nx*i_y )->SetAttribute(1);
                    }
                }
            }

    }
    return embedmesh;
}

std::unique_ptr<Mesh> Create_Embedded_EggModel_Mesh(
        const std::vector<double> & element_size,
        const std::vector<int> & num_embedded_els)
{
    const int embed_nx = 60 + 2*num_embedded_els[0];
    const int embed_ny = 60 + 2*num_embedded_els[1];
    const int embed_nz = 7 + 2*num_embedded_els[2];
    
    std::unique_ptr<Mesh> embedmesh;
    embedmesh = make_unique<Mesh>(embed_nx, embed_ny, embed_nz,
        Element::HEXAHEDRON, 1,
        embed_nx*element_size[0],embed_ny*element_size[1],embed_nz*element_size[2]);
    
    // Shift mesh
    Vector displacement;
    const int nv = embedmesh->GetNV();
    displacement.SetSize(3*nv);
    
    for (int i = 0; i < nv; i++)
    {
        displacement(i) = -num_embedded_els[0]*element_size[0];
        displacement(i + nv) = -num_embedded_els[1]*element_size[1];
        displacement(i + 2*nv) = -num_embedded_els[2]*element_size[2];
    }
    embedmesh->MoveVertices(displacement);

    return embedmesh;
}

void Create_Parallel_Domain_Partitioning( const MPI_Comm & comm,
    const std::unique_ptr<mfem::Mesh> & mesh,
    const std::unique_ptr<mfem::Mesh> & embedmesh,
    std::shared_ptr<mfem::ParMesh> & pmesh,
    std::shared_ptr<mfem::ParMesh> & pembedmesh)
{
        int num_procs;
        MPI_Comm_size(comm, &num_procs);
        std::unique_ptr<int[]> embed_domain_partitioning_data;
        embed_domain_partitioning_data.reset(
            embedmesh->GeneratePartitioning(num_procs,1));
        Array<int> embed_domain_partitioning(embed_domain_partitioning_data.get(),
            embedmesh->GetNE());
        pembedmesh = std::make_shared<ParMesh>(comm, *embedmesh, embed_domain_partitioning);

        // Use same domain_partitioning for original mesh
        Array<int> domain_partitioning(mesh->GetNE());
        domain_partitioning = 0;
        int count = 0;
        for (int i = 0; i < embedmesh->GetNE(); i++)
        {
            if (embedmesh->GetAttribute(i) == 1)
            {
                // Trouble if number of els with attr==1 > mesh->GetNE()
                PARELAG_TEST_FOR_EXCEPTION(
                    count > mesh->GetNE(),
                    std::runtime_error,
                    "Create_Parallel_Domain_Partitioning: All elements of" 
                    << " original mesh must have attribute == 1 in embedded mesh," 
                    << " and all other elements attribute > 1."); 
                domain_partitioning[count] = embed_domain_partitioning[i];
                count++;
            }
        }
        domain_partitioning.MakeDataOwner();
        pmesh = std::make_shared<ParMesh>(comm, *mesh, domain_partitioning);
}

void Create_Parallel_Cartesian_Domain_Partitioning( const MPI_Comm & comm,
    const std::unique_ptr<mfem::Mesh> & mesh,
    const std::unique_ptr<mfem::Mesh> & embedmesh,
    std::shared_ptr<mfem::ParMesh> & pmesh,
    std::shared_ptr<mfem::ParMesh> & pembedmesh)
{
        int num_procs;
        MPI_Comm_size(comm, &num_procs);

        Array<int> n_proc(3);
        n_proc[0] = 1;
        n_proc[1] = num_procs;
        n_proc[2] = 1;

        std::unique_ptr<int[]> embed_domain_partitioning_data;
        embed_domain_partitioning_data.reset(
            embedmesh->CartesianPartitioning(n_proc.GetData()));
        Array<int> embed_domain_partitioning(embed_domain_partitioning_data.get(),
            embedmesh->GetNE());
        pembedmesh = std::make_shared<ParMesh>(comm, *embedmesh, embed_domain_partitioning);

        // Use same domain_partitioning for original mesh
        Array<int> domain_partitioning(mesh->GetNE());
        domain_partitioning = 0;
        int count = 0;
        for (int i = 0; i < embedmesh->GetNE(); i++)
        {
            if (embedmesh->GetAttribute(i) == 1)
            {
                // Trouble if number of els with attr==1 > mesh->GetNE()
                PARELAG_TEST_FOR_EXCEPTION(
                    count > mesh->GetNE(),
                    std::runtime_error,
                    "Create_Parallel_Cartesian_Domain_Partitioning: All elements of " <<  
                    "original mesh must have attribute == 1 in embedded mesh, " 
                    << "and all other elements attribute > 1."); 
                domain_partitioning[count] = embed_domain_partitioning[i];
                count++;
            }
        }
        domain_partitioning.MakeDataOwner();
        pmesh = std::make_shared<ParMesh>(comm, *mesh, domain_partitioning);
}

void ChangeMeshAttributes(ParMesh & mesh, int m,
    std::vector<double> & v_obs_data_coords, const double eps )
{
    const int dim = mesh.Dimension();
    Element * el;
    Array<int> verts;
    double * coords;

    // This is not the best way to loop. Should do elements then obs points, but
    // there's trouble if a obs_point is on an edge or is a vertice 
    // loop over elements of mesh
    double min_x, min_y, min_z, max_x, max_y, max_z;
    double x,y,z;
    // Loop over the elements in the mesh. If observational data point 
    // obs_j is within the element or within h of the element 
    // loop over observational data points

    for (int j=0; j<m; j++)
    {
        for (int e=0; e < mesh.GetNE(); e++)
        {
            el = mesh.GetElement(e);
            el->GetVertices(verts);
            coords = mesh.GetVertex(verts[0]);
            // Determine x,y,z range 
            min_x = coords[0];
            max_x = coords[0];
            min_y = coords[1];
            max_y = coords[1];
            min_z = 0.; max_z = 0.;
            if (dim == 3)
            {
                min_z = coords[2];
                max_z = coords[2];
            }
            for (int i=1; i < verts.Size(); i++)
            {
                coords = mesh.GetVertex(verts[i]);
                if (coords[0]>max_x) max_x = coords[0];
                if (coords[0]<min_x) min_x = coords[0];
                if (coords[1]>max_y) max_y = coords[1];
                if (coords[1]<min_y) min_y = coords[1];
                if (dim == 3) if (coords[2]>max_z) max_z = coords[2];
                if (dim == 3) if (coords[2]<min_z) min_z = coords[2];
            }
            x = v_obs_data_coords[0 + dim*j];
            y = v_obs_data_coords[1 + dim*j];
            if (dim == 2)
            {
                if ( ((min_x-eps) <= x) && (x < (max_x+eps))
                    && ((min_y-eps) <= y) && (y < max_y+eps))
                {
                    el->SetAttribute(j+2);
                }
            }
            else
            {
                z = v_obs_data_coords[2 + dim*j];
                if ( ((min_x-eps) <= x) && (x < (max_x+eps))
                    && ((min_y-eps) <= y) && (y < (max_y+eps))
                    && ((min_z-eps) <= z) && (z < (max_z+eps)) )
                {
                    el->SetAttribute(j+2);
                }
            }
        }
    }
}

void ShiftMesh(std::unique_ptr<mfem::Mesh> & mesh, double dx, 
    double dy, double dz)
{
    const int nDim = mesh->Dimension();
    Vector displacement;
    if (nDim == 3)
        displacement.SetSize(3*mesh->GetNV());
    else
        displacement.SetSize(2*mesh->GetNV());
    for (int i = 0; i < mesh->GetNV(); i++)
    {
        displacement(i) = dx;
        displacement(i + mesh->GetNV()) = dy;
        if (nDim == 3)
            displacement(i + 2*mesh->GetNV()) = dz;
    }

    mesh->MoveVertices(displacement);
}

} /* namespace parelagmc */


