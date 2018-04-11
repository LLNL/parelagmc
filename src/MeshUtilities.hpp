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
 
#ifndef MESHUTILITIES_HPP_
#define MESHUTILITIES_HPP_

#include <memory>
#include <elag.hpp>

namespace parelagmc
{
/// Create quad/hex mesh for SPE10 dataset 
///  Mesh has N_i elements with cell size h_i in each direction.
std::unique_ptr<mfem::Mesh> Create_SPE10_Mesh( 
    const int nDimensions,
    const std::vector<int> & N, 
    const std::vector<double> & h);

/// Create embedded quad/hex mesh for SPE10 dataset 
/// Embedded mesh has N_i + 2n_i elements with cell size h_i in each direction
std::unique_ptr<mfem::Mesh> Create_Embedded_SPE10_Mesh(
    const int nDimensions,
    const std::vector<int> & N, 
    const std::vector<double> & h, 
    const std::vector<int> & n);

/// Create embedded hex mesh for the Egg Model  
/// Embedded mesh has N_i + 2n_i elements with cell size h_i in 
/// each direction where N_i = (60,60,7) and original egg mesh 
/// has h_i = (8,8,4)
std::unique_ptr<mfem::Mesh> Create_Embedded_EggModel_Mesh(
    const std::vector<double> & element_size,
    const std::vector<int> & num_added_els);

/// Create parallel meshes where pmesh has same domain_partitioning as pembedmesh
/// where domain_partitioning is generated using METIS (within mfem)
void Create_Parallel_Domain_Partitioning( const MPI_Comm & comm,
    const std::unique_ptr<mfem::Mesh> & mesh,
    const std::unique_ptr<mfem::Mesh> & embedmesh,
    std::shared_ptr<mfem::ParMesh> & pmesh,
    std::shared_ptr<mfem::ParMesh> & pembedmesh);

/// Create parallel meshes where pmesh has same domain_partitioning as pembedmesh
/// where domain_partitioning is generated using CartesianPartitioning
void Create_Parallel_Cartesian_Domain_Partitioning( const MPI_Comm & comm,
    const std::unique_ptr<mfem::Mesh> & mesh,
    const std::unique_ptr<mfem::Mesh> & embedmesh,
    std::shared_ptr<mfem::ParMesh> & pmesh,
    std::shared_ptr<mfem::ParMesh> & pembedmesh);

/// Change element attributes within eps of data points in v_obs_data_coords
/// Beware: This is wonky and unreliable.  
void ChangeMeshAttributes(mfem::ParMesh & mesh, int m,
    std::vector<double> & v_obs_data_coords, const double eps = 0.01 );

/// Shift a mesh (helpful for mesh embedding)
void ShiftMesh(std::unique_ptr<mfem::Mesh> & mesh, double dx, 
    double dy, double dz = 0.);

} /* namespace parelagmc */
#endif /* MESHUTILITIES_HPP_ */


