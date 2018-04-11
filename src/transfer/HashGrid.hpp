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
 
#ifndef HASHGRID_HPP_
#define HASHGRID_HPP_ 

#include <vector>
#include <mfem.hpp> 

#include "Box.hpp"

namespace parelagmc 
{
class HashGrid 
{
public:

    long Hash(const mfem::Vector &point) const;
    long Hash(const std::vector<long> &coord) const;
    void HashRange(const mfem::Vector &min, const mfem::Vector &max, std::vector<long> &hashes);
    HashGrid(const Box &box, const std::vector<int> &dims);

    inline long NCells() const
    {
        return n_cells_;
    }

private:
    Box box_;
    mfem::Vector range_;
    std::vector<int> dims_;
    long n_cells_;
};

void BuildBoxes(const mfem::Mesh &mesh, std::vector<Box> &element_boxes, Box &mesh_box);
bool HashGridDetectIntersections(const mfem::Mesh &src, const mfem::Mesh &dest, std::vector<int> &pairs);

/// \brief Inefficient n^2 algorithm. Do not use unless it is for test purposes
bool DetectIntersections(const mfem::Mesh &src, const mfem::Mesh &dest, std::vector<int> &pairs);
}

#endif /* HASHGRID_HPP_ */
