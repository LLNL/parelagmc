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
 
#ifndef MORTAR_ASSEMBLE_HPP_
#define MORTAR_ASSEMBLE_HPP_ 

#include <memory>
#include <math.h>
#include <mfem.hpp>
#include <opencl_adapter.hpp>

#include "HashGrid.hpp"

#define USE_DOUBLE_PRECISION
#define DEFAULT_TOLLERANCE 1e-12

namespace parelagmc 
{
class Intersector : public moonolith::OpenCLAdapter 
{
public:
    //I do not know why the compiler wants this...
    template<typename T>
    inline static T sqrt(const T v) {
        return std::sqrt(v);
    }
    #include <kernels/all_kernels.cl>
};


typedef Intersector::PMesh Polyhedron;

void Print(const mfem::IntegrationRule &ir, std::ostream &os = std::cout);

double SumOfWeights(const mfem::IntegrationRule &ir);

double Sum(const mfem::DenseMatrix &mat);

void MakeCompositeQuadrature2D(const mfem::DenseMatrix &polygon, 
    const double weight, const int order, mfem::IntegrationRule &c_ir);

void MakeCompositeQuadrature3D(const Polyhedron &polyhedron, 
    const double weight, const int order, mfem::IntegrationRule &c_ir);

void TransformToReference(mfem::ElementTransformation &Trans, const int type, 
    const mfem::IntegrationRule &global_ir, mfem::IntegrationRule &ref_ir);

void MortarAssemble(const mfem::FiniteElement &trial_fe, 
    const mfem::IntegrationRule &trial_ir, 
    const mfem::FiniteElement &test_fe, 
    const mfem::IntegrationRule &test_ir, 
    mfem::ElementTransformation &Trans, 
    mfem::DenseMatrix &elmat);

bool MortarAssemble(mfem::FiniteElementSpace &src, 
    mfem::FiniteElementSpace &dest, 
    std::shared_ptr<mfem::SparseMatrix> &B);

bool Transfer(mfem::FiniteElementSpace &src, mfem::Vector &src_fun, 
    mfem::FiniteElementSpace &dest, mfem::Vector &dest_fun);

void MakePolyhedron(const mfem::Mesh &m, const int el_index, Polyhedron &polyhedron);

bool Intersect2D(const mfem::DenseMatrix &poly1, const mfem::DenseMatrix &poly2, 
    mfem::DenseMatrix &intersection);

bool Intersect3D(const mfem::Mesh &m1, const int el1, const mfem::Mesh &m2, 
    const int el2, Polyhedron &intersection);
} /* namespace parelagmc */

#undef mortar_assemble
#endif /* MORTAR_ASSEMBLE_HPP_ */
