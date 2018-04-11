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
 
#ifndef UTILITIES_HPP_
#define UTILITIES_HPP_

#include <memory>
#include <cmath>

#include <elag.hpp>

namespace parelagmc
{
/// Build the sequence of agglomerated topologies using structured coarsening. 
void BuildTopologyGeometric(
    std::shared_ptr<mfem::ParMesh> pmesh,
    mfem::Array<int> & level_nElements,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & topology);

/// Build the sequence of agglomerated topologies using structured coarsening 
/// for a mesh and an embedded mesh using LogicalPartitioner.  
void EmbeddedBuildTopologyGeometric(
    std::shared_ptr<mfem::ParMesh> pmesh,
    std::shared_ptr<mfem::ParMesh> pembedmesh,
    mfem::Array<int> & level_nElements,
    mfem::Array<int> & embed_level_nElements,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & topology,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & embed_topology,
    std::vector<mfem::Array<int> > & material_id);

/// Build the sequence of agglomerated topologies using unstructured coarsening. 
void BuildTopologyAlgebraic(
    std::shared_ptr<mfem::ParMesh> pmesh,
    const int coarsening_factor,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & topology);

/// Build the sequence of agglomerated topologies using unstructured coarsening 
/// for a mesh and an embedded mesh using LogicalPartitioner.  
void EmbeddedBuildTopologyAlgebraic(
    std::shared_ptr<mfem::ParMesh> pmesh,
    std::shared_ptr<mfem::ParMesh> pembedmesh,
    const int coarsening_factor,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & topology,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & embed_topology,
    std::vector<mfem::Array<int> > & material_id);
 
/// Determine exponential regression 
double expWRegression(
    const mfem::Vector & y, 
    const mfem::Vector & x, 
    int skip_n_last);

void GeneratePointCoordinates(
    mfem::Mesh *mesh, 
    mfem::DenseMatrix &PointCoordinates);

template <int coord>
inline double xfun(mfem::Vector &v)
{
    return v(coord);
}

int FindClosestPointID(
    mfem::Mesh *mesh, 
    mfem::Vector &x);

mfem::HypreParVector * chi_center_of_mass(mfem::ParMesh * pmesh);

/// Computes dot product of distributed vectors
double dot(
    const mfem::Vector & a, 
    const mfem::Vector & b, 
    MPI_Comm comm);

/// First squares the values of a, then computes dot product
double squared_dot(
    const mfem::Vector & a, 
    const mfem::Vector & b, 
    MPI_Comm comm);

double sum(
    const mfem::Vector & a, 
    MPI_Comm comm);

void OutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters);
   
void OutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& nnz);

void OutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g);

void OutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters);

void OutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz);

void OutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g);

void OutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters);

void OutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz);

void OutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g);

void ReduceAndOutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters);
   
void ReduceAndOutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& nnz);

void ReduceAndOutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g);

void ReduceAndOutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters);

void ReduceAndOutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz);

void ReduceAndOutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g);

void ReduceAndOutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters);

void ReduceAndOutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz);

void ReduceAndOutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g);

void OutputRandomFieldErrors(const mfem::Vector& exp_errors_L2,
                             const mfem::Vector& var_errors_L2);

void ReduceAndOutputRandomFieldErrors(const mfem::Vector& exp_errors_L2,
                                      const mfem::Vector& var_errors_L2);

void ReduceAndOutputRandomFieldErrorsMax(const mfem::Vector& exp_errors_L2,
                                      const mfem::Vector& var_errors_L2);


/// Compute scalar g = (4*PI)^(d/4)*(1/corlen)^nu*sqrt(Gamma(nu+d/2)/Gamma(nu))
inline double ComputeScalingCoefficientForSPDE(const double corlen,
                                        const int nDim)
{
    const double myDim( static_cast<double>( nDim ) );
    const double nu = 2. - (myDim / 2.);
    const double gnu=std::tgamma(nu);
    const double gnudim=std::tgamma(nu + myDim);
    const double c= std::pow(16.*std::atan(1.),.5*myDim);
    const double k= std::pow(1./corlen,2.*nu);

    return sqrt(c*gnudim*k/gnu);
    
}

/// Save a vector in H1 FE space in glvis format
void SaveFieldGLVis_H1(
        mfem::ParMesh * mesh,
        const mfem::Vector & coeff,
        const std::string prefix);

/// \brief Save a coefficient in L2 FE space in glvis format
///
/// Useful for SPE10 examples to plot kinv and k using dataset.
void SaveCoefficientGLVis(
        mfem::ParMesh * mesh,
        mfem::VectorFunctionCoefficient & coeff,
        const std::string prefix);

void SaveCoefficientGLVis_H1(
        mfem::ParMesh * mesh,
        mfem::VectorFunctionCoefficient & coeff,
        const std::string prefix);

// Returns the modified Bessel function of first kind for nu=1, for positive real x           
// This is following Chapter 6.6: Modified Bessel Functions of Integer 
// Order from "Numerical Recipes in C" 
inline double bessi1(double x)                                                 
{           
    double ax, ans, y;
    if ((ax=fabs(x)) < 3.75) 
    {
        y=x/3.75;
        y*=y;
        ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
            +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
    } 
    else 
    {
        y=3.75/ax;
        ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1-y*0.420059e-2));
        ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
            +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
        ans *= (std::exp(ax)/std::sqrt(ax));
    }
    return x < 0.0 ? -ans : ans;
}

// Returns the modified Bessel function of second king for nu=1, for positive real x
// This is following Chapter 6.6: Modified Bessel Functions of Integer 
// Order from "Numerical Recipes in C" 
inline double bessk1(double x)
{
    double y, ans;
    if (x <= 2.0)
    {
        y = x*x/4.0;
        ans = (std::log(x/2.0)*bessi1(x))+(1.0/x)*(1.0+y*(0.15443144
            +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
            +y*(-0.110404e-2+y*(-0.4686e-4)))))));
    }
    else
    {
        y=2.0/x;
        ans=(std::exp(-x)/std::sqrt(x))*(1.25331414+y*(0.23498619
            +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
            +y*(0.325614e-2+y*(-0.68245e-3)))))));
    }
    return ans;
}

} /* namespace parelagmc */
#endif /* UTILITIES_HPP_ */


