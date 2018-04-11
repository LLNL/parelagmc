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
 
#ifndef MATERNCOVARIANCE_HPP_
#define MATERNCOVARIANCE_HPP_

#include <memory>
#include <vector>
#include <elag.hpp>

#include "CovarianceFunction.hpp"

namespace parelagmc
{
class MaternCovariance : public CovarianceFunction
{
public:
    /// Constructor
    MaternCovariance(
        const std::shared_ptr<mfem::ParMesh>& mesh_,
        parelag::ParameterList& params_);

    /// Destructor
    virtual ~MaternCovariance() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    MaternCovariance(MaternCovariance const&) = delete;                                
    MaternCovariance(MaternCovariance&&) = delete;                                     
    MaternCovariance& operator=(MaternCovariance const&) = delete;                               
    MaternCovariance& operator=(MaternCovariance&&) = delete;                                    
    ///@}            

    void ShowMe(std::ostream & os = std::cout);

    /// Generate the Fine Grid covariance matrix right and left scaled by massL2diag???
    void GenerateCovarianceMatrix();

    /// Generate the Fine Grid Point Coordinates
    void GeneratePointCoordinates();

    /// Generate the Coarse covariance matrix, with interpolation P
    void GenerateCovarianceMatrix(
        mfem::SparseMatrix &P, 
        int version=2);

    /// Generate subset of fine covariance matrix
    void GenerateCovarianceMatrix(
        mfem::Array<int> &row_index,
        mfem::Array<int> &col_index,
        mfem::DenseMatrix &G);

    double ComputeCoarseCovarianceMatrixEntry(
        mfem::Array<int> &row_index,
        mfem::Array<int> &col_index,
        mfem::Vector &weights_row, 
        mfem::Vector &weights_col);

    /// Returns scaling factor to impose unit marginal variance of GF 
    /// \f$\sqrt{\frac{(4\pi)^{d/2}\Gamma(\nu+d/2)\kappa^{2\nu}}{\Gamma(\nu)}}\f$
    double ComputeScalingCoefficient() const;
    
    /// Solves generalized ew problem Matern x = lambda massL2diag x 
    void SolveEigenvalue();
    
    inline mfem::Vector & Eigenvalues() 
    { 
        return evals; 
    }

    inline mfem::DenseMatrix & Eigenvectors() 
    { 
        return evect; 
    }

    /// Returns 1/{\Gamma(\nu)2^{\nu-1}}(\kappa || x -y||)^{\nu} K_{\nu} 
    /// where K_{\nu} is the irregular modified Bessel functional 
    /// of order \nu or 1 if value < 1e-10
    double Compute(
        const mfem::Vector &x, 
        const mfem::Vector &y);

    inline int NumberOfModes() const 
    { 
        return totnmodes; 
    }

    /// Use LOBPCG or SymEigenSolver to sovle ew problem
    bool lobpcg;

private:
    /// Generate the Coarse covariance matrix, with input P, v2 is default version
    void generateCovarianceMatrix_v1(mfem::SparseMatrix &P);
    void generateCovarianceMatrix_v2(mfem::SparseMatrix &P);

    /// Calls hypre's Lobpcg to compute the evs and ews
    void solveEigenvalueLOBPCG();

    /// Calls elag::SymEigensolver to compute the evs and ews
    void solveEigenvalueSymEigensolver();

    /// FE mesh
    const std::shared_ptr<mfem::ParMesh>& mesh;   
 
    /// Number of spatial dimensions
    const int nDim;
    
    /// Input list for parameters
    parelag::ParameterList& prob_list;
    
    /// Correlation length of GF 
    const double corlen;

    /// Scaling factor inversely proportional to the correlation length
    const double kappa;
    
    /// Scalar that determines the mean-square differentiability of the underlying process
    const double nu;

    /// Scalar of covariance function scale=1/{\Gamma(\nu)2^{\nu-1}}
    const double scale;
    
    bool point_coords_exist;

    bool cov_matrix_exist;
    
    /// Finite element collection                                              
    std::unique_ptr<mfem::L2_FECollection> fec;

    /// L2 finite element space
    std::unique_ptr<mfem::ParFiniteElementSpace> fespace;
    
    int num_procs;

    int myid;
    
    /// Number of modes in KL expansion
    int totnmodes = 1.;
 
    /// Number of spatial degrees of freedom
    int ndofs;
    
    std::unique_ptr<mfem::ParBilinearForm> massL2;
    
    /// Ndofs x Spatial dimension matrix of point coordinates
    mfem::DenseMatrix PointCoordinates;

    /// Matern covariance matrix
    mfem::DenseMatrix Matern;

    /// Matrix of eigenvectors of matern covariance matrix
    mfem::DenseMatrix evect;

    /// Vector of eigenvalues of matern covariance matrix
    mfem::Vector evals;
    
    /// Diagonal of L2 mass matrix
    mfem::Vector massL2diag;

}; //class MaternCovariance

class MaternKernel{
public:
    static double func(mfem::Vector &x);
    static mfem::Vector y;
    static MaternCovariance *K;
}; 
} /* namespace parelagmc */

#endif /* MATERNCOVARIANCE_HPP_ */
