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
 
#ifndef ANALYTICEXPONENTIALCOVARIANCE_HPP_
#define ANALYTICEXPONENTIALCOVARIANCE_HPP_

#include <elag.hpp>

#include "CovarianceFunction.hpp"

namespace parelagmc
{
/// \class AnalyticExponentialCovariance
/// \brief Class representing a KL expansion of an exponential random field.
///
/// This class provides a means for evaluating a random field
/// \f$k(x,\omega)\f$, \f$x\in D\f$, \f$\omega\in\Omega\f$ through the
/// %KL expansion
/// \f[
///     k(x,\omega) \approx k_0 +
///       \sigma\sum_{k=1}^M \sqrt{\theta_k}b_k(x)\xi_k(\omega)
/// \f]
/// for the case when the covariance function of \f$k\f$ is exponential:
/// \f[
///     \mbox{cov}(x,x') = \sigma^2\exp(-|x_1-x_1'|/\lambda_1-...-|x_d-x_d'|/\lambda_d),
/// \f]
/// where \f$\sigma^2\f$ is the variance of the stochastic field, and \f$\lambda_j\f$
/// is the correlation length in the j-th direction.
/// In this case, the covariance function and domain factor into a product
/// 1-dimensional covariance functions over 1-dimensional domains, and thus
/// the eigenfunctions \f$b_k\f$ and eigenvalues \f$\theta_k\f$ factor into
/// a product of corresponding 1-dimensional eigenfunctions and values.
/// Analytic expressions are used to compute the 1D eigenpairs then for higher
/// spatial dimensions, the appropriate tensor products of the these 
/// one-dimensional eigenfunctions are formed, with corresponding eigenvalue 
/// given by the product of the one-dimensional eigenvalues.   
class AnalyticExponentialCovariance : public CovarianceFunction 
{
public:
    /// Constructor
    AnalyticExponentialCovariance(
        const std::shared_ptr<mfem::ParMesh>& mesh_,
        parelag::ParameterList& params_);

    /// Destructor
    virtual ~AnalyticExponentialCovariance() = default;

    // @{ \brief All special constructors and assignment are deleted.         
    AnalyticExponentialCovariance(AnalyticExponentialCovariance const&) = delete;                                
    AnalyticExponentialCovariance(AnalyticExponentialCovariance&&) = delete;                                     
    AnalyticExponentialCovariance&                                                              
        operator=(AnalyticExponentialCovariance const&) = delete;                               
    AnalyticExponentialCovariance&                                                              
        operator=(AnalyticExponentialCovariance&&) = delete;                                    
    ///@}    

    /// Computes eigenpairs of the covariance operator for fe grid 
    void SolveEigenvalue();
   
    /// Print KL expansion
    void ShowMe(std::ostream & os = std::cout);

    /// Checks orthogonality of computed evs
    void CheckOrthogonalityEigenvectors(std::ostream & os = std::cout);

    /// Returns vector of eigenvalues of exponential covariance operator 
    inline mfem::Vector & Eigenvalues() 
    {
        return eval; 
    }

    /// Returns dense matrix of computed eigenfunctions of exponential covariance operator evaluated at grid points
    inline mfem::DenseMatrix & Eigenvectors() 
    { 
        return evect; 
    }

    /// Returns total number of modes
    inline int NumberOfModes() const 
    { 
        return totnmodes; 
    }

    void SaveVTK(mfem::Mesh * mesh, 
        mfem::GridFunction & coeff, 
        std::string prefix) const;

private:
    /// 
    void computeEigs(mfem::Vector & eval, 
        mfem::DenseMatrix & evect);
   
    /// Computes the vector of frequencies of the eigenfunctions 
    void computeOmega(int nmodes, 
        double scaled_corr_length, 
        mfem::Vector & omega);

    /// Computes 1D eigenvalues 
    void computeEigenvalues1d(const mfem::Vector & omega, 
        double scaled_corr_length, 
        double lx, 
        mfem::Vector & eval1d);
   
    /// Computes 1D eigenvectors
    void computeEigenvectors1d(int coord, 
        const mfem::Vector & omega, 
        double scaled_corr_length, 
        double lx, 
        const mfem::SparseMatrix & mass, 
        mfem::DenseMatrix & evect1d);

    /// Finite element mesh
    const std::shared_ptr<mfem::ParMesh>& mesh;

    /// Number of spatial dimensions of the field
    const int ndim;

    /// Input list for parameters
    parelag::ParameterList& prob_list; 
    
    /// Array of number of modes in each spatial dimension 
    std::vector<int>  nmodes;
    
    /// Number of degrees of freedom of fe space
    const int size;

    /// Variance of random field
    const double var;
    
    /// Array of lengths of the domain in each spatial dimension 
    std::vector<double> domain_lengths;

    /// Total number of modes
    int totnmodes;
    
    /// Finite element collection                                              
    std::unique_ptr<mfem::L2_FECollection> fec;

    /// Finite element space
    std::unique_ptr<mfem::FiniteElementSpace> fespace;
    
    /// Measure of spatial domain \f$D\f$ (i.e. \f$\int_{D} dx\f$)
    double measD;

    /// Vector of correlation lengths 
    mfem::Vector correlation_lengths;

    /// Vector of computed eigenvalues of exponential covariance operator 
    mfem::Vector eval;

    /// Dense matrix of computed eigenfunctions of exponential covariance operator evaluated at grid points
    mfem::DenseMatrix evect;


}; //AnalyticExponentialCovariance

/// Class for KL 1D eigenfunction to be evaluated on a fe space 
class AnalyticExponentialEvect1dCoefficient : public mfem::Coefficient
{
public:
    AnalyticExponentialEvect1dCoefficient(int coord, 
        double lambda, 
        double lx, 
        double omega_n = -1.);
    void SetOmega(double omega_n);
    double Eval(mfem::ElementTransformation &T, 
        const mfem::IntegrationPoint &ip);

private:
    int coord;
    double lambda;
    double omega_n;
    double lx;
}; /* AnalyticExponentialEvect1dCoefficient */

} /* namespace parelagmc */
#endif /* ANALYTICEXPONENTIALCOVARIANCE_HPP_ */
