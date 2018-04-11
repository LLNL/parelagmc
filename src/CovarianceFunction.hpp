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
 
#ifndef COVARIANCEFUNCTION_HPP_
#define COVARIANCEFUNCTION_HPP_

#include <elag.hpp>

namespace parelagmc 
{

class CovarianceFunction 
{
/// \class CovarianceFunction
/// \brief Abstract base class for covariance operators.
public:
    /// Constructor
    CovarianceFunction() {}

    /// Destructor
    virtual ~CovarianceFunction() = default;

    /// Print ews/evs
    virtual void ShowMe(std::ostream & os = std::cout) = 0;

    /// Solve for ew/ev of covariance operator
    virtual void SolveEigenvalue() = 0;

    /// Returns vector of eigenvalues of covariance operator 
    virtual mfem::Vector & Eigenvalues() = 0; 

    /// Returns dense matrix of computed eigenfunctions of covariance operator evaluated at grid points
    virtual mfem::DenseMatrix & Eigenvectors() = 0; 

    virtual int NumberOfModes() const = 0;

private:
    bool lognormal;

}; /* class CovarianceFunction */

} /* namespace parelagmc */

#endif /* ifndef COVARIANCEFUNCTION_HPP_ */
