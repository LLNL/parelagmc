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
 
#ifndef NORMALDISTRIBUTIONSAMPLER_HPP_
#define NORMALDISTRIBUTIONSAMPLER_HPP_

#include <trng/yarn5.hpp>
#include <trng/normal_dist.hpp>

#include <elag.hpp>

namespace parelagmc
{
/// \class NormalDistributionSampler
/// \brief This class uses trng::normal_dist to produce uncorrelated 
/// random numbers with normal distribution with mean mu and 
/// variance sigma2 (i.e. sigma squared).

class NormalDistributionSampler
{
public:
    /// Constructor
    NormalDistributionSampler(double mu, double sigma2);

    /// Destructor
    ~NormalDistributionSampler() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    NormalDistributionSampler(NormalDistributionSampler const&) = delete;    
    NormalDistributionSampler(NormalDistributionSampler&&) = delete;         
    NormalDistributionSampler&                                                
        operator=(NormalDistributionSampler const&) = delete;                 
    NormalDistributionSampler&                                                
        operator=(NormalDistributionSampler&&) = delete;                      
    ///@}            

    /// Provides statistically independent of random numbers to each process
    void Split(int nparts, int mypart);

    /// Get a random number from normal distribution.
    double operator()();

    /// Fill uncorrelated random numbers from normal distribution. 
    void operator()(mfem::Vector & v);

    /// Fill dense matrix m with uncorrelated random numbers from normal
    /// distribution.
    void operator()(mfem::DenseMatrix & m);
    
private:

    /// Specific random number engine from trng
    trng::yarn5 rng;
    
    /// trng class for producing random numbers with normal distribution
    trng::normal_dist<double> d;

}; /* class NormalDistributionSampler */

} /* namespace parelagmc */
#endif /* NORMALDISTRIBUTIONSAMPLER_HPP_ */
