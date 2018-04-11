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
 
#ifndef UNIFORMDISTRIBUTIONSAMPLER_HPP_
#define UNIFORMDISTRIBUTIONSAMPLER_HPP_

#include <mfem.hpp>
#include <trng/yarn5.hpp>
#include <trng/uniform01_dist.hpp>

namespace parelagmc
{
/// \class UniformDistributionSampler
/// \brief Uniform distribution sampler
///
/// This class uses trng::uniform01_dist to produce uncorrelated 
/// random numbers with uniform distribution on the interval [0, 1)

class UniformDistributionSampler
{
public:
    /// Constructor
    UniformDistributionSampler();

    /// Destructor
    ~UniformDistributionSampler() = default;
   
    /// @{ \brief All special constructors and assignment are deleted.
    UniformDistributionSampler(UniformDistributionSampler const&) = delete;
    UniformDistributionSampler(UniformDistributionSampler&&) = delete;
    UniformDistributionSampler& 
        operator=(UniformDistributionSampler const&) = delete;
    UniformDistributionSampler& 
        operator=(UniformDistributionSampler&&) = delete;
    ///@}

    /// Provides statistically independent of random numbers to each process
    void Split(int nparts, int mypart);

    /// Draw a random number from Unif[0,1) distribution.
    double operator()();

    /// Fill vector uncorrelated random numbers from Unif[0,1). 
    void operator()(mfem::Vector & v);

    /// Fill dense matrix m with uncorrelated random numbers from Unif[0,1).
    void operator()(mfem::DenseMatrix & m);

private:

    /// Specific random number engine from trng
    trng::yarn5 rng;
    
    /// trng class for producing random numbers with uniform distribution
    trng::uniform01_dist<double> d;

}; /* class UniformDistributionSampler */

} /* namespace parelagmc */
#endif /* UNIFORMDISTRIBUTIONSAMPLER_HPP_ */
