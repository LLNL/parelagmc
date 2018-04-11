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
 
#include "UniformDistributionSampler.hpp"

namespace parelagmc 
{

UniformDistributionSampler::UniformDistributionSampler():
        rng(),
        d() {}

void UniformDistributionSampler::Split(int nparts, int mypart)
{
    rng.split(nparts, mypart);
}

double UniformDistributionSampler::operator()()
{
    return d(rng);
}

void UniformDistributionSampler::operator()(mfem::Vector & x)
{
    double * it = x.GetData();
    double * end = it+x.Size();
    for( ; it != end; ++it)
        *it = d(rng);
}

void UniformDistributionSampler::operator()(mfem::DenseMatrix & m)
{
    double * it = m.Data();
    double * end = it+ (m.Height()*m.Width());
    for( ; it != end; ++it)
        *it = d(rng);
}

} /* namespace parelagmc */
