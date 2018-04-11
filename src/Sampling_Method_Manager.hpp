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
 
#ifndef SAMPLING_METHOD_MANAGER_HPP_
#define SAMPLING_METHOD_MANAGER_HPP_

#include <fstream>
#include <elag.hpp>

namespace parelagmc
{
/// \class Sampling_Method_Manager
/// \brief Abstract class for all sorts of Monte Carlo managers 
class Sampling_Method_Manager
{
public:
    /// Default constructor
    Sampling_Method_Manager() {};

    /// Run MC simulation and compute necessary samples ``on the fly''
    virtual void Run() = 0;
    
    /// Print current values of MC estimator
    virtual void ShowMe(std::ostream & os = std::cout) = 0;

    /// Destructor
    virtual ~Sampling_Method_Manager() = default;

};

} /* namespace parelagmc */
#endif /* SAMPLING_METHOD_MANAGER_HPP_ */
