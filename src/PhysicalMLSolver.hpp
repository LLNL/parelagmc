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
 
#ifndef PHYSICALMLSOLVER_HPP_
#define PHYSICALMLSOLVER_HPP_

#include <memory>
#include <elag.hpp>

namespace parelagmc 
{
/// \class PhysicalMLSolver 
/// \brief Abstract base class for ML solvers of forward deterministic problem. 
class PhysicalMLSolver 
{
public:
    
    /// Default constructor
    PhysicalMLSolver() {};

    /// Destructor
    virtual ~PhysicalMLSolver() = default;

    /// Solve and update quantity of interest Q, cost C 
    virtual void SolveFwd(
        int ilevel, 
        mfem::Vector & k_over_k_ref,
        double & Q,
        double & C) = 0;
    
    /// Computes parameter-to-observable map that maps the uncertain input parameter to the observation data.
    virtual void SolveFwd_RtnPressure(
        int ilevel,
        mfem::Vector & k_over_k_ref,
        mfem::Vector & P,
        double & C,
        double & Q,
        bool compute_Q) = 0;

    /// Returns the DeRham sequence
    virtual std::vector<std::shared_ptr<parelag::DeRhamSequence>> & GetSequence() = 0;
    
    /// Return pressure space
    virtual mfem::FiniteElementSpace * GetPressureSpace() const = 0;
    
    virtual int GetNumberOfDofs(
        int ilevel) const = 0;

    virtual int GetGlobalNumberOfDofs(
        int ilevel) const = 0;
    
    virtual int GetNNZ(
        int ilevel) const = 0;

}; /* class PhysicalMLSolver */ 

} /* namespace parelagmc */

#endif /* PHYSICALMLSOLVER_HPP_ */
