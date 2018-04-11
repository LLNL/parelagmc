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
 
#ifndef MLSAMPLER_HPP_
#define MLSAMPLER_HPP_

#include <memory>
#include <elag.hpp>

namespace parelagmc 
{
/// \class MLSampler
/// \brief Abstract base class for ML samplers to sample a random field
class MLSampler
{
public:
    
    /// Default constructor
    MLSampler() {};

    /// Destructor
    virtual ~MLSampler() = default;

    /// Fill vector with random sample using dist_sampler
    virtual void Sample(
        const int level, 
        mfem::Vector & xi) = 0;
    
    /// Evaluate random field at level with random sample xi
    virtual void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector &s) = 0;

    virtual void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector &s,
        mfem::Vector &u,
        bool use_init) = 0;
    
    virtual int SampleSize(int level) const = 0;

    virtual size_t GetNNZ(int level) const = 0;
    
    virtual void BuildDeRhamSequence(
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & topology) {};

    virtual void BuildDeRhamSequence(
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & topology,
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & embed_topology) {};
    
    virtual void SetDeRhamSequence(
        std::vector<std::shared_ptr<parelag::DeRhamSequence>> & sequence) {};
    
    virtual void BuildHierarchy() = 0;
    
    virtual mfem::HypreParMatrix * GetTrueP(int level ) = 0;

    /// Save fine mesh in glvis format
    virtual void SaveMeshGLVis(const std::string prefix) const = 0;

    /// Save random field in glvis format
    virtual void SaveFieldGLVis(
        int level,
        const mfem::Vector & coeff,
        const std::string prefix) const = 0;

    /// Compute L2 Error of coeff and exact soln (double) for level
    virtual double ComputeL2Error(
        int level,
        const mfem::Vector & coeff,
        double exact) const = 0;

    /// Compute Max Error of coeff and exact soln (double) for level
    virtual double ComputeMaxError(
        int level,
        const mfem::Vector & coeff,
        double exact) const = 0;

private:
    bool lognormal;
}; // class MLSampler

} /* namespace parelagmc */

#endif /* MLSAMPLER_HPP_ */
