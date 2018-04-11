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
 
#ifndef KLSAMPLER_HPP_
#define KLSAMPLER_HPP_

#include <memory>
#include <elag.hpp>

#include "MLSampler.hpp"
#include "NormalDistributionSampler.hpp"
#include "CovarianceFunction.hpp"

namespace parelagmc
{
/// \class KLSampler
/// \brief Samples a KL expansion of a random field

class KLSampler : public MLSampler 
{
public:
    /// Constructor
    KLSampler(
        const std::shared_ptr<mfem::ParMesh>& mesh_, 
        NormalDistributionSampler& dist_sampler, 
        CovarianceFunction& covariance, 
        parelag::ParameterList& prec_params);

    /// Destructor
    virtual ~KLSampler();

    /// @{ \brief All special constructors and assignment are deleted.         
    KLSampler(KLSampler const&) = delete;                                
    KLSampler(KLSampler&&) = delete;                                     
    KLSampler& operator=(KLSampler const&) = delete;                               
    KLSampler& operator=(KLSampler&&) = delete;                                    
    ///@}          

    /// Compute the DeRhamSequence for each level 
    void BuildDeRhamSequence(
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & topology);

    /// Set the sequence (if already computed from PhysicalMLSolver)  
    void SetDeRhamSequence(
        std::vector<std::shared_ptr<parelag::DeRhamSequence>> & sequence);

    /// Call the CovarianceFunction's SolveEigenvalue function and project result to each level 
    void BuildHierarchy();

    /// Fills xi with a random sample using dist_sampler 
    void Sample(
        int level, 
        mfem::Vector & xi);

    /// Computes truncated KLE of log-normal random field  
    void Eval(
        int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s);

    inline void Eval(
        int level,
        const mfem::Vector & xi,
        mfem::Vector & s,
        mfem::Vector & u,
        bool use_init) 
    {
        this->Eval(level, xi, s);
    }
    
    /// Returns sample size for level
    inline int SampleSize(int level) const
    {
        return level_size[level];
    }

    inline size_t GetNNZ(int level) const
    {
        return nnz[level];
    }

    /// Returns TrueP prolongator for level
    inline mfem::HypreParMatrix * GetTrueP(int level )
    {
        return P[level].get();
    }

    /// Displays realization of random field 
    inline void ShowField(
        int ilevel, 
        mfem::Vector & field,
        const std::string prefix,
        std::ostream & socket)
    { 
        glvis_plot(ilevel, field, prefix, socket);
    }

    /// Compute L2 Error of coeff and exact soln (double) for level
    double ComputeL2Error(
        int level,
        const mfem::Vector & coeff,
        double exact) const;

    /// Compute Max Error of coeff and exact soln (double) for level
    double ComputeMaxError(
        int level,
        const mfem::Vector & coeff,
        double exact) const;

    /// Save fine mesh in glvis format
    void SaveMeshGLVis(const std::string prefix) const;

    /// Save random field in glvis format
    void SaveFieldGLVis(
        int level, 
        const mfem::Vector & coeff, 
        const std::string prefix) const;
       
    void SaveVTK(
        mfem::Mesh * mesh, 
        mfem::GridFunction & coeff, 
        std::string prefix) const;

private:
    void glvis_plot(
        int ilevel, 
        mfem::Vector & sol, 
        std::string prefix,
        std::ostream & socket);

    /// For plotting 
    void prolongate_to_fine_grid(
        int ilevel, 
        const mfem::Vector & sol, 
        mfem::GridFunction & x) const;

    /// Mesh
    const std::shared_ptr<mfem::ParMesh>& mesh; 
    
    /// Distribution sampler
    NormalDistributionSampler& dist_sampler;
    
    /// Covariance operator which supplies ew/ev 
    CovarianceFunction& covariance; 
    
    /// Spatial dimension
    const int nDim;

    parelag::ParameterList& prob_list;
    
    /// Whether random filed is lognormal or Gaussian
    const bool lognormal;
    
    /// Number of levels
    int nLevels;

    /// Finite element collection                                              
    std::unique_ptr<mfem::L2_FECollection> fec;
                                                                               
    /// L2 finite element space                                                
    std::unique_ptr<mfem::FiniteElementSpace> fespace;
    
    int myid;
    
    int num_procs;

    /// Eigenvalues of covariance function from KLExpansion
    mfem::Vector * eval;

    /// Array of eigenfunctions of covariance function from KLExpansion 
    /// evaluated on fespace for each level of hierarchy
    mfem::Array<mfem::DenseMatrix *> evect;

    /// Number of total number of modes
    int totnmodes = 1.;

    mfem::Array<int> level_size;
    
    mfem::Array<int> nnz;

    /// DeRham sequence
    std::vector< std::shared_ptr<parelag::DeRhamSequence> > sequence;

    /// Prolongation operators
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > P;

}; //class KLSampler

} /* namespace parelagmc */
#endif /* KLSAMPLER_HPP_ */
