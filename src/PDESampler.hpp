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
 
#ifndef PDESAMPLER_HPP_
#define PDESAMPLER_HPP_

#include <memory>
#include <vector>
#include <elag.hpp>

#include "MLSampler.hpp"
#include "NormalDistributionSampler.hpp"

namespace parelagmc
{
/// \class PDESampler
/// \brief Operator that samples a Gaussian random field with Matern Covariance 
///
/// A realization is found by solving the SPDE on a mesh.
/// BEWARE: The variance will be artificially inflated along the boundary.
///
/// The SPDE is discretized and the resulting linear system is
///     |  M     B^t     | | u | = | 0              |
///     |  B    -kappa2W | | s |   | -g W^{1/2} xi  |,
///
/// where:
/// M is the RT mass matrix (essential bc)
/// W is the L2 mass matrix
/// D is the discrete divergence operator
/// B = W*D is the divergence bilinear form
/// kappa2 = 1/corlen^2 and g is the correction for the Matern covariance
/// xi is White Noise.
///
/// The block system is solved with the solver/preconditioner,
/// specified in the parelag::ParameterList.

class PDESampler : public MLSampler
{
public:
    /// Constructor
    /// \param mesh MFEM mesh object of embedded mesh
    /// \param dist_sampler Normal distribution sampler
    /// \param prec_params ParameterList
    PDESampler(
        const std::shared_ptr<mfem::ParMesh>& mesh,
        NormalDistributionSampler&  dist_sampler,
        parelag::ParameterList& prec_params);

    /// Destructor
    virtual ~PDESampler() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    PDESampler(PDESampler const&) = delete;    
    PDESampler(PDESampler&&) = delete;         
    PDESampler& operator=(PDESampler const&) = delete;                 
    PDESampler& operator=(PDESampler&&) = delete;                      
    ///@}                                  
 
    /// Compute the DeRhamSequence for each level 
    void BuildDeRhamSequence( 
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & topology);

    /// Set the sequence (if already computed from PhysicalMLSolver)  
    void SetDeRhamSequence( 
        std::vector<std::shared_ptr<parelag::DeRhamSequence>> & sequence);
    
    /// Build the hierarchy of linear operators and solvers for each level
    void BuildHierarchy();  
    
    /// Fills xi with a random sample using dist_sampler
    void Sample(
        const int level, 
        mfem::Vector & xi);

    /// Computes a realization of Gaussian random field with Matern Covariance function by solving SPDE
    void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s);
   
    /// Computes a realization of Gaussian random field with Matern Covariance function by solving SPDE
    /// Writes realization s (= embed_s), a sample from a Gaussian random field.
    /// If use_init = true, then embed_s (on correct level) is used as initial guess.
    void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s,
        mfem::Vector & embed_s,
        bool use_init); 
    
    /// Computes a realization of Gaussian random field (s) and gradient of realization (u) 
    /// Helpful for visualization purposes
    void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s, 
        mfem::Vector & u);
    
    /// Returns sample size for level (i.e. size of stochastic dimension)
    inline int SampleSize(int level) const 
    { 
        return level_size[level];
    }
    
    /// Returns global sample size for level
    inline int GlobalSampleSize(int level) const
    {
        return sequence[level]->GetDofHandler(sform)
            ->GetDofTrueDof().GetTrueGlobalSize();
    }

    /// Returns local dofs
    inline int GetNumberOfDofs(int level) const 
    {
        return level_size[level] +
            sequence[level]->GetNumberOfDofs(uform);
    }

    /// Returns global dofs 
    inline int GetGlobalNumberOfDofs(int level) const
    {
        return  sequence[level]->GetDofHandler(uform)
            ->GetDofTrueDof().GetTrueGlobalSize() +
            sequence[level]->GetDofHandler(sform)
            ->GetDofTrueDof().GetTrueGlobalSize();
    }

    inline size_t GetNNZ(int level) const
    {
        return nnz[level];
    }

    /// Number of iterations of most recent call of Eval; nope just -1
    inline int GetNumIters() const
    {
        return -1;
    }
    
    mfem::FiniteElementSpace * GetFESpace()
    {
        return fespace;
    }
    
    /// Returns TrueP prolongator for level
    inline mfem::HypreParMatrix * GetTrueP(int level )
    { 
        return Ps[level].get();
    }

    /// Displays realization of random field
    void ShowField(
        int ilevel, 
        const mfem::Vector & field, 
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

    void SaveFieldGLVis_u(
        int level, 
        const mfem::Vector & coeff, 
        const std::string prefix) const;
    
    /// Save random field in H1 in glvis format
    void SaveFieldGLVis_H1(
        int level, 
        const mfem::Vector & coeff, 
        const std::string prefix) const;

    /// Save random field in H1 glvis format
    // coeff in L2 on any level whereas mu is in H1 on level 0
    void SaveFieldGLVis_H1Add(
        int level, 
        const mfem::Vector & coeff, 
        const std::string prefix,
        const mfem::Vector & mu) const;

private:

    void glvis_plot(
        int ilevel, 
        const mfem::Vector & sol,
        const std::string prefix, 
        std::ostream & socket) const;

    /// Used for plotting and error evaluation
    void prolongate_to_fine_grid(
        int ilevel, 
        const mfem::Vector & sol, 
        mfem::GridFunction & x) const;

    void prolongate_to_fine_grid_u(
        int ilevel, 
        const mfem::Vector & sol, 
        mfem::GridFunction & x) const;

    /// Used for plotting and error evaluation
    void prolongate_to_fine_grid_H1(
        int ilevel, 
        const mfem::Vector & sol, 
        mfem::GridFunction & x) const;

    /// Mesh of domain
    const std::shared_ptr<mfem::ParMesh>& mesh;
    
    /// Number of spatial dimensions
    const int nDim;

    const int uform;
    const int sform;
    
    /// Normal distribution sampler
    NormalDistributionSampler&  dist_sampler;

    /// Number of levels
    int nLevels;

    mfem::FiniteElementSpace * fespace;
    mfem::FiniteElementSpace * fespace_u;
    
    parelag::ParameterList& master_list;
    parelag::ParameterList& prob_list;

    const bool save_vtk;
    const bool lognormal;
    const bool verbose;
    const bool entire_seq;

    /// Correlation length of matern covariance
    const double corlen;

    /// 1/corlen^2
    const double alpha;
   
    /// Matern scaling coefficient for SPDE
    const double matern_coeff;

    /// Flag if hybridization is the linear solver
    bool if_solver_hybridization;
    
    mfem::Array<int> level_size;
    mfem::Array<size_t> nnz;
    
    mfem::Array2D<int> block_offsets;

    mfem::Array2D<int> true_block_offsets;

    mfem::Array<parelag::SharingMap *> umap;
    mfem::Array<parelag::SharingMap *> pmap; 
    
    int num_procs;
    int myid;
    
    /// DeRham sequence
    std::vector< std::shared_ptr<parelag::DeRhamSequence> > sequence;
    std::vector< std::shared_ptr<parelag::MfemBlockOperator> > A;

    std::vector< std::unique_ptr<mfem::Vector> > w_sqrt;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > Ps;
    std::vector< std::unique_ptr<mfem::Solver> > invA;
    
};

} /* namespace parelagmc */
#endif /* PDESAMPLER_HPP_ */
