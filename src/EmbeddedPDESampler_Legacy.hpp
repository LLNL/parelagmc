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
 
#ifndef EMBEDDEDPDESAMPLER_LEGACY_HPP_
#define EMBEDDEDPDESAMPLER_LEGACY_HPP_

#include <memory>
#include <vector>
#include <elag.hpp>

#include "MLSampler.hpp"
#include "NormalDistributionSampler.hpp"

namespace parelagmc
{
/// \class EmbeddedPDESampler
/// \brief Operator that samples a Gaussian random field with Matern Covariance 
///
/// A realization is found by solving the SPDE on an embedded mesh then projecting
/// to the orignal mesh (assuming the meshes match).
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
/// By computing the Schur Complement with respect to u we get
///
/// A u = - g/kappa2 D^t W^{1/2} xi, A = M + 1/kappa2 B^tWB
///
/// which is solved with CG preconditioned with hypre's ADS.
/// Then, the realization if found by computing
///
/// s = 1/kappa2 D u + g/kappa2 W^{-1/2} xi 

class EmbeddedPDESampler_Legacy : public MLSampler
{
public:
    /// Constructor
    /// \param orig_mesh MFEM mesh object of original mesh
    /// \param embed_mesh MFEM mesh oject of embedded mesh
    /// \param dist_sampler Normal distribution sampler
    /// \param materialId Generated from LogicalPartitioner to keep track of material
    /// id of grid for each level 
    /// \param prec_params ParameterList
    EmbeddedPDESampler_Legacy(
        const std::shared_ptr<mfem::ParMesh>& orig_mesh_,                  
        const std::shared_ptr<mfem::ParMesh>& embed_mesh_,
        NormalDistributionSampler& dist_sampler,
        std::vector< mfem::Array <int>> & materialId,
        parelag::ParameterList& prec_params);
 
    /// Destructor
    virtual ~EmbeddedPDESampler_Legacy() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    EmbeddedPDESampler_Legacy(EmbeddedPDESampler_Legacy const&) = delete;                                
    EmbeddedPDESampler_Legacy(EmbeddedPDESampler_Legacy&&) = delete;                                     
    EmbeddedPDESampler_Legacy&                                                              
        operator=(EmbeddedPDESampler_Legacy const&) = delete;                               
    EmbeddedPDESampler_Legacy&                                                              
        operator=(EmbeddedPDESampler_Legacy&&) = delete;                                    
    ///@}         

    /// Compute the DeRhamSequence for each level using embedded topology 
    void BuildDeRhamSequence(
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & embed_topology);

    /// Build the hierarchy of linear operators and solvers for each level
    void BuildHierarchy();
 
    /// Fills xi with a random sample using dist_sampler
    void Sample(
        const int level, 
        mfem::Vector & xi);

    /// \brief Computes a realization on the original mesh
    /// Returns a sample from a Gaussian random field with Matern Covariance 
    /// function found by solving the SPDE on the embedded mesh then projecting
    /// to the orignal mesh.
    void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s);

    /// \brief Computes a realization of Gaussian random field (for consistency with other samplers) 
    /// Writes realization s on original mesh, and realization embed_s on embedded mesh. 
    /// Beware: use_init is not used here.
    void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s,
        mfem::Vector & embed_s,
        bool use_init);

    /// Computes a realization on the embedded mesh (useful for visualization)
    void EmbedEval(
        const int level,
        const mfem::Vector & xi,
        mfem::Vector & s,
        mfem::Vector & u);
    
    void EmbedEval(
        const int level,
        const mfem::Vector & xi,
        mfem::Vector & s);

    /// Returns sample size for level for embedded mesh
    inline int EmbedSampleSize(int level) const 
    { 
        return level_size[level];
    }

    /// Returns sample size for level for original mesh
    inline int SampleSize(int level) const 
    {
        return orig_level_size[level];
    }

    /// Returns global sample size for level
    inline int GlobalSampleSize(int level) const
    {
        return embed_sequence[level]->GetDofHandler(sform)
            ->GetDofTrueDof().GetTrueGlobalSize();
    }

    /// Returns local dofs
    inline int GetNumberOfDofs(int level) const
    {
        return level_size[level] +
            embed_sequence[level]->GetNumberOfDofs(uform);
    }

    /// Returns global dofs 
    inline int GetGlobalNumberOfDofs(int level) const
    {
        return  embed_sequence[level]->GetDofHandler(uform)
            ->GetDofTrueDof().GetTrueGlobalSize() +
            embed_sequence[level]->GetDofHandler(sform)
            ->GetDofTrueDof().GetTrueGlobalSize();
    }

    inline size_t GetNNZ(int level) const
    {
        return nnz[level];
    }

    /// Number of iterations of most recent call of Eval
    inline int GetNumIters() const
    {
        return iter;
    }
    
    /// Returns mesh prolongator for level (embedded -> original)
    inline mfem::HypreParMatrix * GetMeshP(int level )
    {
        return meshP[level].get();
    }
  
    /// Returns TrueP prolongator for level
    inline mfem::HypreParMatrix * GetTrueP(int level )
    { 
        return Ps[level].get();
    }

    /// Displays realization of random field on embedded mesh
    void EmbedShowField(
        int ilevel, 
        const mfem::Vector & field,
        const std::string prefix, 
        std::ostream & socket)
    { 
        embed_glvis_plot(ilevel, field, prefix, socket);
    }
    
    /// Displays realization of random field on original mesh
    void ShowField(
        int ilevel,
        const mfem::Vector & field,
        const std::string prefix,
        std::ostream & socket)
    {
        glvis_plot(ilevel, field, prefix, socket);
    }
  
    /// Compute L2 Error of coeff and exact soln (double) for level
    double EmbedComputeL2Error(
        int level, 
        const mfem::Vector & coeff,
        double exact) const;

    double ComputeL2Error(
        int level,
        const mfem::Vector & coeff,
        double exact) const;

    /// Compute L2 Error of coeff and exact soln (double) for level
    double EmbedComputeMaxError(
        int level,
        const mfem::Vector & coeff,
        double exact) const;

    double ComputeMaxError(
        int level,
        const mfem::Vector & coeff,
        double exact) const;

    /// Save embedded fine mesh in glvis format
    void EmbedSaveMeshGLVis(const std::string prefix) const;

    /// Save original fine mesh in glvis format
    void SaveMeshGLVis(const std::string prefix) const;
    
    /// Save random field on embedded mesh in glvis format
    void EmbedSaveFieldGLVis(
        int level, 
        const mfem::Vector & coeff, 
        const std::string prefix) const;
    
    void EmbedSaveFieldGLVis_u(
        int level, 
        const mfem::Vector & coeff, 
        const std::string prefix) const;

    /// Save random field on original mesh in glvis format
    void SaveFieldGLVis(
        int level,
        const mfem::Vector & coeff,
        const std::string prefix) const;

    /// Transfer vector on embedded mesh to original mesh 
    void Transfer(
        int level,
        const mfem::Vector & coeff,
        mfem::Vector & orig_coeff) const;

    parelag::HypreExtension::HypreADSData adsData;
    parelag::HypreExtension::HypreAMSData amsData;

private:

    void embed_glvis_plot(
        int ilevel,
        const mfem::Vector & sol, 
        const std::string prefix, 
        std::ostream & socket) const;
    
    void glvis_plot(
        int ilevel,
        const mfem::Vector & sol,
        const std::string prefix,
        std::ostream & socket) const;

    /// Used for plotting and error evaluation
    void embed_prolongate_to_fine_grid(
        int ilevel, 
        const mfem::Vector & sol, 
        mfem::GridFunction & x) const;

    void embed_prolongate_to_fine_grid_u(
        int ilevel, 
        const mfem::Vector & sol, 
        mfem::GridFunction & x) const;
    
    /// Used for plotting and error evaluation
    void prolongate_to_fine_grid(
        int ilevel,
        const mfem::Vector & sol,
        mfem::GridFunction & x) const;

    /// Mesh of original domain
    const std::shared_ptr<mfem::ParMesh>& orig_mesh;
    
    /// Mesh of embedded domain
    const std::shared_ptr<mfem::ParMesh>& embed_mesh;
    
    /// Number of spatial dimensions
    const int nDim;

    const int uform;
    const int sform;

    /// Normal distribution sampler
    NormalDistributionSampler& dist_sampler;

    /// Number of levels
    int nLevels;

    mfem::FiniteElementSpace * fespace;
    mfem::FiniteElementSpace * fespace_u;
    
    /// Number of iterations on last linear solver call
    int iter;
   
    parelag::ParameterList& prob_list;

    const bool save_vtk;
    const bool lognormal;
    const bool verbose;
    
    /// Correlation length of matern covariance
    const double corlen;

    /// 1/corlen^2
    const double alpha;
   
    /// Matern scaling coefficient for SPDE
    const double matern_coeff;
    
    mfem::Array<int> level_size;
    mfem::Array<int> orig_level_size;
    mfem::Array<size_t> nnz;
    
    int myid;
    
    int num_procs;
 
    /// DeRham embed_sequence
    std::vector< std::shared_ptr<parelag::DeRhamSequence> > embed_sequence;

    std::vector< std::unique_ptr<mfem::HypreParMatrix> > W;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > D;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > A;
    
    std::vector< std::unique_ptr<mfem::Vector> > w_sqrt;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > Ps;
    std::vector< std::unique_ptr<mfem::Solver> > precA;
    std::vector< std::unique_ptr<mfem::IterativeSolver> > invA;
    
    /// Sparse matrices (local) of transfer from embedded mesh to original
    std::vector< std::unique_ptr<mfem::SparseMatrix> > meshP_s;
    
    /// Parallel matrices of transfer from embedded mesh to original
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > meshP;
    
};

} /* namespace parelagmc */
#endif /* EmbeddedPDESAMPLER_LEGACY_HPP_ */
