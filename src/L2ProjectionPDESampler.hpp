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
 
#ifndef L2PROJECTIONPDESAMPLER_HPP_
#define L2PROJECTIONPDESAMPLER_HPP_

#include <cmath>
#include <memory>
#include <vector>

#include <elag.hpp>

#include "MLSampler.hpp"
#include "NormalDistributionSampler.hpp"

namespace parelagmc 
{
/// \class L2ProjectionPDESampler 
/// \brief Operator that samples a Gaussian random field with Matern Covariance 
///
/// A realization is found by solving the SPDE on an enlarged mesh then projecting
/// to the orignal mesh (the meshes can be non-matching and arbitrarilty distributed
/// across MPI processes).
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

class L2ProjectionPDESampler : public MLSampler
{
public:
    /// Constructor
    /// \param orig_mesh MFEM mesh object of mesh
    /// \param embed_mesh MFEM mesh oject of enlarged mesh
    /// \param dist_sampler Normal distribution sampler
    /// \param prec_params ParameterList
    L2ProjectionPDESampler(
        const std::shared_ptr<mfem::ParMesh>& orig_mesh_,                      
        const std::shared_ptr<mfem::ParMesh>& embed_mesh_,
        NormalDistributionSampler& dist_sampler,
        parelag::ParameterList& prec_params); 
    
    /// Destructor
    virtual ~L2ProjectionPDESampler() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    L2ProjectionPDESampler(L2ProjectionPDESampler const&) = delete;                                
    L2ProjectionPDESampler(L2ProjectionPDESampler&&) = delete;                                     
    L2ProjectionPDESampler&                                                              
        operator=(L2ProjectionPDESampler const&) = delete;                               
    L2ProjectionPDESampler&                                                              
        operator=(L2ProjectionPDESampler&&) = delete;                                    
    ///@}          

    /// Set the sequence of original mesh (if already computed from PhysicalMLSolver)  
    void SetDeRhamSequence(
        std::vector<std::shared_ptr<parelag::DeRhamSequence>> & orig_sequence);
    
    /// Compute the DeRhamSequence for each level
    /// First checks if orig_sequence has been set, then builds the DeRhamSequence
    /// orig_sequence (if necessary) and sequence for enlarged domain. 
    void BuildDeRhamSequence(
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & orig_topology,
        std::vector<std::shared_ptr<parelag::AgglomeratedTopology>> & topology);

    /// Build the hierarchy of linear operators, solvers, and L2-projection operator for each level
    void BuildHierarchy();

    /// Fills xi with a random sample using dist_sampler
    void Sample(
        const int level, 
        mfem::Vector & xi);

    /// Computes a realization on the original mesh
    /// Returns a sample from a Gaussian random field with Matern Covariance 
    /// function found by solving the SPDE on the embedded mesh then projecting
    /// to the orignal mesh.
    void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s);

    /// Computes a realization on the original mesh 
    /// Returns a sample from a Gaussian random field s and returns embed_s, the realization
    /// on the embedded mesh. If use_init = true, then embed_s is used as an initial guess
    void Eval(
        const int level, 
        const mfem::Vector & xi, 
        mfem::Vector & s,
        mfem::Vector & embed_s,
        bool use_init); 
    
    /// Computes a realization on the enlarged mesh (without projection)
    /// Returns a sample from a Gaussian random field with Matern Covariance 
    /// function found by solving the SPDE on the embedded mesh. 
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

    /// Returns global sample size for level for original mesh
    inline int GlobalSampleSize(int level) const
    {
        return orig_sequence[level]->GetDofHandler(sform)
            ->GetDofTrueDof().GetTrueGlobalSize();
    }

    /// Returns global sample size for level for original mesh
    inline int EmbedGlobalSampleSize(int level) const
    {
        return sequence[level]->GetDofHandler(sform)
            ->GetDofTrueDof().GetTrueGlobalSize();
    }

    /// Returns local dofs for original mesh
    inline int GetNumberOfDofs(int level) const
    {
        return orig_level_size[level] +
            orig_sequence[level]->GetNumberOfDofs(uform);
    }

    /// Returns global dofs for original mesh
    inline int GetGlobalNumberOfDofs(int level) const
    {
        return  orig_sequence[level]->GetDofHandler(uform)
                    ->GetDofTrueDof().GetTrueGlobalSize() +
                    orig_sequence[level]->GetDofHandler(sform)
                    ->GetDofTrueDof().GetTrueGlobalSize();
    }

    /// Returns TrueP prolongator for level for embedded mesh
    inline mfem::HypreParMatrix * GetEmbedTrueP(int level )
    { 
        return Ps[level].get();
    }
    
    /// Returns TrueP prolongator for level for embedded mesh
    inline mfem::HypreParMatrix * GetTrueP(int level )
    { 
        return orig_Ps[level].get();
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

    /// Save random field on original mesh in glvis format
    void SaveFieldGLVis(
        int level,
        const mfem::Vector & coeff,
        const std::string prefix) const;

    /// Save random field in H1 in glvis format
    void SaveFieldGLVis_H1(
        int level,
        const mfem::Vector & coeff,
        const std::string prefix) const;

    /// Save random field in H1 glvis format
    /// coeff in L2 on any level whereas mu is in H1 on level 0
    void SaveFieldGLVis_H1Add(
        int level,
        const mfem::Vector & coeff,
        const std::string prefix,
        const mfem::Vector & mu) const;

    /// Number of iterations of most recent call of Eval; now just -1
    inline int GetNumIters() const
    {
        return -1;
    }

    /// Number of nonzeros of most recent call of SolveFwd
    inline size_t GetNNZ(int level) const
    {
        return nnz[level];
    }

    /// Transfer vector on enlarged mesh to original mesh 
    void Transfer(
        int level,
        const mfem::Vector & coeff,
        mfem::Vector & orig_coeff) const;
    
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

    /// Used for plotting and error evaluation
    void prolongate_to_fine_grid_H1(
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

    /// Mesh of enlarged domain
    const std::shared_ptr<mfem::ParMesh>& embed_mesh;

    /// Normal distribution sampler
    NormalDistributionSampler& dist_sampler;

    /// Number of spatial dimensions
    const int nDim;
    
    const int uform;
    const int sform;

    /// Number of levels
    int nLevels;

    mfem::FiniteElementSpace * fespace;
    mfem::FiniteElementSpace * orig_fespace;

    parelag::ParameterList& master_list;
    parelag::ParameterList& prob_list;

    const bool entire_seq;
    const bool save_vtk;
    const bool lognormal;
    const bool verbose;
    
    /// Correlation length of matern covariance
    const double corlen;

    /// 1/corlen^2
    const double alpha;
    
    /// Matern scaling coefficient for SPDE
    const double matern_coeff;
    
    bool if_solver_hybridization;
    bool compute_orig_sequence;

    mfem::Array<int> level_size;
    mfem::Array<int> orig_level_size;
    mfem::Array<size_t> nnz;
    
    mfem::Array2D<int> block_offsets;
    mfem::Array2D<int> true_block_offsets;

    mfem::Array<parelag::SharingMap *> umap;
    mfem::Array<parelag::SharingMap *> pmap;

    int num_procs;
    int myid;

    /// DeRham sequence
    std::vector< std::shared_ptr<parelag::DeRhamSequence> > sequence;
    std::vector< std::shared_ptr<parelag::DeRhamSequence> > orig_sequence;


    std::vector< std::shared_ptr<parelag::MfemBlockOperator> > A;

    std::vector< std::unique_ptr<mfem::Vector> > w_sqrt;
    std::vector< std::unique_ptr<mfem::Vector> > orig_w_sqrt;

    std::vector< std::unique_ptr<mfem::HypreParMatrix> > Ps;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > orig_Ps;
    std::vector< std::unique_ptr<mfem::Solver> > invA;
    
    // L2 projection operators
    std::vector< std::shared_ptr<mfem::HypreParMatrix> > Gt; 

};

} /* namespace parelagmc */
#endif /* L2ProjectionPDESAMPLER_HPP_ */
