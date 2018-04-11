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
 
#ifndef L2PROJECTIONPDESAMPLER_LEGACY_HPP_
#define L2PROJECTIONPDESAMPLER_LEGACY_HPP_

#include <memory>
#include <vector>
#include <elag.hpp>

#include "MLSampler.hpp"
#include "NormalDistributionSampler.hpp"

namespace parelagmc
{
/// \class L2ProjectionPDESampler_Legacy 
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
/// By computing the Schur Complement with respect to u we get
///
/// A u = - g/kappa2 D^t W^{1/2} xi, A = M + 1/kappa2 B^tWB
///
/// which is solved with CG preconditioned with hypre's ADS.
/// Then, the realization if found by computing
///
/// s = 1/kappa2 D u + g/kappa2 W^{-1/2} xi 

class L2ProjectionPDESampler_Legacy : public MLSampler
{
public:
    /// Constructor
    /// \param orig_mesh MFEM mesh object of mesh
    /// \param embed_mesh MFEM mesh oject of enlarged mesh
    /// \param dist_sampler Normal distribution sampler
    /// \param prec_params ParameterList
    L2ProjectionPDESampler_Legacy(
        const std::shared_ptr<mfem::ParMesh>& orig_mesh_,                      
        const std::shared_ptr<mfem::ParMesh>& embed_mesh_,
        NormalDistributionSampler& dist_sampler,
        parelag::ParameterList& prec_params); 
    
    /// Destructor
    virtual ~L2ProjectionPDESampler_Legacy() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    L2ProjectionPDESampler_Legacy(L2ProjectionPDESampler_Legacy const&) = delete;                                
    L2ProjectionPDESampler_Legacy(L2ProjectionPDESampler_Legacy&&) = delete;                                     
    L2ProjectionPDESampler_Legacy&                                                              
        operator=(L2ProjectionPDESampler_Legacy const&) = delete;                               
    L2ProjectionPDESampler_Legacy&                                                              
        operator=(L2ProjectionPDESampler_Legacy&&) = delete;                                    
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
    /// Writes a sample from a Gaussian random field s on original mesh
    /// and embed_s, the realization on the embedded mesh. 
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

    /// Number of iterations of most recent call of Eval
    inline int GetNumIters() const
    {
        return iter;
    }

    inline size_t GetNNZ(int level) const
    {
        return nnz[level];
    }

    /// Returns mesh prolongator for level (embedded -> original)
    inline mfem::HypreParMatrix * GetMeshP(int level )
    {
        return MeshPs[level].get();
    }
 
    /// Returns TrueP prolongator for level for embedded mesh                  
    inline mfem::HypreParMatrix * GetEmbedTrueP(int level )                    
    {                                                                          
        return Ps[level].get();                                                
    }                    
 
    /// Returns TrueP prolongator for level
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

    /// Transfer vector on enlarged mesh to original mesh 
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

    /// Used for plotting and error evaluation
    void prolongate_to_fine_grid(
        int ilevel,
        const mfem::Vector & sol,
        mfem::GridFunction & x) const;

    /// Mesh of original domain                                                
    const std::shared_ptr<mfem::ParMesh>& orig_mesh;                           
                                                                               
    /// Mesh of embedded domain                                                
    const std::shared_ptr<mfem::ParMesh>& embed_mesh;

    mfem::FiniteElementSpace * fespace;
    mfem::FiniteElementSpace * orig_fespace;

    /// Number of spatial dimensions
    const int nDim;

    const int uform;
    const int sform;
   
    /// Normal distribution sampler
    NormalDistributionSampler& dist_sampler;

    /// Number of levels
    int nLevels;

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

    int iter;
    bool compute_orig_sequence;

    int num_procs;
    int myid;

    /// DeRham sequence
    std::vector< std::shared_ptr<parelag::DeRhamSequence> > sequence;
    std::vector< std::shared_ptr<parelag::DeRhamSequence> > orig_sequence;

    mfem::Array<int> level_size;
    mfem::Array<int> orig_level_size;
    mfem::Array<size_t> nnz;
    
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > W;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > D;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > A;
    std::vector< std::unique_ptr<mfem::Vector> > w_sqrt;                       
    std::vector< std::unique_ptr<mfem::Vector> > orig_w_sqrt;

    // Assembled sparse matrices of prolongators to original mesh
    std::vector< std::unique_ptr<mfem::SparseMatrix> > mesh_p;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > Ps;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > orig_Ps;
    std::vector< std::unique_ptr<mfem::HypreParMatrix> > MeshPs;
    std::vector< std::unique_ptr<mfem::Solver> > precA;
    std::vector< std::shared_ptr<mfem::HypreParMatrix> > Gt; 
    std::vector< std::unique_ptr<mfem::IterativeSolver> > invA;
};

} /* namespace parelagmc */
#endif /* L2ProjectionPDESAMPLER_LEGACY_HPP_ */
