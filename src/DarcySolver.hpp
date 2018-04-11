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
 
#ifndef DARCYSOLVER_HPP_
#define DARCYSOLVER_HPP_

#include <memory>
#include <elag.hpp>

#include "PhysicalMLSolver.hpp"

namespace parelagmc
{
/// \class DarcySolver
/// \brief Constructs the DeRham sequence to solve Darcy's Equation
///
/// Assembles the finite element matrices for the Darcy operator
///
///                           D = [ M(k)  B^T ]
///                               [ B     0   ]
///    where:
///
///    M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
///    B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
/// The solver/preconditioner strategy is specified via the ParameterList.
/// M(k) and the preconditioner is recomputed for each input k.
    
class DarcySolver : public PhysicalMLSolver
{
public:
    /// Constructor  
    DarcySolver( 
        const std::shared_ptr<mfem::ParMesh>& mesh,
        parelag::ParameterList& prec_params); 
    
    /// Destructor
    virtual ~DarcySolver() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    DarcySolver(DarcySolver const&) = delete;                                
    DarcySolver(DarcySolver&&) = delete;                                     
    DarcySolver& operator=(DarcySolver const&) = delete;                               
    DarcySolver& operator=(DarcySolver&&) = delete;                                    
    ///@}        

    /// Build DeRhamSequences, B, Bt, and prolongation operators for the hierarchy
    void BuildHierachySpaces(
        std::vector< std::shared_ptr <parelag::AgglomeratedTopology > > & topos, 
        std::unique_ptr<mfem::BilinearFormIntegrator> massIntegrator);

    /// Set obs_func
    void BuildVolumeObservationFunctional_P(
        mfem::LinearFormIntegrator * observationFunctional_p);

    /// Set obs_func
    void BuildVolumeObservationFunctional(
        mfem::LinearFormIntegrator * observationFunctional_u, 
        mfem::LinearFormIntegrator * observationFunctional_p);
    
    /// Set obs_func
    void BuildBdrObservationFunctional(
        mfem::LinearFormIntegrator * observationFunctional);
    
    void BuildPWObservationFunctional_p( 
        std::vector<double> & v_obs_data_coords, 
        const double eps = 0.01);
    
    /// Set ess_bc and ess_data
    void SetEssBdrConditions(
        mfem::Array<int> & ess_bc, 
        mfem::VectorCoefficient & u_bdr );
    
    /// Set the rhs
    void BuildForcingTerms(
        mfem::VectorCoefficient & f, 
        mfem::Coefficient & p_bdr, 
        mfem::Coefficient & q);

    /// Solve and update quantity of interest Q, cost C (number of dofs)
    void SolveFwd(
        int ilevel, 
        mfem::Vector & k_over_k_ref, 
        double & Q, 
        double & C);

    /// Solve and return the p-portion of soln, cost C (number of dofs)
    void SolveFwd_RtnPressure(
        int ilevel, 
        mfem::Vector & k_over_k_ref, 
        mfem::Vector & P, 
        double & C,
        double & Q,
        bool compute_Q);  
    
    /// Number of iterations of most recent call of SolveFwd; now just -1
    inline int GetNumIters() const 
    {
        return -1;
    }
    
    /// Number of nonzeros of most recent call of SolveFwd
    inline int GetNNZ(int level) const 
    {
        return nnz[level];
    }
    
    /// Return pressure space
    inline mfem::FiniteElementSpace * GetPressureSpace() const
    {
        return pspace;
    }
    
    /// Return velocity space
    inline mfem::FiniteElementSpace * GetVelocitySpace() const
    {
        return uspace;
    }

    inline int GetSizeOfStochasticData(int ilevel) const
    { 
        return sequence[ilevel]->GetNumberOfDofs(pform); 
    }

    inline int GetSizeOfGlobalStochasticData(int ilevel) const
    { 
        return sequence[ilevel]->GetDofHandler(pform)
           ->GetDofTrueDof().GetTrueGlobalSize(); 
    }
    
    inline int GetNumberOfDofs(int ilevel) const
    { 
        return sequence[ilevel]->GetNumberOfDofs(uform)
                +sequence[ilevel]->GetNumberOfDofs(pform); 
    }

    inline int GetGlobalNumberOfDofs(int ilevel) const 
    { 
        return sequence[ilevel]->GetDofHandler(uform)
                   ->GetDofTrueDof().GetTrueGlobalSize() 
               +sequence[ilevel]->GetDofHandler(pform)
                   ->GetDofTrueDof().GetTrueGlobalSize();
    }

    /// Returns the DeRham sequence
    inline std::vector<std::shared_ptr<parelag::DeRhamSequence>> & GetSequence()
    {
        return sequence;
    }

    void SaveMeshGLVis(const std::string prefix) const;

    /// Save u, p of coeff for visualization via GLVis
    void SaveFieldGLVis(
        int level, 
        const mfem::Vector & coeff, 
        const std::string prefix_u, 
        const std::string prefix_p) const;

    std::string prefix_u;

    std::string prefix_p;

private:

    std::shared_ptr<parelag::MfemBlockOperator> assemble(
        int ilevel, 
        mfem::Vector & k_over_k_ref,
        mfem::Vector & rhs_bc);

    void solve(
        int level, 
        const std::shared_ptr<parelag::MfemBlockOperator> A, 
        const mfem::Vector & k_over_k_ref,
        const mfem::Vector & rhs_bc, 
        mfem::Vector & sol);

    std::unique_ptr<mfem::Vector> parAssemble(
        int ilevel, 
        const mfem::Vector & rhs);

    std::unique_ptr<mfem::Vector> parDistribute(
        int ilevel, 
        const mfem::Vector & sol);

    void prolongate_to_fine_grid(
        int ilevel, 
        const mfem::Vector & coeff, 
        mfem::Vector & x) const;

    const std::shared_ptr<mfem::ParMesh>& mesh;
    
    /// Velocity space
    mfem::FiniteElementSpace *uspace;
    
    /// Pressure space
    mfem::FiniteElementSpace *pspace;
    
    int nLevels;
    
    mfem::Array2D<int> offsets;
    
    mfem::Array2D<int> true_offsets;
    
    const int nDimensions;
    
    const int uform;
    
    const int pform;

    /// Input list of parameters
    parelag::ParameterList& master_list;
    parelag::ParameterList& prob_list;
   
    const bool entire_seq;

    const bool verbose;
    
    const int saveGLVis;

    const int feorder;

    const int upscalingOrder;

    const std::string prec_type;
    
    int myid;

    int num_procs;
    
    mfem::Vector rhs_bc;
    
    std::vector<std::shared_ptr<parelag::DeRhamSequence>> sequence;

    mfem::Array<parelag::SharingMap *> umap;
    
    mfem::Array<parelag::SharingMap *> pmap;
    
    /// Interpolation matrices
    std::vector< std::unique_ptr<mfem::BlockMatrix> > P;
    
    /// Cochain projectors 
    std::vector< std::unique_ptr<mfem::BlockMatrix> > Pi;
    
    mfem::Array<int> ess_bc;
    std::vector<std::unique_ptr<mfem::SparseMatrix> > B;
    std::vector<std::unique_ptr<mfem::SparseMatrix> > Bt;
    
    std::vector<std::unique_ptr<mfem::Vector> > obs_func;
    
    std::vector<std::unique_ptr<mfem::Vector> > p_obs_func;

    std::vector<std::unique_ptr<mfem::Vector> > g_obs_func;
    
    std::vector<std::unique_ptr<mfem::Vector> > p_g_obs_func;
    
    std::vector<std::unique_ptr<mfem::Vector> > ess_data;
    
    /// int_\Omega f v + \int_\Partial\Omega p * vn + int_\Omega q qtest
    std::vector<std::unique_ptr<mfem::Vector> > rhs; 

    mfem::Array<size_t> nnz; 

    /// BCs for parelag's solver_state
    std::vector<std::vector<int>> ess_attr;
};

} /* namespace parelagmc */
#endif /* DARCYSOLVER_HPP_ */
