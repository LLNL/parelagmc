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
 
#ifndef BAYESIANINVERSEPROBLEM_HPP_
#define BAYESIANINVERSEPROBLEM_HPP_

#include <memory>
#include <elag.hpp>

#include "MLSampler.hpp"
#include "PhysicalMLSolver.hpp"
#include "Utilities.hpp"

namespace parelagmc
{
/// \class BayesianInverseProblem
/// \brief This class implements many functions necessary for a Bayesian inverse
/// problems. A prior distribution (parelagmc::MLSampler) and a forward model
/// problem (parelagmc::PhysicalMLSolver) is specified. 

class BayesianInverseProblem 
{
public:
    /// Constructor  
    BayesianInverseProblem(
            const std::shared_ptr<mfem::ParMesh>& mesh,
            PhysicalMLSolver & solver,
            MLSampler & prior,
            parelag::ParameterList& master_list_);
    
    /// Destructor
    ~BayesianInverseProblem() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    BayesianInverseProblem(BayesianInverseProblem const&) = delete;                                
    BayesianInverseProblem(BayesianInverseProblem&&) = delete;                                     
    BayesianInverseProblem&                                                              
        operator=(BayesianInverseProblem const&) = delete;                               
    BayesianInverseProblem&                                                              
        operator=(BayesianInverseProblem&&) = delete;                                    
    ///@}        

    /// Will either read in reference obs data from file or generate y=G_obs+noise
    void GenerateObservationalData();

    /// Computes parameter-to-observable map that maps the uncertain input
    /// parameter to the observation data. 
    void ComputeG(
        int ilevel,
        mfem::Vector & k_over_k_ref,
        mfem::Vector & G,
        double & C,
        double & Q,
        bool compute_Q); 
    
    /// Evaluate the likelihood, cost C (number of dofs)
    void ComputeLikelihood(
        int ilevel,
        mfem::Vector & k_over_k_ref,
        double & likelihood,
        double & C);
    
    /// Evaluate the likelihood, cost C (number of dofs)
    void ComputeLikelihoodAndQ(
        int ilevel,
        mfem::Vector & k_over_k_ref,
        double & likelihood,
        double & C,
        double & Q);
    
    /// Computes R = Q x Likelihood, for use in a ratio estimator. 
    void ComputeR(
        int ilevel,
        mfem::Vector & k_over_k_ref,
        double & R,
        double & C);

    inline void SamplePrior(const int level, mfem::Vector & xi)
    {
        prior.Sample(level, xi); 
    }
    
    inline void EvalPrior(const int level, const mfem::Vector & xi, mfem::Vector & s)
    {
        prior.Eval(level, xi, s);
    }
    
    PhysicalMLSolver & GetSolver()
    {
        return solver;
    }

    MLSampler & GetPrior()
    {
        return prior;
    }
    
    void SaveObsPointsOfInterestGLVis(const std::string prefix)const; 

private:
    
    void prolongate_to_fine_grid(
        int ilevel,
        const mfem::Vector & coeff,
        mfem::Vector & x) const;

    const std::shared_ptr<mfem::ParMesh>& mesh;
    
    PhysicalMLSolver & solver;

    std::vector<std::shared_ptr<parelag::DeRhamSequence>> solver_sequence;
    
    MLSampler & prior;
    
    const int nLevels;
    
    /// Spatial dimension 
    const int dim;
    
    /// Input list for bayesian parameters
    parelag::ParameterList& bayesian_list;

    const double noise;

    /// Number of observational points
    const int m;
    
    /// Epsilon for local average pressure
    const double h;

    std::vector<double> v_obs_data_coords;
    
    /// Actual size of obs data: m, except of m == 0 then size_obs_data = 1
    int size_obs_data; 

    std::unique_ptr<mfem::FiniteElementCollection> fec;

    std::unique_ptr<mfem::FiniteElementSpace> fespace;

    int myid;

    int num_procs;

    /// Observational data
    std::unique_ptr<mfem::Vector> G_obs;
    
    std::vector< std::vector <std::unique_ptr<mfem::Vector>>> g_obs_func;
    
    std::unique_ptr<mfem::Vector> p;
    
    mfem::Vector Gl;    

    double q = 0., c = 0.;
};

} /* namespace parelagmc */
#endif /* BAYESIANINVERSEPROBLEM_HPP_ */
