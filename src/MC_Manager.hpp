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
 
#ifndef MC_MANAGER_HPP_
#define MC_MANAGER_HPP_

#include <elag.hpp>

#include "MLSampler.hpp"
#include "PhysicalMLSolver.hpp"

namespace parelagmc
{
/// \class MC_Manager
/// \brief Implements a serial single-level MC manager
///
/// The desired mean-square error, initial number of samples,    
/// the MSE splitting ratio, and the output file name for the manager to 
/// print output are specified in the parelag::ParameterList params. 
class MC_Manager
{
public:
    /// Constructor
    MC_Manager(MPI_Comm comm, 
        PhysicalMLSolver & pSolver, 
        MLSampler & sampler,
        parelag::ParameterList& params); 

    /// Destructor
    ~MC_Manager() = default;
   
    /// @{ \brief All special constructors and assignment are deleted.         
    MC_Manager(MC_Manager const&) = delete;                                
    MC_Manager(MC_Manager&&) = delete;                                     
    MC_Manager& operator=(MC_Manager const&) = delete;                               
    MC_Manager& operator=(MC_Manager&&) = delete;                                    
    ///@}           
  
    /// Run MC simulation and compute necessary samples ``on the fly''   
    void Run();

    /// MC simulation with level_nsamples
    void InitRun(int level_nsamples);

    /// Print current values of MC estimator
    void ShowMe(std::ostream & os = std::cout);
    
    /// Use wallTime or ndofs as cost
    bool wallTime;

private:

    enum {Q2 = 0, Q = 1, ABSQ = 2, C = 3, NVAR = 4};

    /// \f$N_l = eps2 / 2 *( 1/ sum_l sqrt(V_l C_l) ) * sqrt( V_l / C_l)\f$
    void computeNSamplesMSE();

    MPI_Comm comm;
    int rank;
    int pid;
    
    /// Solver
    PhysicalMLSolver & pSolver;

    /// Samples random field at level i and i+1
    MLSampler & sampler;

    /// Contains data for MC simulation
    parelag::ParameterList& prob_list;

    /// Target MSE
    double eps2;

    /// If eps2 < 0, determines eps2 based on (estimated) discretization error  
    const int auto_eps2;
   
    /// MSE splitting ratio: sampling_error < ratio*eps
    const double ratio;

    /// Output filename for MC manager
    const std::string file_name;
 
    /// Number of initial samples
    const int init_nsamples;

    /// ml_estimator_variance = \f$sum_l N_l^-1 V_l\f$
    double ml_estimator_variance;
    
    /// Estimator Bias expected_discretization_error2 = \f$yExpV[0]*yExpV[0]\f$
    double expected_discretization_error2;

    /// Actual MSE = expected_discretization_error2 + ml_estimator_variance
    double actualMSE;

    /// variance/mean at level i (i=0 Fine Grid)
    mfem::DenseMatrix sums;
    mfem::DenseMatrix expectations;

    /// eQ[i]=E[Q_i]
    mfem::Vector eQ;

    /// eQ[i]=E[abs(Q_i)]
    mfem::Vector eABSQ;
       
    /// Cost (measured in degrees of freedom) of obtaining one sample of Y_l 
    mfem::Vector eC;
    
    /// varQ[i] = Var[Q_i]
    mfem::Vector varQ;

    /// mfem::Vector of number of (spatial) degrees of freedom on each level
    mfem::Vector M;

    int level_nsamples;
    int level_nsamples_missing;

    std::ofstream logger;

    /// Used in InitRun
    mfem::Vector xi, sparam;
    double q = 0., c = 0.;
};

} /* namespace parelagmc */
#endif /* MC_Manager_HPP_ */
