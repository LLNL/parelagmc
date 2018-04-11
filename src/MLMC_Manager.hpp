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
 
#ifndef MLMC_MANAGER_HPP_
#define MLMC_MANAGER_HPP_

#include <fstream>
#include <elag.hpp>

#include "PhysicalMLSolver.hpp"
#include "MLSampler.hpp"

namespace parelagmc
{
/// \class MLMC_Manager
/// \brief Implements a serial multi-level MC manager
///
/// The desired mean-square error, initial number of samples ( or an array of 
/// initial samples for each level ), the MSE splitting ratio, and the 
/// output file name for the manager to print output are specified in the 
/// parelag::ParameterList params. 
class MLMC_Manager
{
public:
    /// Constructor
    MLMC_Manager(MPI_Comm comm, 
        const int nlevels, 
        PhysicalMLSolver & pSolver, 
        MLSampler & sampler, 
        parelag::ParameterList& params);

    /// Destructor
    ~MLMC_Manager() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    MLMC_Manager(MLMC_Manager const&) = delete;    
    MLMC_Manager(MLMC_Manager&&) = delete;         
    MLMC_Manager& operator=(MLMC_Manager const&) = delete;                 
    MLMC_Manager& operator=(MLMC_Manager&&) = delete;                      
    ///@}           

    /// Run MC simulation and compute necessary samples ``on the fly''
    /// Initial number of samples on each level is determined in ParameterList
    void Run();
    
    /// MC simulation with level_nsamples
    void InitRun(std::vector<int> & level_nsamples);

    /// Print current values of MC estimator
    void ShowMe(std::ostream & os = std::cout);

    /// Use wallTime or ndofs as cost
    bool wallTime;

private:

    enum {Y2 = 0, Y = 1, ABSY = 2, Q2 = 3, Q = 4, ABSQ = 5, C = 6, Y3 = 7, Y4 = 8, NVAR = 9};

    /// \f$N_l = eps2 / 2 *( 1/ sum_l sqrt(V_l C_l) ) * sqrt( V_l / C_l)\f$
    void computeNSamplesMSE();

    MPI_Comm comm;
    int rank;
    int pid;
    
    /// Number of levels in MLMC hierarchy
    const int nlevels;

    /// Solver
    PhysicalMLSolver & pSolver;

    /// Samples random field at level i and i+1
    MLSampler & sampler;

    /// Contains data for MLMC simulation
    parelag::ParameterList& prob_list;
    
    /// Target MSE
    double eps2;
    
    /// If eps2 < 0, determines eps2 based on (estimated) discretization error
    const int auto_eps2;
    
    /// MSE splitting ratio: sampling_error < ratio*eps
    const double ratio;

    /// Output filename for MLMC manager
    const std::string file_name;

    /// Number of initial samples
    const int init_nsamples;

    /// Whether or not to specify an arry of initial samples 
    bool use_array_samples;

    /// Vector of initial samples
    std::vector<int> v_init_nsamples;

    /// ml_estimator_variance = \f$sum_l N_l^-1 V_l\f$
    double ml_estimator_variance;
    
    /// Estimator Bias expected_discretization_error2 = \f$yExpV[0]*yExpV[0]\f$
    double expected_discretization_error2;

    /// Actual MSE = expected_discretization_error2 + ml_estimator_variance
    double actualMSE;
    
    /// variance/mean at level i (i=0 Fine Grid)
    mfem::DenseMatrix sums;
    mfem::DenseMatrix expectations;

    /// eY[i] = E[Q_i - Q_{i+1}]
    mfem::Vector eY;

    /// eABSY[i] =E[ abs(Q_i - Q_{i+1})]
    mfem::Vector eABSY;

    /// eQ[i]=E[Q_i]
    mfem::Vector eQ;

    /// eQ[i]=E[abs(Q_i)]
    mfem::Vector eABSQ;
       
    /// Cost (measured in degrees of freedom) of obtaining one sample of Y_l 
    mfem::Vector eC;
    
    /// varY[i]=Var[Q_i - Q_{i+1}]
    mfem::Vector varY;

    /// varQ[i] = Var[Q_i]
    mfem::Vector varQ;

    /// Consistency check:  If >1, indicates the identity E[Q_i - Q_{i+1}] = E[Q_i] - E[Q_{i+1}] is not satisfied 
    mfem::Vector consistency;
    
    mfem::Vector kurtosis;
    
    /// mfem::Vector of number of (spatial) degrees of freedom on each level
    mfem::Vector M;

    mfem::Vector sampler_nnz;
    
    mfem::Vector physical_nnz;
    
    /// VarY[i]*Cost[i] 
    mfem::Vector VC;
    
    /// | E[ Y - Y_l ] | <= M_l^alpha
    double alpha;

    /// E[ |Y - Y_l| ] <= M_l^alphaABS
    double alphaABS;

    /// Var[ Y_l ] <= M_l^beta
    double beta;

    /// T_l <= M_l^gamma where T_l cost (measured in seconds) of calling Sample, Eval, SolveFwd for nsamples  
    double gamma;
    
    //locals
    mfem::Array<int> level_nsamples;
    mfem::Array<int> level_nsamples_missing;

    std::ofstream logger;

    /// Used in InitRun
    mfem::Vector xi, sparam, init_s;
    double y=0., q=0., qc=0., c=0., cc=0;

};

} /* namespace parelagmc */
#endif /* MLMC_Manager_HPP_ */
