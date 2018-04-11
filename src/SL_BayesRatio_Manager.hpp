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
 
#ifndef SL_BAYESRATIO_MANAGER_HPP_
#define SL_BAYESRATIO_MANAGER_HPP_

#include <limits>
#include <fstream>
#include <elag.hpp>

#include "Utilities.hpp"
#include "BayesianInverseProblem.hpp"
#include "Sampling_Method_Manager.hpp"

namespace parelagmc
{
/// \class SL_BayesRatio_Manager
/// \brief Class that implements a serial SL ratio estimator manager
///
/// This computes E_post[Q] = E[R]/E[Z] (sum i.e. compute MC estimate, then divide)
/// where R=Q\cdot \Pi_like and Z = \Pi_like.
/// The desired mean-square error, initial number of samples,    
/// the MSE splitting ratio, and the output file name for the manager to 
/// print output are specified in the parelag::ParameterList params.

class SL_BayesRatio_Manager : public Sampling_Method_Manager
{
public:
    /// Constructor
    SL_BayesRatio_Manager(MPI_Comm comm, 
        BayesianInverseProblem & problem,  
        parelag::ParameterList& params);

    /// Destructor
    virtual ~SL_BayesRatio_Manager() = default;
    
    /// @{ \brief All special constructors and assignment are deleted
    SL_BayesRatio_Manager() = delete;
    SL_BayesRatio_Manager(SL_BayesRatio_Manager const&) = delete;
    SL_BayesRatio_Manager(SL_BayesRatio_Manager&&) = delete;
    SL_BayesRatio_Manager& operator=(SL_BayesRatio_Manager const&) = delete;
    SL_BayesRatio_Manager& operator=(SL_BayesRatio_Manager&&) = delete;
    ///@}

    /// Run MC simulation and compute necessary samples ``on the fly''
    void Run();

    /// MC simulation with level_nsamples
    void InitRun(int level_nsamples);

    /// Print current values of MC estimator
    void ShowMe(std::ostream & os = std::cout);
    
private:

    enum {R2 = 0, R = 1, ABS_R = 2, Z2 = 3, Z = 4, ABS_Z = 5,
        Ratio2 = 6, Ratio = 7, ABS_Ratio = 8, C = 9, T = 10, NVAR = 11};

    /// \f$N_l = eps2 / 2 *( 1/ sum_l sqrt(V_l C_l) ) * sqrt( V_l / C_l)\f$
    void computeNSamplesMSE();

    MPI_Comm comm;
    int rank;
    int pid;
    
    /// Has physical solver, sampler, and ComputeLikelihood
    BayesianInverseProblem & problem;

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

    /// eR[i]=E[R_i]
    mfem::Vector eR;

    /// eR[i]=E[abs(R_i)]
    mfem::Vector eABS_R;
       
    /// varR[i] = Var[R_i]
    mfem::Vector varR;
    
    /// eZ[i]=E[Z_i]
    mfem::Vector eZ;

    /// eZ[i]=E[abs(Z_i)]
    mfem::Vector eABS_Z;
       
    /// varZ[i] = Var[Z_i]
    mfem::Vector varZ;
    
    /// eRatio[i]=E[R_i/Z_i]
    mfem::Vector eRatio;

    /// eRatio[i]=E[abs(R_i/Z_i)]
    mfem::Vector eABS_Ratio;
       
    /// varRatio[i] = Var[R_i/Z_i]
    mfem::Vector varRatio;
    
    /// Cost (measured in degrees of freedom) of obtaining one sample of ratio 
    mfem::Vector eC;
    
    /// Number of (spatial) degrees of freedom on each level of forward problem
    mfem::Vector M;

    int level_nsamples;
    
    int level_nsamples_missing;

    bool wallTime;

    std::ofstream logger;

    // Used in InitRun
    mfem::Vector xi, sparam;
    double q=0., r=0., z=0., c=0., tot_c=0.;
};

SL_BayesRatio_Manager::SL_BayesRatio_Manager(MPI_Comm comm_, 
        BayesianInverseProblem & problem_,
        parelag::ParameterList& master_list):    
    comm(comm_),
    problem(problem_),
    prob_list(master_list.Sublist("Problem parameters", true)),
    eps2(prob_list.Get("Mean square error", 0.001)),
    auto_eps2(eps2 < 0 ? 1 : 0),
    ratio(prob_list.Get("MSE splitting ratio", 0.5)),
    file_name(prob_list.Get("Output filename for MC managers", "BayesRatio_Splitting.dat")),
    init_nsamples(prob_list.Get("Number of samples", 10)),
    ml_estimator_variance( std::numeric_limits<double>::infinity() ),
    expected_discretization_error2( std::numeric_limits<double>::infinity() ),
    actualMSE( std::numeric_limits<double>::infinity() ),
    sums(1, NVAR),
    expectations(1, NVAR),
    eR(1),
    eABS_R(1),
    varR(1),
    eZ(1),
    eABS_Z(1),
    varZ(1),
    eRatio(1),
    eABS_Ratio(1),
    varRatio(1),
    eC(1),
    M(1),
    level_nsamples(0),
    level_nsamples_missing(0),
    wallTime(true)
{

    MPI_Comm_size(comm, &rank);
    MPI_Comm_rank(comm, &pid);

    sums       = 0.;
    eR         = 0.;
    eABS_R     = 0.;
    varZ       = 0.;
    eR         = 0.;
    eABS_R     = 0.;
    varR       = 0.;
    eRatio     = 0.;
    eABS_Ratio = 0.;
    varRatio   = 0.;
    eC         = 0.;
    level_nsamples = 0;
    level_nsamples_missing = 0;

    M(0) = problem.GetSolver().GetNumberOfDofs(0);

    if(pid == 0)
        logger.open( file_name );

    // Create various timers
    {
        parelag::Timer timer = parelag::TimeManager::AddTimer(
            "Ratio MC Sample");
    }

    // Print manager info
    if(!pid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
            << "*  SL_BayesRatio_Manager \n"
            << "*    MSE: " << eps2 << '\n'
            << "*    MSE splitting ratio: " << ratio << '\n'
            << "*    Number of Initial Samples: " << init_nsamples << '\n'
            << "*    Output filename: " << file_name << '\n';
        std::cout << std::string(50,'*') << '\n';
    }
}

void SL_BayesRatio_Manager::InitRun(
        int level_nsamples_init)
{
    tot_c=0.;
    if(level_nsamples == 0 && pid == 0)
        logger << "%" << std::setw(13) << "level " << std::setw(14) << "R(xi) " 
                << std::setw(14) << "Z(xi) " << std::setw(14) << "Ratio(xi) " 
                << std::setw(14) << "c \n";
    //Solve 
    {
        const int nsamples = level_nsamples_init;
        std::ostringstream name;
        name << "MC Sample ";
        for(int isample(0); isample < nsamples; ++isample)
        {
            {
                parelag::Timer mc_timer = parelag::TimeManager::GetTimer(
                    "Ratio MC Sample");
                problem.SamplePrior(0, xi);
                problem.EvalPrior(0, xi, sparam);
                // Compute Z
                problem.ComputeLikelihood(0, sparam, z, c); tot_c += c;
                // Compute R = Q*pi_like with independent samples 
                problem.SamplePrior(0, xi);
                problem.EvalPrior(0, xi, sparam);
                problem.ComputeR(0, sparam, r, c); tot_c += c;
            }

            sums(0, R2) += r*r;
            sums(0, R) += r;
            sums(0, ABS_R) += fabs(r);
            sums(0, Z2) += z*z;
            sums(0, Z) += z;
            sums(0, ABS_Z) += fabs(z);
            //sums(0, Ratio2) += q*q;
            //sums(0, Ratio) += q;
            //sums(0, ABS_Ratio) += fabs(q);
            sums(0, C) += tot_c;

            if(pid == 0)
                logger << std::setw(14) << 0 << std::setw(14) << r 
                    << std::setw(14) << z 
                    << std::setw(14) << q 
                    << std::setw(14) << c << "\n"; 
        }

        level_nsamples += nsamples;
    }

    if(pid == 0)
        logger << std::flush;
    computeNSamplesMSE();
}

void SL_BayesRatio_Manager::Run()
{
    sums    = 0.;
    expectations = 0.;
    eR        = 0.;
    eABS_R     = 0.;
    varR      = 0.;
    eZ        = 0.;
    eABS_Z     = 0.;
    varZ      = 0.;
    eRatio    = 0.;
    eABS_Ratio = 0.;
    varRatio  = 0.;
    eC        = 0.;
    level_nsamples = 0;
    level_nsamples_missing = 0;

    int level_nsamples_grain = init_nsamples;
    InitRun(level_nsamples_grain);

    level_nsamples_grain = 0;

    while( ml_estimator_variance > ratio * eps2 )
    {
        level_nsamples_grain = std::min(level_nsamples_missing, 
                    init_nsamples + level_nsamples_grain + 
                    level_nsamples_missing/10);
        InitRun(level_nsamples_grain);
    }
    
    if (!pid)
        std::cout << "FINAL SL BayesRatio ERRORS" << std::endl;
    ShowMe();
}

void SL_BayesRatio_Manager::ShowMe(std::ostream & os)
{
    int total_width = 79;
    int name_width = 40;

    if(pid == 0)
    {
        os.precision(8);
        os << std::string(total_width, '=') << std::endl;
        os << "SL_BayesRatio_Manager Errors: " << std::endl
           << std::string(total_width, '-') << std::endl
           << std::setw(name_width+2) << std::left << "R Estimate"
           << std::setw(18) << std::left << eR.Sum() << '\n'
           << std::setw(name_width+2) << std::left << "Z Estimate"
           << std::setw(18) << std::left << eZ.Sum() << '\n'
           << std::setw(name_width+2) << std::left << "Ratio Estimate"
           << std::setw(18) << std::left << eRatio.Sum() << "\n\n"
           << std::setw(name_width+2) << std::left << "Target MSE"
           << std::setw(18) << std::left << eps2 << '\n'
           << std::setw(name_width+2) << std::left << "Actual MSE"
           << std::setw(18) << std::left << actualMSE << '\n'
           << std::setw(name_width+2) << std::left << "SL Estimator Variance"
           << std::setw(18) << std::left << ml_estimator_variance << '\n'
           << std::setw(name_width+2) << std::left << "Estimator Bias"
           << std::setw(18) << std::left << expected_discretization_error2 << '\n'
           << std::setw(name_width+2) << std::left << "DOFS in Forward Problem"
           << std::setw(2) << std::left;
        M.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "C_l "
            << std::setw(2) << std::left;
        eC.Print(std::cout);
        os << '\n' << std::setw(name_width+2) << std::left << "NumSamples "
            << std::setw(2) << std::left << level_nsamples << '\n';
        os << std::setw(name_width+2) << std::left << "E[R_l] "
            << std::setw(2) << std::left;
        eR.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|R_l|] "
            << std::setw(2) << std::left;
        eABS_R.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[R_l] "
            << std::setw(2) << std::left;
        varR.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[Z_l] "
            << std::setw(2) << std::left;
        eZ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Z_l|] "
            << std::setw(2) << std::left;
        eABS_Z.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Z_l] "
            << std::setw(2) << std::left;
        varZ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[Ratio_l] "
            << std::setw(2) << std::left;
        eRatio.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Ratio_l|] "
            << std::setw(2) << std::left;
        eABS_Ratio.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Ratio_l] "
            << std::setw(2) << std::left;
        varRatio.Print(std::cout);
        os << std::string(total_width, '=') << std::endl;
   }
}

//----------------------------------------------------------------------------------------------------------//
void SL_BayesRatio_Manager::computeNSamplesMSE()
{
    const double nl = static_cast<double>( level_nsamples );
    for(int ivar(0); ivar < NVAR; ++ivar)
        expectations(0, ivar) = sums(0, ivar)/nl; 

    expectations.GetColumn(R, eR);
    expectations.GetColumn(R2, varR);
    expectations.GetColumn(ABS_R, eABS_R);
    
    expectations.GetColumn(Z, eZ);
    expectations.GetColumn(Z2, varZ);
    expectations.GetColumn(ABS_Z, eABS_Z);
    
    //expectations.GetColumn(Ratio, eRatio);
    //expectations.GetColumn(Ratio2, varRatio);
    //expectations.GetColumn(ABS_Ratio, eABS_Ratio);
    
    expectations.GetColumn(C, eC);

    // Compute variances
    varR(0) -= eR(0)*eR(0);
    varR(0) *= nl / (nl - 1.);

    varZ(0) -= eZ(0)*eZ(0);
    varZ(0) *= nl / (nl - 1.);
   
    // Sum then divide
    eRatio(0) = eR(0) / eZ(0); 
    eABS_Ratio(0) = eABS_R(0) / eABS_Z(0);
    // R^2 / Z^2
    varRatio(0) = varR(0) / varZ(0); 
    varRatio(0) -= eRatio(0)*eRatio(0);
    varRatio(0) *= nl / (nl - 1.);
    
    // Want to estimate (E[Q-Q_0])^2 where Q_0 or Q_h is computed on finest level
    // discretization error; independent of Var[Q]
    // nlevels == 1: error very small since on finest level?
    expected_discretization_error2 = 0.;

    if(auto_eps2)
        eps2 = expected_discretization_error2/(1.-ratio);

    ml_estimator_variance = varRatio(0) / nl;

    actualMSE = expected_discretization_error2 + ml_estimator_variance;

    mfem::Vector cost;
    if(wallTime)
    {
        cost.SetSize(1);
        std::ostringstream name;
        name << "Ratio MC Sample";
        cost[0] = parelag::TimeManager::GetWatch(name.str()).GetElapsedTime() / nl;
    }
    else
        cost.SetDataAndSize(eC.GetData(), 1);

    const double prop = sqrt( varRatio(0) * cost(0) )/(ratio*eps2);
    {
        const double missings = prop * sqrt( varRatio(0) / cost(0) ) - nl;
        level_nsamples_missing = std::max( 
                static_cast<int>( ceil(missings) ), 0);
    }

    ShowMe();
}

} /* namespace parelagmc */
#endif /* SL_BayesRatio_Manager_HPP_ */
