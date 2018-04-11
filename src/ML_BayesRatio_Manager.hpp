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
 
#ifndef ML_BAYESRATIO_MANAGER_HPP_
#define ML_BAYESRATIO_MANAGER_HPP_

#include <limits>
#include <fstream>

#include <elag.hpp>

#include "Utilities.hpp"
#include "BayesianInverseProblem.hpp"
#include "Sampling_Method_Manager.hpp"

namespace parelagmc
{
/// \class ML_BayesRatio_Manager 
/// \brief Implements a serial multi-level MC manager for a ratio estimator to 
/// estimate the Bayesian posterior expectation of a QoI, that is              
/// E_post[Q]=E[R]/E[Z] (subtract, then divide i.e. sum(r-rc)/sum(z-zc)). 
///
/// The desired mean-square error, initial number of samples, the MSE splitting 
/// ratio, and the output file name for the manager to print output are  
/// specified in the parelag::ParameterList params.

class ML_BayesRatio_Manager : public Sampling_Method_Manager
{
public:
    /// Constructor
    ML_BayesRatio_Manager(MPI_Comm comm, 
        const int nlevels, 
        BayesianInverseProblem & problem,
        parelag::ParameterList& params);

    /// Destructor
    virtual ~ML_BayesRatio_Manager() = default;

    /// @{ \brief All special constructors and assignment are deleted.         
    ML_BayesRatio_Manager(ML_BayesRatio_Manager const&) = delete;    
    ML_BayesRatio_Manager(ML_BayesRatio_Manager&&) = delete;         
    ML_BayesRatio_Manager&                                                
        operator=(ML_BayesRatio_Manager const&) = delete;                 
    ML_BayesRatio_Manager&                                                
        operator=(ML_BayesRatio_Manager&&) = delete;                      
    ///@}    

    /// Run MC simulation and compute necessary samples ``on the fly''
    void Run();

    /// MC simulation with level_nsamples_init
    void InitRun(std::vector<int> & level_nsamples_init);

    /// Print current values of MC estimator
    void ShowMe(std::ostream & os = std::cout);

private:
    
    enum {YZ2 = 0, YZ = 1, ABS_YZ = 2, Z2 = 3, Z = 4, ABS_Z = 5,
        YR2 = 6, YR = 7, ABS_YR = 8, R2 = 9, R = 10, ABS_R = 11, 
        YRatio2 = 12, YRatio = 13, ABS_YRatio = 14, Ratio2 = 15, 
        Ratio = 16, ABS_Ratio = 17, C = 18, T = 19, NVAR = 20};

    /// \f$N_l = eps2 / 2 *( 1/ sum_l sqrt(V_l C_l) ) * sqrt( V_l / C_l)\f$
    void computeNSamplesMSE();

    MPI_Comm comm;
    int rank;
    int pid;
    
    /// Number of levels in parelagmc hierarchy
    const int nlevels;

    /// Has physical solver, sampler, and ComputeLikelihood
    BayesianInverseProblem & problem;

    /// Contains data for MLMC simulation
    parelag::ParameterList& prob_list;

    /// Target MSE
    double eps2;

    /// If eps2 < 0, determines eps2 based on (estimated) discretization error
    const int auto_eps2;

    /// MSE splitting ratio: sampling_error < ratio*eps
    const double ratio;

    /// Output filename for ML_BayesRatio_Manager
    const std::string file_name;

    /// Number of initial samples
    const int init_nsamples;

    /// ml_estimator_variance = \f$sum_l N_l^-1 V_l\f$
    double ml_estimator_variance;
    
    /// ml_estimator_variance for R
    double ml_estimator_variance_R;

    /// ml_estimator_variance for Z
    double ml_estimator_variance_Z;

    /// Estimator Bias expected_discretization_error2 = \f$yExpV[0]*yExpV[0]\f$
    double expected_discretization_error2;

    /// Estimator Bias for R
    double expected_discretization_error2_R;
    
    /// Estimator Bias for Z
    double expected_discretization_error2_Z;
   
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

    /// eYR[i] = E[R_i - R_{i+1}]
    mfem::Vector eYR;

    /// eABSYR[i] =E[ abs(R_i - R_{i+1})]
    mfem::Vector eABS_YR;

    /// varYR[i]=Var[R_i - R_{i+1}]
    mfem::Vector varYR;

    /// eZ[i]=E[Z_i]
    mfem::Vector eZ;

    /// eZ[i]=E[abs(Z_i)]
    mfem::Vector eABS_Z;

    /// varZ[i] = Var[Z_i]
    mfem::Vector varZ;

    /// eY_Z[i] = E[Z_i - Z_{i+1}]
    mfem::Vector eYZ;

    /// eABSYZ[i] =E[ abs(Z_i - Z_{i+1})]
    mfem::Vector eABS_YZ;

    /// varYZ[i]=Var[Z_i - Z_{i+1}]
    mfem::Vector varYZ;
    
    /// eRatio[i]=E[R_i/Z_i]
    mfem::Vector eRatio;

    /// eRatio[i]=E[abs(R_i/Z_i)]
    mfem::Vector eABS_Ratio;

    /// varRatio[i] = Var[R_i/Z_i]
    mfem::Vector varRatio;

    /// eYRatio[i] = E[R_i/Z_i - R_{i+1}/Z_{i+1}]
    mfem::Vector eYRatio;

    /// eABSYRatio[i] =E[ abs(R_i/Z_i - R_{i+1}/Z_{i+1})]
    mfem::Vector eABS_YRatio;

    /// varYRatio[i]=Var[R_i/Z_i - R_{i+1}/Z_{i+1}]
    mfem::Vector varYRatio;

    /// Cost (measured in degrees of freedom) of obtaining one sample  
    mfem::Vector eC;

    /// mfem::Vector of number of (spatial) degrees of freedom on each level
    mfem::Vector M;

    /// | E[ X - X_l ] | <= M_l^alpha_X
    double alpha;
    double alpha_R;
    double alpha_Z;

    /// E[ |X - X_l| ] <= M_l^alphaABS_X
    double alphaABS;
    double alphaABS_R;
    double alphaABS_Z;
    
    /// Var[ Y_l ] <= M_l^beta
    double beta;
    double beta_R;
    double beta_Z;
 
    /// T_l <= M_l^gamma where T_l cost (measured in seconds)   
    double gamma;
    double gamma_R;
    double gamma_Z;

    mfem::Array<int> level_nsamples;
    mfem::Array<int> level_nsamples_missing;

    bool wallTime;

    std::ofstream logger;

    // Used in InitRun
    mfem::Vector xi, zxi;
    mfem::Vector sparam, zparam;
    double y=0., y_r=0., y_z=0., r=0., rc=0., z=0., zc=0., c=0.;
    double c_tot=0.;
};

ML_BayesRatio_Manager::ML_BayesRatio_Manager(MPI_Comm comm_, 
            const int nlevels_, 
            BayesianInverseProblem & problem_,
            parelag::ParameterList& master_list):    
        comm(comm_),
        nlevels( nlevels_ ),
        problem(problem_),
        prob_list(master_list.Sublist("Problem parameters", true)),
        eps2(prob_list.Get("Mean square error", 0.001)),
        auto_eps2(eps2 < 0 ? 1 : 0),
        ratio(prob_list.Get("MSE splitting ratio", 0.5)),
        file_name(prob_list.Get("Output filename for MC managers", "ML_BayesRatio.dat")),
        init_nsamples(prob_list.Get("Number of samples", 10)),
        ml_estimator_variance( std::numeric_limits<double>::infinity() ),
        ml_estimator_variance_R( std::numeric_limits<double>::infinity() ),
        ml_estimator_variance_Z( std::numeric_limits<double>::infinity() ),
        expected_discretization_error2( std::numeric_limits<double>::infinity() ),
        expected_discretization_error2_R( std::numeric_limits<double>::infinity() ),
        expected_discretization_error2_Z( std::numeric_limits<double>::infinity() ),
        actualMSE( std::numeric_limits<double>::infinity() ),
        sums(nlevels, NVAR),
        expectations(nlevels, NVAR),
        eR(nlevels),
        eABS_R(nlevels),
        varR(nlevels),
        eYR(nlevels),
        eABS_YR(nlevels),
        varYR(nlevels),
        eZ(nlevels),
        eABS_Z(nlevels),
        varZ(nlevels),
        eYZ(nlevels),
        eABS_YZ(nlevels),
        varYZ(nlevels),
        eRatio(nlevels),
        eABS_Ratio(nlevels),
        varRatio(nlevels),
        eYRatio(nlevels),
        eABS_YRatio(nlevels),
        varYRatio(nlevels),
        eC(nlevels),
        M(nlevels),
        alpha(0.),
        alpha_R(0.),
        alpha_Z(0.),
        alphaABS(0.),
        alphaABS_R(0.),
        alphaABS_Z(0.),
        beta(0.),
        beta_R(0.),
        beta_Z(0.),
        gamma(0.),
        gamma_R(0.),
        gamma_Z(0.),
        level_nsamples(nlevels),
        level_nsamples_missing(nlevels),
        wallTime(true)
{

    MPI_Comm_size(comm, &rank);
    MPI_Comm_rank(comm, &pid);

    level_nsamples = 0;
    level_nsamples_missing = 0;

    for(int i = 0; i < nlevels; ++i)
        M(i) = problem.GetSolver().GetNumberOfDofs(i); 

    if(pid == 0)
        logger.open( file_name );

    // Create various timers
    for (int i = 0; i < nlevels; i++)
    {
        parelag::Timer timer = parelag::TimeManager::AddTimer(
            std::string("Ratio MC Sample -- Level ")
            .append(std::to_string(i)));
    }

    // Print manager info
    if(!pid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
            << "*  ML_BayesRatio_Manager \n"
            << "*    MSE: " << eps2 << '\n'
            << "*    MSE splitting ratio: " << ratio << '\n'
            << "*    Number of Initial Samples: " << init_nsamples << '\n';
        std::cout << "*    Output filename: " << file_name << '\n';
        std::cout << std::string(50,'*') << '\n';
    }
}

void ML_BayesRatio_Manager::InitRun(
        std::vector<int> & level_nsamples_init)
{
    c_tot=0.;
    if(level_nsamples.Max() == 0 && pid == 0)
        logger << "%" << std::setw(13) << "level " << std::setw(14) << "R(xi) " 
                << std::setw(14) << "Y_R(xi) " << std::setw(14) << "Z(xi)" 
                << std::setw(14) << "Y_Z(xi) " <<std::setw(14) << "c \n";
    
    //Solve on the coarsest level
    {
        int ilevel = nlevels-1;
        int nsamples = level_nsamples_init[ilevel];
        for(int isample(0); isample < nsamples; ++isample)
        {
            {
                parelag::Timer timer = parelag::TimeManager::GetTimer(
                    std::string("Ratio MC Sample -- Level ")
                    .append(std::to_string(ilevel)));

                problem.SamplePrior(ilevel, zxi);
                problem.EvalPrior(ilevel, zxi, zparam);
                // Compute Z
                problem.ComputeLikelihood(ilevel, zparam, z, c); c_tot += c;
                // Compute R = Q*pi_like with independent samples 
                problem.SamplePrior(ilevel, xi);
                problem.EvalPrior(ilevel, xi, sparam);
                problem.ComputeR(ilevel, sparam, r, c); c_tot += c;
            }
            
            sums(ilevel, R ) += r;
            sums(ilevel, ABS_R ) += fabs(r);
            sums(ilevel, R2) += r*r;
            sums(ilevel, YR ) += r;
            sums(ilevel, ABS_YR ) += fabs(r);
            sums(ilevel, YR2) += r*r;
            
            sums(ilevel, Z) += z;
            sums(ilevel, ABS_Z) += fabs(z);
            sums(ilevel, Z2) += z*z;
            sums(ilevel, YZ ) += z;
            sums(ilevel, ABS_YZ ) += fabs(z);
            sums(ilevel, YZ2) += z*z;
            
            sums(ilevel, C ) += c_tot;
            c_tot = 0.;
                
            if(pid == 0)
                logger << std::setw(14) << ilevel << std::setw(14) << r << 
                    std::setw(14) << r << std::setw(14) << z << 
                    std::setw(14) << z << std::setw(14) << c_tot << 
                    std::setw(14) << "\n"; 
        }
        level_nsamples[ilevel] += nsamples;
    }

    //Solve on the other levels
    for(int ilevel(nlevels-2); ilevel >= 0; --ilevel)
    {
        int nsamples = level_nsamples_init[ilevel];
        for(int isample(0); isample < nsamples; ++isample)
        {
            {
                parelag::Timer timer = parelag::TimeManager::GetTimer(
                    std::string("Ratio MC Sample -- Level ")
                    .append(std::to_string(ilevel)));

                problem.SamplePrior(ilevel, zxi);
                problem.EvalPrior(ilevel, zxi, zparam);
                // Compute Z
                problem.ComputeLikelihood(ilevel, zparam, z, c); c_tot += c;
                // Compute R = Q*pi_like with independent samples 
                problem.SamplePrior(ilevel, xi);
                problem.EvalPrior(ilevel, xi, sparam);
                problem.ComputeR(ilevel, sparam, r, c); c_tot += c;
            
                // Compute Z_c
                problem.EvalPrior(ilevel+1, zxi, zparam);
                problem.ComputeLikelihood(ilevel+1, zparam, zc, c); c_tot += c;

                // Compute Rc = Qc*pi_like
                problem.EvalPrior(ilevel+1, xi, sparam);
                problem.ComputeR(ilevel+1, sparam, rc, c); c_tot += c;
                
                y_r = r - rc;
                y_z = z - zc;
            }
            
            sums(ilevel, R ) += r;
            sums(ilevel, ABS_R ) += fabs(r);
            sums(ilevel, R2) += r*r;
            sums(ilevel, YR ) += y_r;
            sums(ilevel, ABS_YR ) += fabs(y_r);
            sums(ilevel, YR2) += y_r*y_r;
            
            sums(ilevel, Z) += z;
            sums(ilevel, ABS_Z) += fabs(z);
            sums(ilevel, Z2) += z*z;
            sums(ilevel, YZ ) += y_z;
            sums(ilevel, ABS_YZ ) += fabs(y_z);
            sums(ilevel, YZ2) += y_z*y_z;
            
            sums(ilevel, C ) += c_tot;
            c_tot = 0.;
            
            if(pid == 0)
                logger << std::setw(14) << ilevel << std::setw(14) << r << 
                    std::setw(14) << y_r << std::setw(14) << z << 
                    std::setw(14) << y_z << std::setw(14) << c << 
                    std::setw(14) << "\n"; 
        }
        level_nsamples[ilevel] += nsamples;
    }
    if(pid == 0)
        logger << std::flush;
    computeNSamplesMSE();
}

void ML_BayesRatio_Manager::Run()
{
    expectations   = 0.;
    sums           = 0.;
    eR             = 0.;
    eABS_R         = 0.;
    varR           = 0.;
    eYR            = 0.;
    eABS_YR        = 0.;
    varYR          = 0.;
    eZ             = 0.;
    eABS_Z         = 0.;
    varZ           = 0.;
    eYZ            = 0.;
    eABS_YZ        = 0.;
    varYZ          = 0.;
    eRatio         = 0.;
    eABS_Ratio     = 0.;
    varRatio       = 0.;
    eYRatio        = 0.; 
    eABS_YRatio    = 0.;
    varYRatio      = 0.;   
    eC             = 0.;
    
    level_nsamples = 0;
    level_nsamples_missing = 0;

    std::vector<int> v_init_nsamples;
    v_init_nsamples.assign(nlevels, init_nsamples);
    InitRun(v_init_nsamples);

    std::vector<int> level_nsamples_grain;
    level_nsamples_grain.assign(nlevels, 0);

    while( ml_estimator_variance > ratio * eps2 )
    {
        for(int i(0); i < nlevels; ++i)
            level_nsamples_grain[i] = std::min(level_nsamples_missing[i], 
                    init_nsamples + level_nsamples_grain[i] + 
                    level_nsamples_missing[i]/10);
        InitRun(level_nsamples_grain);
    }

    if (!pid)
        std::cout << "FINAL ML_BayesRatio_Manager ERRORS" << std::endl;
    ShowMe();
}

void ML_BayesRatio_Manager::ShowMe(std::ostream & os)
{
    int total_width = 79;
    int name_width = 40;

    if(pid == 0)
    {
        os.precision(8);
        os << std::string(total_width, '=') << std::endl;
        os << "ML_BayesRatio_Manager Errors: " << std::endl
           << std::string(total_width, '-') << std::endl
           << std::setw(name_width+2) << std::left << "R Estimate"
           << std::setw(18) << std::left << eYR.Sum() << '\n'
           << std::setw(name_width+2) << std::left << "Z Estimate"
           << std::setw(18) << std::left << eYZ.Sum() << '\n'
           << std::setw(name_width+2) << std::left << "Ratio Estimate"
           << std::setw(18) << std::left << eYR.Sum()/eYZ.Sum() << "\n\n"
           << std::setw(name_width+2) << std::left << "Target MSE"
           << std::setw(18) << std::left << eps2 << '\n'
           << std::setw(name_width+2) << std::left << "Actual MSE"
           << std::setw(18) << std::left << actualMSE << '\n'
           << std::setw(name_width+2) << std::left << "ML Estimator Variance"
           << std::setw(18) << std::left << ml_estimator_variance << '\n'
           << std::setw(name_width+2) << std::left << "Estimator Bias (Max of R,Z)"
           << std::setw(18) << std::left << expected_discretization_error2 << '\n'
           << std::setw(name_width+2) << std::left << "DOFS in Forward Problem"
           << std::setw(2) << std::left;
        M.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Cost (dofs)"
            << std::setw(2) << std::left;
        eC.Print(std::cout);
        os << '\n' << std::setw(name_width+2) << std::left << "NumSamples "
            << std::setw(2) << std::left;
        level_nsamples.Print(std::cout);
        os << "\n";
        os << std::setw(name_width+2) << std::left << "Alpha_R"
           << std::setw(18) << std::left << alpha_R << '\n'
           << std::setw(name_width+2) << std::left << "AlphaAbs_R"
           << std::setw(18) << std::left << alphaABS_R << '\n'
           << std::setw(name_width+2) << std::left << "Beta_R"
           << std::setw(18) << std::left << beta_R << "\n"
           << std::setw(name_width+2) << std::left << "Gamma_R"
           << std::setw(18) << std::left << gamma_R << "\n";
        os << std::setw(name_width+2) << std::left << "E[R] "
           << std::setw(2) << std::left;
        eR.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|R|] "
            << std::setw(2) << std::left;
        eABS_R.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[R] "
            << std::setw(2) << std::left;
        varR.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[Y_R] "
            << std::setw(2) << std::left;
        eYR.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Y_R|] "
            << std::setw(2) << std::left;
        eABS_YR.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Y_R] "
            << std::setw(2) << std::left;
        varYR.Print(std::cout);
        os << "\n";
        os << std::setw(name_width+2) << std::left << "Alpha_Z"
           << std::setw(18) << std::left << alpha_Z << '\n'
           << std::setw(name_width+2) << std::left << "AlphaAbs_Z"
           << std::setw(18) << std::left << alphaABS_Z << '\n'
           << std::setw(name_width+2) << std::left << "Beta_Z"
           << std::setw(18) << std::left << beta_Z << "\n"
           << std::setw(name_width+2) << std::left << "Gamma_Z"
           << std::setw(18) << std::left << gamma_Z << "\n";
        os << std::setw(name_width+2) << std::left << "E[Z] "
            << std::setw(2) << std::left;
        eZ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Z|] "
            << std::setw(2) << std::left;
        eABS_Z.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Z] "
            << std::setw(2) << std::left;
        varZ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[Y_Z] "
            << std::setw(2) << std::left;
        eYZ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Y_Z|] "
            << std::setw(2) << std::left;
        eABS_YZ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Y_Z] "
            << std::setw(2) << std::left;
        varYZ.Print(std::cout);
        os << std::string(total_width, '=') << std::endl;
   }
}

//----------------------------------------------------------------------------------------------------------//
void ML_BayesRatio_Manager::computeNSamplesMSE()
{

    for(int ivar(0); ivar < NVAR; ++ivar)
        for(int ilevel(0); ilevel < nlevels; ++ilevel)
            expectations(ilevel, ivar) = sums(ilevel, ivar)/
                    static_cast<double>( level_nsamples[ilevel] );

    expectations.GetColumn(R, eR);
    expectations.GetColumn(R2, varR);
    expectations.GetColumn(ABS_R, eABS_R);
    expectations.GetColumn(YR, eYR);
    expectations.GetColumn(YR2, varYR);
    expectations.GetColumn(ABS_YR, eABS_YR);
   
    expectations.GetColumn(Z, eZ);
    expectations.GetColumn(Z2, varZ);
    expectations.GetColumn(ABS_Z, eABS_Z);
    expectations.GetColumn(YZ, eYZ);
    expectations.GetColumn(YZ2, varYZ);
    expectations.GetColumn(ABS_YZ, eABS_YZ);
   
    expectations.GetColumn(C, eC);

    // Compute variances
    for(int ilevel(0); ilevel < nlevels; ++ilevel)
    {
        varR(ilevel) -= eR(ilevel)*eR(ilevel);
        varR(ilevel) *= static_cast<double>(level_nsamples[ilevel]) 
            / static_cast<double>(level_nsamples[ilevel] - 1);
        varYR(ilevel) -= eYR(ilevel)*eYR(ilevel);
        varYR(ilevel) *= static_cast<double>(level_nsamples[ilevel]) 
            / static_cast<double>(level_nsamples[ilevel] - 1);
        
        varZ(ilevel) -= eZ(ilevel)*eZ(ilevel);
        varZ(ilevel) *= static_cast<double>(level_nsamples[ilevel]) 
            / static_cast<double>(level_nsamples[ilevel] - 1);
        varYZ(ilevel) -= eYZ(ilevel)*eYZ(ilevel);
        varYZ(ilevel) *= static_cast<double>(level_nsamples[ilevel]) 
            / static_cast<double>(level_nsamples[ilevel] - 1);
        
    }

    // Get cost i.e. time
    mfem::Vector cost;
    if(wallTime)
    {
        // Get time for Level i 
        cost.SetSize(nlevels);
        for (int i = 0; i < nlevels; i++)
        {
            std::ostringstream name;
            name << "Ratio MC Sample -- Level " << i;
            double time = parelag::TimeManager::GetWatch(
                name.str()).GetElapsedTime();
            time /= static_cast<double>(level_nsamples[i]);
            cost[i] = time;
        }
    }
    else
        cost.SetDataAndSize(eC.GetData(), nlevels);

    alpha_R    = expWRegression(eYR, M,1);
    alphaABS_R = expWRegression(eABS_YR, M,1);
    beta_R     = expWRegression(varYR, M,1);
    gamma_R    = expWRegression(cost, M,0);
    
    alpha_Z    = expWRegression(eYZ, M,1);
    alphaABS_Z = expWRegression(eABS_YZ, M,1);
    beta_Z     = expWRegression(varYZ, M,1);
    gamma_Z    = expWRegression(cost, M,0);
    
    // Compute estimator bias for R and Z, then determine the max
    // Want to estimate (E[Q-Q_0])^2 where Q_0 or Q_h is computed on finest level
    // discretization error; independent of Var[Q]
    if(nlevels == 1)
        expected_discretization_error2_R = 0.;
    else
    {
        double m = M(0)/M(1);
        if(nlevels > 3)
            expected_discretization_error2_R = std::max(
                pow(m, 2.*alphaABS_R) * eABS_YR(1)*eABS_YR(1), eABS_YR(0)*eABS_YR(0))
                / ( pow(pow(m, -2.*alphaABS_R) - 1.,2));
        else if( nlevels ==3 )
            expected_discretization_error2_R = (eABS_YR(0)*eABS_YR(0)) /
                ( pow(pow(m, -alphaABS_R) - 1., 2) );
        else if( nlevels == 2)
            expected_discretization_error2_R = (eABS_YR(0)*eABS_YR(0));
    }

    if(nlevels == 1)
        expected_discretization_error2_Z = 0.;
    else
    {
        double m = M(0)/M(1);
        if(nlevels > 3)
            expected_discretization_error2_Z = std::max(
                pow(m, 2.*alphaABS_Z) * eABS_YZ(1)*eABS_YZ(1), eABS_YZ(0)*eABS_YZ(0))
                / ( pow(pow(m, -2.*alphaABS_Z) - 1.,2));
        else if( nlevels ==3 )
            expected_discretization_error2_Z = (eABS_YZ(0)*eABS_YZ(0)) /
                ( pow(pow(m, -alphaABS_Z) - 1., 2) );
        else if( nlevels == 2)
            expected_discretization_error2_Z = (eABS_YZ(0)*eABS_YZ(0));
    }

    expected_discretization_error2 = std::max(expected_discretization_error2_R, 
        expected_discretization_error2_Z);

    if(auto_eps2)
        eps2 = expected_discretization_error2/(1.-ratio);


    ml_estimator_variance_Z = 0.;
    for(int ilevel(0); ilevel < nlevels; ++ilevel)
        ml_estimator_variance_Z += varYZ(ilevel)/
                static_cast<double>(level_nsamples[ilevel]);

    ml_estimator_variance_R = 0.;
    for(int ilevel(0); ilevel < nlevels; ++ilevel)
        ml_estimator_variance_R += varYR(ilevel)/
                static_cast<double>(level_nsamples[ilevel]);
   
    ml_estimator_variance = std::max(ml_estimator_variance_Z,
        ml_estimator_variance_R);
 
    actualMSE = expected_discretization_error2 + ml_estimator_variance;

    // FIXME: need cost_Z and cost_R
    double prop_R = 0.;
    for(int i(0); i < nlevels; ++i)
        prop_R += sqrt( varYR(i) * cost(i) );
    prop_R /= ratio*eps2;

    double prop_Z = 0.;
    for(int i(0); i < nlevels; ++i)
        prop_Z += sqrt( varYZ(i) * cost(i) );
    prop_Z /= ratio*eps2;
    
    for(int i(0); i < nlevels; ++i)
    {
        double missings_R = prop_R * sqrt( varYR(i) / cost(i) );
        missings_R -= static_cast<double>(level_nsamples[i] );
        double missings_Z = prop_Z * sqrt( varYZ(i) / cost(i) );
        missings_Z -= static_cast<double>(level_nsamples[i] );
        level_nsamples_missing[i] = std::max( std::max( 
                static_cast<int>( ceil(missings_R) ),
                static_cast<int>( ceil(missings_Z) )), 0);
    }

    ShowMe();
}

} /* namespace parelagmc */
#endif /* ML_BAYESRATIO_MANAGER_HPP_ */
