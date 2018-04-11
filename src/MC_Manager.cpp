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
 
#include "MC_Manager.hpp"

#include <limits>
#include <fstream>

namespace parelagmc
{
MC_Manager::MC_Manager(MPI_Comm comm_, 
            PhysicalMLSolver & pSolver_, 
            MLSampler & sampler_, 
            parelag::ParameterList& master_list):       
        wallTime(true),     
        comm(comm_),
        pSolver(pSolver_),
        sampler(sampler_),
        prob_list(master_list.Sublist("Problem parameters", true)),
        eps2(prob_list.Get("Mean square error", 0.001)),
        auto_eps2(eps2 < 0 ? 1 : 0),
        ratio(prob_list.Get("MSE splitting ratio", 0.5)),
        file_name(prob_list.Get("Output filename for MC managers", "MLMC.dat")),
        init_nsamples(prob_list.Get("Number of samples", 10)),
        ml_estimator_variance( std::numeric_limits<double>::infinity() ),
        expected_discretization_error2( std::numeric_limits<double>::infinity() ),
        actualMSE( std::numeric_limits<double>::infinity() ),
        sums(1, NVAR),
        expectations(1, NVAR),
        eQ(1),
        eABSQ(1),
        eC(1),
        varQ(1),
        M(1),
        level_nsamples(0),
        level_nsamples_missing(0)
{

    MPI_Comm_size(comm, &rank);
    MPI_Comm_rank(comm, &pid);

    sums    = 0.;
    eQ      = 0.;
    eABSQ   = 0.;
    eC      = 0.;
    varQ    = 0.;
    level_nsamples = 0;
    level_nsamples_missing = 0;

    M(0) = pSolver.GetGlobalNumberOfDofs(0);

    if(pid == 0)
        logger.open( file_name );

    // Create timer 
    {
        parelag::Timer timer = parelag::TimeManager::AddTimer(
            "MC Sample ");
    }

    // Print manager info
    if(!pid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
            << "*  MC_Manager \n"
            << "*    MSE: " << eps2 << '\n'
            << "*    MSE splitting ratio: " << ratio << '\n'
            << "*    Number of Initial Samples: " << init_nsamples << '\n'
            << "*    Output filename: " << file_name << '\n';
        std::cout << std::string(50,'*') << '\n';
    }
}

void MC_Manager::InitRun(
        int level_nsamples_init)
{
    if(level_nsamples == 0 && pid == 0)
        logger << "%" << std::setw(13) << "level " << std::setw(14) << "Q(xi) " 
                << std::setw(14) << "c \n";
    //Solve 
    {
        int nsamples = level_nsamples_init;
        for(int isample(0); isample < nsamples; ++isample)
        {
            {
                parelag::Timer mc_timer = parelag::TimeManager::GetTimer(
                    "MC Sample ");
                sampler.Sample(0, xi);
                sampler.Eval(0, xi, sparam);
                pSolver.SolveFwd(0, sparam, q, c);
            }

            sums(0, Q2) += q*q;
            sums(0, Q ) += q;
            sums(0, ABSQ ) += fabs(q);
            sums(0, C ) += c;

            if(pid == 0)
                logger << std::setw(14) << 0 << std::setw(14) << q << 
                    std::setw(14) << c << "\n";
        }
        level_nsamples += nsamples;
    }

    if(pid == 0)
        logger << std::flush;
    computeNSamplesMSE();
}

void MC_Manager::Run()
{
    sums    = 0.;
    expectations = 0.;
    eQ      = 0.;
    eABSQ   = 0.;
    eC      = 0.;
    varQ    = 0.;
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
        std::cout << "FINAL SLMC ERRORS" << std::endl;
    ShowMe();
}

void MC_Manager::ShowMe(std::ostream & os)
{
    int total_width = 79;
    int name_width = 40;

    if(!pid)
    {
        os.precision(8);
        os << std::string(total_width, '=') << std::endl;
        os << "SLMC Manager Errors: " << std::endl
           << std::string(total_width, '-') << std::endl
           << std::setw(name_width+2) << std::left << "Estimate"
           << std::setw(18) << std::left << eQ.Sum() << '\n'
           << std::setw(name_width+2) << std::left << "Target MSE"
           << std::setw(18) << std::left << eps2 << '\n'
           << std::setw(name_width+2) << std::left << "Actual MSE"
           << std::setw(18) << std::left << actualMSE << '\n'
           << std::setw(name_width+2) << std::left << "SL Estimator Variance"
           << std::setw(18) << std::left << ml_estimator_variance << '\n'
           << std::setw(name_width+2) << std::left << "Estimator Bias"
           << std::setw(18) << std::left << expected_discretization_error2 << '\n'
           << std::setw(name_width+2) << std::left << "Target Bias Error"
           << std::setw(18) << std::left << std::sqrt(eps2)/sqrt(2) << '\n'

           << std::setw(name_width+2) << std::left << "DOFS in Forward Problem"
           << std::setw(2) << std::left;
        M.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "C_l "
            << std::setw(2) << std::left;
        eC.Print(std::cout);
        os << '\n' << std::setw(name_width+2) << std::left << "NumSamples "
            << std::setw(2) << std::left << level_nsamples << '\n';
        os << std::setw(name_width+2) << std::left << "E[Q_l] "
            << std::setw(2) << std::left;
        eQ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Q_l|] "
            << std::setw(2) << std::left;
        eABSQ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Q_l] "
            << std::setw(2) << std::left;
        varQ.Print(std::cout);
        os << std::string(total_width, '=') << std::endl;
   }
}

//----------------------------------------------------------------------------------------------------------//
void MC_Manager::computeNSamplesMSE()
{
    const double nl = static_cast<double>( level_nsamples );
    for(int ivar(0); ivar < NVAR; ++ivar)
        expectations(0, ivar) = sums(0, ivar) / nl;

    expectations.GetColumn(Q, eQ);
    expectations.GetColumn(ABSQ, eABSQ);
    expectations.GetColumn(C, eC);
    expectations.GetColumn(Q2, varQ);

    varQ(0) -= eQ(0)*eQ(0);
    varQ(0) *= nl / (nl - 1.); 

    // Want to estimate (E[Q-Q_0])^2 where Q_0 or Q_h is computed on finest level
    // discretization error; independent of Var[Q]
    // nlevels == 1: error very small since on finest level?
    expected_discretization_error2 = 0.;

    if(auto_eps2)
        eps2 = expected_discretization_error2/(1.-ratio);

    ml_estimator_variance = varQ(0) / nl;

    actualMSE = expected_discretization_error2 + ml_estimator_variance;

    mfem::Vector cost;
    if(wallTime)
    {
        cost.SetSize(1);
        std::ostringstream name;
        name << "MC Sample ";
        cost[0] = parelag::TimeManager::GetWatch(name.str()).GetElapsedTime() / nl;
    }
    else
        cost.SetDataAndSize(eC.GetData(), 1);
    
    const double prop = sqrt( varQ(0) * cost(0) )/(ratio*eps2);
    {
        const double missings = prop * sqrt( varQ(0) / cost(0) ) - nl;
        level_nsamples_missing = std::max( 
                static_cast<int>( ceil(missings) ), 0);
    }

    ShowMe();
}

} /* namespace parelagmc */
