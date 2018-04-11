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
 
#include "MLMC_Manager.hpp"

#include <limits>
#include "Utilities.hpp"

namespace parelagmc
{
MLMC_Manager::MLMC_Manager(MPI_Comm comm_, 
            const int nlevels, 
            PhysicalMLSolver & pSolver_, 
            MLSampler & sampler_, 
            parelag::ParameterList& master_list):
        wallTime(true),
        comm(comm_),
        nlevels( nlevels ),
        pSolver(pSolver_),
        sampler(sampler_),
        prob_list(master_list.Sublist("Problem parameters", true)),
        eps2(prob_list.Get("Mean square error", 0.001)),
        auto_eps2(eps2 < 0 ? 1 : 0),
        ratio(prob_list.Get("MSE splitting ratio", 0.5)),
        file_name(prob_list.Get("Output filename for MC managers", "MLMC.dat")),
        init_nsamples(prob_list.Get("Number of samples", 10)),
        use_array_samples(prob_list.Get("Use array samples", false)),
        v_init_nsamples(prob_list.Get("Array number of samples", std::vector<int>())),
        ml_estimator_variance( std::numeric_limits<double>::infinity() ),
        expected_discretization_error2( std::numeric_limits<double>::infinity() ),
        actualMSE( std::numeric_limits<double>::infinity() ),
        sums(nlevels, NVAR),
        expectations(nlevels, NVAR),
        eY(nlevels),
        eABSY(nlevels),
        eQ(nlevels),
        eABSQ(nlevels),
        eC(nlevels),
        varY(nlevels),
        varQ(nlevels),
        consistency(nlevels),
        kurtosis(nlevels),
        M(nlevels),
        sampler_nnz(nlevels),
        physical_nnz(nlevels),
        VC(nlevels),
        alpha(0.),
        alphaABS(0.),
        beta(0.),
        gamma(0.),
        level_nsamples(nlevels),
        level_nsamples_missing(nlevels)
{
    MPI_Comm_size(comm, &rank);
    MPI_Comm_rank(comm, &pid);
 
    for(int i = 0; i < nlevels; ++i)
        M(i) = pSolver.GetGlobalNumberOfDofs(i);

    if(pid == 0)
        logger.open( file_name );

    // Create various timers
    for (int i = 0; i < nlevels; i++)
    {
        parelag::Timer timer = parelag::TimeManager::AddTimer(
            std::string("MC Sample -- Level ")
            .append(std::to_string(i)));
    }

    // Check 'Array number of samples' is correct length
    if (use_array_samples)
        if (static_cast<int>(v_init_nsamples.size()) != nlevels)
            use_array_samples = false;

    if (!use_array_samples)
        v_init_nsamples.assign(nlevels, init_nsamples); 

    // Print manager info
    if(!pid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
            << "*  MLMC_Manager \n"
            << "*    MSE: " << eps2 << '\n'
            << "*    MSE splitting ratio: " << ratio << '\n'
            << "*    Number of Initial Samples: ";
        for (auto i: v_init_nsamples)
            std::cout << i << " ";
        std::cout << "\n";
        std::cout << "*    Output filename: " << file_name << '\n';
        std::cout << std::string(50,'*') << '\n';
    }
}

void MLMC_Manager::InitRun(
        std::vector<int> & level_nsamples_init)
{
    if(level_nsamples.Max() == 0 && pid == 0)
        logger << "%" << std::setw(13) << "level " << std::setw(14) << "Y(xi) " 
                << std::setw(14) << "Q(xi)" << std::setw(14) << "Q_c(xi)" << std::setw(14) << "c \n";
    //Solve on the coarsest level
    {
        int ilevel = nlevels-1;
        int nsamples = level_nsamples_init[ilevel];
        for(int isample(0); isample < nsamples; ++isample)
        {
            {
                parelag::Timer timer = parelag::TimeManager::GetTimer(
                    std::string("MC Sample -- Level ")
                    .append(std::to_string(ilevel)));
                sampler.Sample(ilevel, xi);
                sampler.Eval(ilevel, xi, sparam);
                pSolver.SolveFwd(ilevel, sparam, q, c);
            }
            sums(ilevel, Y3) += q*q*q;
            sums(ilevel, Y4) += q*q*q*q;
            sums(ilevel, Y2) += q*q;
            sums(ilevel, Y ) += q;
            sums(ilevel, ABSY ) += fabs(q);
            sums(ilevel, Q2) += q*q;
            sums(ilevel, Q ) += q;
            sums(ilevel, ABSQ ) += fabs(q);
            sums(ilevel, C ) += c;

            if(!pid)
                logger << std::setw(14) << ilevel << std::setw(14) << q << 
                    std::setw(14) << q << std::setw(14) << "0" << std::setw(14) << c << "\n";
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
                    std::string("MC Sample -- Level ")
                    .append(std::to_string(ilevel))); 
                sampler.Sample(ilevel, xi);
                //sampler.Eval(ilevel+1, xi, sparam);
                sampler.Eval(ilevel+1, xi, sparam, init_s, false);
                pSolver.SolveFwd(ilevel+1, sparam, qc, cc);
                //sampler.Eval(ilevel, xi, sparam);
                sampler.Eval(ilevel, xi, sparam, init_s, true);
                pSolver.SolveFwd(ilevel, sparam, q, c);
            }
            y = q - qc;
            c = c + cc;
            sums(ilevel, Y3) += y*y*y;
            sums(ilevel, Y4) += y*y*y*y;
            sums(ilevel, Y2) += y*y;
            sums(ilevel, Y ) += y;
            sums(ilevel, ABSY ) += fabs(y);
            sums(ilevel, Q2) += q*q;
            sums(ilevel, Q ) += q;
            sums(ilevel, ABSQ ) += fabs(q);
            sums(ilevel, C ) += c;

            if(!pid)
                logger << std::setw(14) << ilevel << std::setw(14) << y << 
                    std::setw(14) << q << std::setw(14) << qc << std::setw(14) << c << "\n";
        }
        level_nsamples[ilevel] += nsamples;
    }
    if(pid == 0)
        logger << std::flush;
    computeNSamplesMSE();
}

void MLMC_Manager::Run()
{
    sums    = 0.;
    expectations = 0.;
    eY      = 0.;
    eABSY   = 0.;
    eQ      = 0.;
    eABSQ   = 0.;
    eC      = 0.;
    varY    = 0.;
    varQ    = 0.;
    consistency  = 0.;
    kurtosis = 0.;
    level_nsamples = 0;
    level_nsamples_missing = 0;
    
    InitRun(v_init_nsamples);

    std::vector<int> level_nsamples_grain;
    level_nsamples_grain.assign(nlevels, 0);

    while( ml_estimator_variance > ratio * eps2 )
    {
        for(int i(0); i < nlevels; ++i)
            level_nsamples_grain[i] = std::min(level_nsamples_missing[i], 
                    v_init_nsamples[i] + level_nsamples_grain[i] + 
                    level_nsamples_missing[i]/10);
        InitRun(level_nsamples_grain);
    }
    
    if (!pid)
        std::cout << "FINAL MLMC ERRORS" << std::endl;
    ShowMe();
}

void MLMC_Manager::ShowMe(std::ostream & os)
{
    for (int i = 0; i < nlevels; i++)
    {    
        physical_nnz(i) = pSolver.GetNNZ(i);
        sampler_nnz(i) = sampler.GetNNZ(i);
    }   
 
    int total_width = 79;
    int name_width = 40;
    
    if (!pid)
    {
        os.precision(8);
        os << std::string(total_width, '=') << std::endl;
        os << "MLMC Manager Errors: " << std::endl
           << std::string(total_width, '-') << std::endl
           << std::setw(name_width+2) << std::left << "Estimate"
           << std::setw(18) << std::left << eY.Sum() << '\n'
           << std::setw(name_width+2) << std::left << "Target MSE"
           << std::setw(18) << std::left << eps2 << '\n'
           << std::setw(name_width+2) << std::left << "Actual MSE"
           << std::setw(18) << std::left << actualMSE << '\n'
           << std::setw(name_width+2) << std::left << "ML Estimator Variance"
           << std::setw(18) << std::left << ml_estimator_variance << '\n'
           << std::setw(name_width+2) << std::left << "Estimator Bias"
           << std::setw(18) << std::left << expected_discretization_error2 << '\n'
           << std::setw(name_width+2) << std::left << "Alpha"
           << std::setw(18) << std::left << alpha << '\n'
           << std::setw(name_width+2) << std::left << "AlphaAbs"
           << std::setw(18) << std::left << alphaABS << '\n'
           << std::setw(name_width+2) << std::left << "Beta"
           << std::setw(18) << std::left << beta << "\n"
           << std::setw(name_width+2) << std::left << "Gamma"
           << std::setw(18) << std::left << gamma << "\n\n"

           << std::setw(name_width+2) << std::left << "DOFS in Forward Problem" 
           << std::setw(2) << std::left;
        M.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "C_l "
            << std::setw(2) << std::left;
        eC.Print(std::cout);
        os << '\n' << std::setw(name_width+2) << std::left << "NumSamples "
            << std::setw(2) << std::left;
        level_nsamples.Print(std::cout);
        os << '\n' << std::setw(name_width+2) << std::left << "E[Y_l] "
            << std::setw(2) << std::left;
        eY.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Y_l|] "
            << std::setw(2) << std::left;
        eABSY.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Y_l] "
            << std::setw(2) << std::left;
        varY.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[Q_l] "
            << std::setw(2) << std::left;
        eQ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "E[|Q_l|] "
            << std::setw(2) << std::left;
        eABSQ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Var[Q_l] "
            << std::setw(2) << std::left;
        varQ.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "V[Y_l]*C_l "
            << std::setw(2) << std::left;
        VC.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Consistency "
            << std::setw(2) << std::left;
        consistency.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "Kurtosis"
            << std::setw(2) << std::left;
        kurtosis.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "NNZ-Sampler"
            << std::setw(2) << std::left;
        sampler_nnz.Print(std::cout);
        os << std::setw(name_width+2) << std::left << "NNZ-ForwardSolve"
            << std::setw(2) << std::left;
        physical_nnz.Print(std::cout);
        os << std::string(total_width, '=') << std::endl;    
    }
 
}

//----------------------------------------------------------------------------------------------------------//
void MLMC_Manager::computeNSamplesMSE()
{
    // Compute eY, eQ, eC, varY, varQ, MSE, and target number of samples
    for(int ivar(0); ivar < NVAR; ++ivar)
        for(int ilevel(0); ilevel < nlevels; ++ilevel)
            expectations(ilevel, ivar) = sums(ilevel, ivar)/
                    static_cast<double>( level_nsamples[ilevel] );
    expectations.GetColumn(Y, eY);
    expectations.GetColumn(ABSY, eABSY);
    expectations.GetColumn(Q, eQ);
    expectations.GetColumn(ABSQ, eABSQ);
    expectations.GetColumn(C, eC);

    expectations.GetColumn(Y2, varY);
    expectations.GetColumn(Q2, varQ);
    expectations.GetColumn(Y4, kurtosis);
    for(int ilevel(0); ilevel < nlevels; ++ilevel)
    {
        kurtosis(ilevel) /= varY(ilevel)*varY(ilevel);
        varY(ilevel) -= eY(ilevel)*eY(ilevel);
        varY(ilevel) *= static_cast<double>(level_nsamples[ilevel]) 
            / static_cast<double>(level_nsamples[ilevel] - 1);
        varQ(ilevel) -= eQ(ilevel)*eQ(ilevel);
        varQ(ilevel) *= static_cast<double>(level_nsamples[ilevel]) 
            / static_cast<double>(level_nsamples[ilevel] - 1);
    }

    for (int ilevel(0); ilevel < nlevels-1; ++ilevel)
    {
        consistency(ilevel) = std::abs(eQ(ilevel) - eQ(ilevel+1) + eY(ilevel)) / 
            (3*(std::sqrt(varQ(ilevel)) + std::sqrt(varQ(ilevel+1)) + std::sqrt(varY(ilevel)))); 
    }

    alpha = expWRegression(eY, M,1);
    alphaABS = expWRegression(eABSY, M,1);
    beta    = expWRegression(varY, M,1);
    
    // Want to estimate (E[Q-Q_0])^2 where Q_0 or Q_h is computed on finest level
    // discretization error; independent of Var[Q]
    if(nlevels == 1)
        expected_discretization_error2 = 0.;
    else
    {
        double m = M(0)/M(1);
    
        if(nlevels > 3)
            expected_discretization_error2 = std::max(
                pow(m, 2.*alphaABS) * eABSY(1)*eABSY(1), eABSY(0)*eABSY(0)) 
                / ( pow(pow(m, -2.*alphaABS) - 1.,2));
        else if( nlevels ==3 )
            expected_discretization_error2 = (eABSY(0)*eABSY(0)) /
                ( pow(pow(m, -alphaABS) - 1., 2) );
        else if( nlevels == 2)
            expected_discretization_error2 = (eABSY(0)*eABSY(0));
    
    }

    if(auto_eps2)
        eps2 = expected_discretization_error2/(1.-ratio);

    ml_estimator_variance = 0.;
    for(int ilevel(0); ilevel < nlevels; ++ilevel)
        ml_estimator_variance += varY(ilevel)/
                static_cast<double>(level_nsamples[ilevel]);

    actualMSE = expected_discretization_error2 + ml_estimator_variance;

    mfem::Vector cost;
    if(wallTime)
    {
        // Get time for Level i 
        cost.SetSize(nlevels);
        for (int i = 0; i < nlevels; i++)
        {
            std::ostringstream name;
            name << "MC Sample -- Level " << i;
            double time = parelag::TimeManager::GetWatch(name.str()).GetElapsedTime();
            time /= static_cast<double>(level_nsamples[i]);
            cost[i] = time;
        }
    }
    else
        cost.SetDataAndSize(eC.GetData(), nlevels);

    gamma  = expWRegression(cost, M,0);
    
    double prop = 0.;
    for(int i(0); i < nlevels; ++i)
        prop += sqrt( varY(i) * cost(i) );
    prop /= ratio*eps2;

    for(int i(0); i < nlevels; ++i)
    {
        double missings = prop * sqrt( varY(i) / cost(i) );
        missings -= static_cast<double>(level_nsamples[i] );
        level_nsamples_missing[i] = std::max( 
                static_cast<int>( ceil(missings) ), 0);
        VC[i] = varY(i) * cost(i);
    }

    ShowMe();
}

} /* namespace parelagmc */
