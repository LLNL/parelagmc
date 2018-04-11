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
 
#ifndef CREATEBAYESIANPARAMETERLIST_HPP__
#define CREATEBAYESIANPARAMETERLIST_HPP__

#include <elag.hpp>

namespace parelagmc
{
namespace examplehelpers
{
std::unique_ptr<parelag::ParameterList> CreateBayesianTestParameters()
{
    auto ret = parelag::make_unique<parelag::ParameterList>("Default");

    // Problem parameters
    {
        auto& prob_params = ret->Sublist("Problem parameters");

        prob_params.Set("Mesh file","BuildHexMesh");
        prob_params.Set("Embedded mesh file","BuildEmbedHexMesh");
        prob_params.Set("Serial refinement levels",0);
        prob_params.Set("Parallel refinement levels",2);
        prob_params.Set("Number of samples",10);
        prob_params.Set("Number boundary attributes",6);
        prob_params.Set("Essential attributes",std::vector<int>{0,1,1,1,1,0});
        prob_params.Set("Observational attributes",std::vector<int>{1,0,0,0,0,0});
        prob_params.Set("Inflow attributes",std::vector<int>{0,0,0,0,0,1});
        prob_params.Set("Lognormal",true);
        prob_params.Set("Sampler name","pde");
        prob_params.Set("Correlation length",0.1);
        prob_params.Set("nDimensions",3);
        prob_params.Set("Print timings", false);
    }// Problem parameters

    // Bayesian inverse problem parameters
    {
        auto& output_params = ret->Sublist("Bayesian inverse problem parameters");
        output_params.Set("Noise",0.1);
        output_params.Set("Number of observational data points", 1);
        output_params.Set("Observational data coordinates",std::vector<double>{1.0,1.0,1.0});
        output_params.Set("Generate reference observational data", true);
    }// Bayesian inverse problem parameters
    // Physical problem parameters
    {
        auto& output_params = ret->Sublist("Physical problem parameters");
        output_params.Set("Linear solver","MINRES-BJ-GS");
    }// Physical problem parameters

    // Sampler problem parameters
    {
        auto& output_params = ret->Sublist("Sampler problem parameters");
        output_params.Set("Linear solver","MINRES-BJ-GS");
    }// Sampler problem parameters
    // Preconditioner Library
    {
        auto& prec_lib = ret->Sublist("Preconditioner Library");

        // MINRES-BJ-GS
        {
            auto& list = prec_lib.Sublist("MINRES-BJ-GS");
            list.Set("Type", "Krylov");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Forms", "2 3");
                solver_list.Set("Solver name", "MINRES");
                solver_list.Set("Preconditioner","BJ-GS");
                solver_list.Set("Maximum iterations",300);
                solver_list.Set("Relative tolerance",1e-6);
                solver_list.Set("Absolute tolerance",1e-12);
            }
        }// MINRES-BJ-GS

        // BJ-GS 
        {
            auto& list = prec_lib.Sublist("BJ-GS");
            list.Set("Type","Block Jacobi");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("A00 Inverse","Gauss-Seidel");
                solver_list.Set("A11 Inverse","BoomerAMG Solver");
                solver_list.Set("S Type","Diagonal");
            }
        }// BJ-GS

        // Gauss-Seidel
        {
            auto& list = prec_lib.Sublist("Gauss-Seidel");
            list.Set("Type","Hypre");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Type","L1 Gauss-Seidel");
                solver_list.Set("Sweeps",3);
                solver_list.Set("Damping Factor",1.0);
                solver_list.Set("Omega",1.0);
                solver_list.Set("Cheby Poly Order",2);
                solver_list.Set("Cheby Poly Fraction",0.3);
            }
        }// Gauss-Seidel

        // BoomerAMG Solver
        {
            auto& list = prec_lib.Sublist("BoomerAMG Solver");
            list.Set("Type","BoomerAMG");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Coarsening type",10);
                solver_list.Set("Aggressive coarsening levels",1);
                solver_list.Set("Relaxation type",8);
                solver_list.Set("Theta",0.25);
                solver_list.Set("Interpolation type",6);
                solver_list.Set("P max",4);
                solver_list.Set("Print level",0);
                solver_list.Set("Dim",1);
                solver_list.Set("Maximum levels",25);
                solver_list.Set("Tolerance",0.0);
                solver_list.Set("Maximum iterations",1);
            }
        }// BoomerAMG Solver

    }// Preconditioner Library

    return ret;
}


}// namespace examplehelpers
}// namespace parelagmc
#endif /* CREATEBAYESIANPARAMETERLIST */
