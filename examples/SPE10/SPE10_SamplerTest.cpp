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
 
#include <fstream>
#include <memory>
#include <elag.hpp>

#include "NormalDistributionSampler.hpp"
#include "PDESampler.hpp"
#include "MLSampler.hpp"
#include "KLSampler.hpp"
#include "MaternCovariance.hpp"
#include "AnalyticExponentialCovariance.hpp"
#include "Utilities.hpp"
#include "MeshUtilities.hpp"

using namespace parelagmc;
using namespace parelag;
using namespace mfem;

// Computes realizations of a random field (Gaussian or log-normal) on the 
// SPE10 domain using a KLE with an analytic exponential covariance or matern 
// covariance function or SPDE sampler (without mesh-embedding).
int main (int argc, char *argv[])
{
  // Initialize MPI
  mpi_session sess(argc,argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int num_procs, myid;
  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &myid);
  try 
  {
    // Get options from command line
    const char *xml_file_c = "spe10_3D_parameters.xml";
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&xml_file_c, "-f", "--xml-file", "XML parameter list.");
    args.Parse();
    PARELAG_ASSERT(args.Good());
    std::string xml_file(xml_file_c);

    // Read the parameter list from file
    std::unique_ptr<ParameterList> master_list;
    {
        std::ifstream xml_in(xml_file);
        if (!xml_in.good())
        {
            std::cerr << "ERROR: Unable to obtain a good filestream "
                      << "from input file: " << xml_file << ".\n";
            return EXIT_FAILURE;
        }

        SimpleXMLParameterListReader reader;
        master_list = reader.GetParameterList(xml_in);

        xml_in.close();
    }
    // Standard problem parameters
    ParameterList& prob_list = master_list->Sublist("Problem parameters",true);
    // SPE10 specific params
    ParameterList& spe_list = master_list->Sublist("SPE10 problem parameters",true);
    std::string permFile = spe_list.Get("SPE10 permFile", "data/spe_perm.dat");
    const int nDimensions = spe_list.Get("nDimensions", 3);
    std::vector<int> v_N = spe_list.Get("Number of elements", std::vector<int>{60, 220, 85});
    std::vector<double> v_h = spe_list.Get("Element sizes", std::vector<double>{20, 10, 2});
    
    // Number of mesh refinements
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);

    // AMGe hierarchy specifications
    int nLevels = prob_list.Get("Number of levels", 2);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);

    // Output options
    const bool verbose = prob_list.Get("Verbosity",false);
    const bool print_time = prob_list.Get("Print timings",true);
    const bool visualize = prob_list.Get("Visualize",false);

    // Uncertainty parameters
    const int nsamples = prob_list.Get("Number of samples", 10);
    const double variance = prob_list.Get("Variance", 1.0);
    std::string sampler_name = prob_list.Get("Sampler name", "pde");
    const bool lognormal = prob_list.Get("Lognormal", false);

    if (!unstructured) nLevels = par_ref_levels + 1;

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  SPE10_sampler_test.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  nDimensions: " << nDimensions << '\n';
        if (nDimensions == 2) std::cout << "*  Slice: " << par_ref_levels << '\n';
        std::cout << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Samples: " << nsamples << '\n'
                 << "*  Number of elements: ";
        for (const auto i: v_N) std::cout << i << ' ';
        std::cout << "\n*  Element sizes: "; 
        for (const auto i: v_h) std::cout << i << ' ';
            std::cout << "*  Sampler type:" ;
        if (sampler_name == "analytic" || sampler_name == "matern")
            std::cout << " KL with " << sampler_name << " covariance " << std::endl;
        else
            std::cout << " PDE sampler " << std::endl;
        std::cout << std::string(50,'*') << '\n';

    }

    // Create the finite element mesh
    std::unique_ptr<Mesh> mesh = Create_SPE10_Mesh(nDimensions, v_N, v_h);

    if (!myid && verbose) std::cout << "-- Serial mesh refinement \n";
    for (int l = 0; l < ser_ref_levels; l++)
        mesh->UniformRefinement();

    auto pmesh = std::make_shared<ParMesh>(comm, *mesh);
    
    // Free the serial mesh
    mesh.reset();
    Array<int> level_nElements(nLevels);

    if (!myid && verbose) std::cout << "-- Parallel mesh refinement \n"; 
    for (int l = 0; l < par_ref_levels; l++)
    {
        if (!unstructured)
            level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    if (!unstructured) level_nElements[0] = pmesh->GetNE();

    if (!myid && verbose) std::cout << "-- Agglomerating the meshes\n";
    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);
    if (unstructured)
        BuildTopologyAlgebraic(pmesh, coarsening_factor, topology);
    else
        BuildTopologyGeometric(pmesh, level_nElements, topology);

    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    if (!myid && verbose) std::cout << "-- Generate CovarianceFunction\n";
    std::unique_ptr<CovarianceFunction>  cov_func;
    std::unique_ptr<MLSampler> sampler;
    if (sampler_name.compare("analytic") == 0)
    {
        if (myid == 0) std::cout << "-- Build AnalyticExponentialCovariance " << std::endl;
        cov_func = make_unique<AnalyticExponentialCovariance>(                 
                pmesh, *master_list);
        sampler = make_unique<KLSampler>(pmesh, dist, *cov_func, *master_list);
    }
    else if (sampler_name.compare("matern") == 0)
    {
        if (myid == 0) std::cout << "-- Build MaternCovariance " << std::endl;
        cov_func = make_unique<MaternCovariance>(pmesh, *master_list);   
        sampler = make_unique<KLSampler>(pmesh, dist, *cov_func, *master_list);
    }
    else // PDE
    {
        if (myid == 0) std::cout << "-- Build PDESampler " << std::endl;
        sampler = make_unique<PDESampler>(pmesh, dist, *master_list);
    }

    if (!myid && verbose) std::cout << "-- BuildHierarchy\n";
    sampler->BuildDeRhamSequence(topology);
    sampler->BuildHierarchy();
 
    Vector xi, coef;
    std::vector< std::unique_ptr<HypreParVector >> chi(nLevels);
    chi[0].reset(chi_center_of_mass(pmesh.get()));
    std::string name = "center";
    chi[0]->Print(name.c_str());
    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        chi[ilevel+1] = make_unique<HypreParVector>( *sampler->GetTrueP(ilevel), 0 );
        sampler->GetTrueP(ilevel)->MultTranspose(*chi[ilevel], *chi[ilevel+1]);
    }
    if (visualize) sampler->SaveMeshGLVis("mesh");
    
    // Realization
    if (!myid) std::cout << "-- Realization calculation" << std::endl;
    sampler->Sample(0,xi);
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        sampler->Eval(ilevel, xi, coef);
        if (visualize) sampler->SaveFieldGLVis(ilevel, coef, "realization");
    }

    // Error
    Vector exp_error(nLevels);
    Vector var_error(nLevels);
    const double exact_expectation(lognormal ? std::exp(variance/2.) : 0.0);
    const double exact_variance(lognormal ?
        std::exp(variance)*(std::exp(variance)-1.) : variance);

    // Dof stats
    Vector nnz(nLevels);
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector iters(nLevels);
    iters = 0.0;
    if (!myid) std::cout << "-- Computing samples statistics" << std::endl;
    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        int s_size = sampler->SampleSize(ilevel);
        Vector expectation(s_size);
        Vector chi_cov(s_size);
        Vector marginal_variance(s_size);
        expectation = 0.0;
        chi_cov = 0.0;
        marginal_variance = 0.0;
        Vector & chi_level(*chi[ilevel]);
        if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";
        {    
            Timer timer = TimeManager::AddTimer(
                std::string("Sample Generation -- Level ")
                .append(std::to_string(ilevel)));
            for(int i(0); i < nsamples; ++i)
            {
                sampler->Sample(ilevel, xi);
                for (int i = 0; i < xi.Size(); i++)
                    xi[i] = 1.0;
                sampler->Eval(ilevel, xi, coef);
                double chi_coef = dot(coef, chi_level, comm);
                chi_cov.Add(chi_coef, coef);
                expectation.Add(1., coef);
                for(int k = 0; k < s_size; ++k)
                    marginal_variance(k) += coef(k)*coef(k);
            }
        }
        iters[ilevel] /= static_cast<double>(nsamples);
        ndofs_l[ilevel] = s_size;
        ndofs_g[ilevel] = chi[ilevel]->GlobalSize();
        nnz[ilevel] = sampler->GetNNZ(ilevel);
        chi_cov *= 1./static_cast<double>(nsamples);
        expectation *= 1./static_cast<double>(nsamples);
        marginal_variance *= 1./static_cast<double>(nsamples);
        if (visualize) sampler->SaveFieldGLVis(ilevel, expectation, "expectation");
        if (visualize) sampler->SaveFieldGLVis(ilevel, chi_cov, "cov_chi");
        if (visualize) sampler->SaveFieldGLVis(ilevel, marginal_variance, "marginal_variance");
        // Error
        exp_error[ilevel] = sampler->ComputeL2Error(
                ilevel, expectation, exact_expectation);
        var_error[ilevel] = sampler->ComputeL2Error(
                ilevel, marginal_variance, exact_variance);
    }

    if (myid == 0)                                                             
    {                                                                          
        std::cout << "\nSampler Error: Expected E[u] = " << exact_expectation  
        << ", Expected V[u] = " << exact_variance << '\n'                      
        << "\n L2 Error PDE Sampler \n";                                       
    } 
    ReduceAndOutputRandomFieldErrors(exp_error, var_error);
    ReduceAndOutputStochInfo(ndofs_l, ndofs_g, nnz, iters);

    if (print_time) TimeManager::Print(std::cout);
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
