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
                                                              
#include "NormalDistributionSampler.hpp"
#include "PDESampler.hpp"
#include "Utilities.hpp"

#include <elag.hpp>
#include <fstream>
#include <memory>

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateSamplerParameterList.hpp"

using namespace parelagmc;
using namespace parelag;
using namespace mfem;

// Computes various statistics and realizations of the SPDE sampler
// without mesh embedding (the variance will be artificially inflated 
// along the boundary, especially near corners) solving the saddle point linear 
// system with solver specified in XML parameter list. 

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
    const char *xml_file_c = "BuildTestParameters";
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&xml_file_c, "-f", "--xml-file", "XML parameter list.");
    args.Parse();
    PARELAG_ASSERT(args.Good());
    std::string xml_file(xml_file_c);

    // Read the parameter list from file
    std::unique_ptr<ParameterList> master_list;
    if (xml_file == "BuildTestParameters")
    {
        master_list = examplehelpers::CreateSamplerTestParameters();
    }
    else
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
    
    ParameterList& prob_list = master_list->Sublist("Problem parameters",true);
   
    // The file from which to read the mesh
    const std::string meshfile =
        prob_list.Get("Mesh file", "no mesh name found");

    // Number of mesh refinements
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);

    // Output options
    const bool visualize = prob_list.Get("Visualize",false);
    const bool verbose = prob_list.Get("Verbosity",false);
    const bool print_time = prob_list.Get("Print timings",true);

    // AMGe hierarchy parameters
    int nLevels = prob_list.Get("Number of levels", 2);
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);

    // Uncertainty parameters
    const double variance = prob_list.Get("Variance", 1.0);
    const int nsamples = prob_list.Get("Number of samples", 10);
    const bool lognormal = prob_list.Get("Lognormal", false);

    if (!unstructured)
        nLevels = par_ref_levels + 1;

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  PDESamplerTest.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << '\n' 
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Number of Samples: " << nsamples << "\n*\n";
        if (unstructured)
        {
            std::cout << "*  Unstructured (algebraic) coarsening \n" 
                    << "*  Coarsening Factor: " << coarsening_factor;
        }
        else
            std::cout << "*  Structured (geometric) coarsening ";
        std::cout << '\n' << std::string(50,'*') << '\n';
     
    }

    // Read the (serial) mesh from the given mesh file and uniformly refine it.
    std::shared_ptr<mfem::ParMesh> pmesh;
    {
        std::unique_ptr<mfem::Mesh> mesh;

        std::ifstream imesh(meshfile.c_str());
        if (meshfile == "BuildHexMesh" || !imesh)
        {
            if (!myid && meshfile != "BuildHexMesh")
                std::cerr << std::endl << "-- Cannot open mesh file: "
                              << meshfile << std::endl;
            if (!myid)
                std::cout << "-- Generating cube mesh with 64 hexahedral elements.\n";
            mesh = examplehelpers::Build3DHexMesh();
        }
        else
        {
            mesh = make_unique<mfem::Mesh>(imesh, 1, 1);
            imesh.close();
        }

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        pmesh = std::make_shared<ParMesh>(comm, *mesh);
    }

    const int nDimensions = pmesh->Dimension();

    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        if (!unstructured) level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    if (!unstructured) level_nElements[0] = pmesh->GetNE();

    if(nDimensions == 3)
        pmesh->ReorientTetMesh();
    
    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);
    
    {
        Timer agg = TimeManager::AddTimer("Mesh Agglomeration -- Total");
        if (unstructured)
            BuildTopologyAlgebraic(pmesh, coarsening_factor, topology);
        else
            BuildTopologyGeometric(pmesh, level_nElements, topology);
    }
    
    for(int ilevel = 0; ilevel < nLevels; ++ilevel)
        ShowTopologyAgglomeratedElements(topology[ilevel].get(),pmesh.get());

    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);
    Vector xi, coef;

    PDESampler sampler(pmesh, dist, *master_list);
    {
        Timer timer = TimeManager::AddTimer("PDESampler Build DeRhamSequence -- Total");
        sampler.BuildDeRhamSequence(topology);
        sampler.BuildHierarchy();
    }
    
    std::vector< std::unique_ptr<HypreParVector >> chi(nLevels);
    chi[0].reset(chi_center_of_mass(pmesh.get()));
    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        chi[ilevel+1] = make_unique<HypreParVector>( *sampler.GetTrueP(ilevel), 0 );
        sampler.GetTrueP(ilevel)->MultTranspose(*chi[ilevel], *chi[ilevel+1]);
    }
    
    if (visualize) sampler.SaveMeshGLVis("mesh");

    // Realization computation
    sampler.Sample(0,xi);
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        sampler.Eval(ilevel, xi, coef);
        if (visualize) sampler.SaveFieldGLVis(ilevel, coef, "pde_realization");
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
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);

    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        int s_size = sampler.SampleSize(ilevel);
        Vector expectation(s_size);
        Vector chi_cov(s_size);
        Vector marginal_variance(s_size);
        expectation = 0.0;
        chi_cov = 0.0;
        marginal_variance = 0.0;
        Vector & chi_level(*chi[ilevel]);
        {
            if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";
            Timer mc_timer = TimeManager::AddTimer(
                std::string("Sample Generation -- Level ")
                .append(std::to_string(ilevel)));
            for(int i(0); i < nsamples; ++i)
            {
                sampler.Sample(ilevel, xi);
                sampler.Eval(ilevel, xi, coef);
                double chi_coef = dot(coef, chi_level, comm);
                chi_cov.Add(chi_coef, coef);
                expectation.Add(1., coef);
                for(int k = 0; k < s_size; ++k)
                    marginal_variance(k) += coef(k)*coef(k);
            }
        }
        nnz[ilevel] = sampler.GetNNZ(ilevel);
        ndofs_l[ilevel] = sampler.SampleSize(ilevel);
        ndofs_g[ilevel] = sampler.GlobalSampleSize(ilevel);
        stoch_size_l[ilevel] = sampler.GetNumberOfDofs(ilevel); 
        stoch_size_g[ilevel] = sampler.GetGlobalNumberOfDofs(ilevel);

        chi_cov *= 1./static_cast<double>(nsamples);
        expectation *= 1./static_cast<double>(nsamples);
        marginal_variance *= 1./static_cast<double>(nsamples);

        if (visualize)
        {
            sampler.SaveFieldGLVis(ilevel, expectation, "pde_expectation");
            sampler.SaveFieldGLVis(ilevel, chi_cov, "pde_cov_chi");
            sampler.SaveFieldGLVis(ilevel, marginal_variance, 
                "pde_marginal_variance");
        }
        
        // Error calculation
        exp_error[ilevel] = sampler.ComputeL2Error(
                ilevel, expectation, exact_expectation);
        var_error[ilevel] = sampler.ComputeL2Error(
                ilevel, marginal_variance, exact_variance);
    }

    if (myid == 0)
    {
        std::cout << "\nSampler Error: Expected E[u] = " << exact_expectation
        << ",  Expected V[u] = " << exact_variance << '\n'
        << "\n L2 Error PDE Sampler \n";
    }
    ReduceAndOutputRandomFieldErrors(exp_error, var_error);
    ReduceAndOutputBothInfo(ndofs_l, ndofs_g,
        stoch_size_l, stoch_size_g, nnz);
    
    if (print_time) TimeManager::Print(std::cout); 
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
                  
  return EXIT_SUCCESS;
}
