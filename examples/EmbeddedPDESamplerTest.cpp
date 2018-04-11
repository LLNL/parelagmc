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
#include "EmbeddedPDESampler.hpp"
#include "Utilities.hpp"
#include "MeshUtilities.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateSamplerParameterList.hpp"

using namespace parelagmc;
using namespace parelag;
using namespace mfem;
using std::cout;

// Computes various statistics and realizations of the SPDE sampler
// with matching mesh embedding solving the saddle point linear 
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

    // Read the parameter list from file or generate the list
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

    // The files from which to read the meshes
    const std::string meshfile =
        prob_list.Get("Mesh file", "no mesh name found");
    const std::string embedfile =
        prob_list.Get("Embedded mesh file", "no mesh name found");

    // Number of mesh refinements
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);

    // Output options
    const bool print_time = prob_list.Get("Print timings",true);
    const bool visualize = prob_list.Get("Visualize", false);
    const bool verbose = prob_list.Get("Verbosity",false);

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
                 << "*  EmbeddedPDESamplerTest.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << '\n'
                 << "*  Embedded Mesh: " << embedfile << '\n'
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Number of Samples: " << nsamples << '\n'
                 << "*\n";
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
    std::shared_ptr<mfem::ParMesh> pembedmesh;
    {
        std::unique_ptr<mfem::Mesh> mesh;
        std::unique_ptr<mfem::Mesh> embedmesh;

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

        std::ifstream emesh(embedfile.c_str());
        if (meshfile == "BuildEmbedHexMesh" || !emesh)
        {
            if (!myid && embedfile != "BuildEmbedHexMesh")
                std::cerr << std::endl << "-- Cannot open embedded mesh file: "
                              << embedfile << std::endl;
            if (!myid)
                std::cout << "-- Generating embedded cube mesh with 216 hexahedral elements.\n";
            embedmesh = examplehelpers::Build3DHexEmbeddedMesh();
        }
        else
        {
            embedmesh = make_unique<mfem::Mesh>(emesh, 1, 1);
            emesh.close();
        }

        for (int l = 0; l < ser_ref_levels; l++)
            embedmesh->UniformRefinement();

        // Build parallel meshes with same domain partitioning for embedded mesh
        // Uses metis for domain partitioning
        Create_Parallel_Domain_Partitioning(comm, mesh, embedmesh, pmesh, pembedmesh);
    }

    const int nDimensions = pmesh->Dimension();

    if (myid == 0) std::cout << "-- Parallel mesh refinement " << std::endl;
    Array<int> level_nElements(nLevels);
    Array<int> embed_level_nElements(nLevels);

    for (int l = 0; l < par_ref_levels; l++)
    {    
        if (!unstructured)
            level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    for (int l = 0; l < par_ref_levels; l++)
    {    
        if (!unstructured)
            embed_level_nElements[par_ref_levels-l] = pembedmesh->GetNE();
        pembedmesh->UniformRefinement();
    }
    
    if (!unstructured) level_nElements[0] = pmesh->GetNE();
    if (!unstructured) embed_level_nElements[0] = pembedmesh->GetNE();

    if(nDimensions == 3)
    {
        pembedmesh->ReorientTetMesh();
        pmesh->ReorientTetMesh();
    }

    // topology for embedmesh and mesh
    std::vector< std::shared_ptr< AgglomeratedTopology > > embed_topology(nLevels);
    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);

    // Used to store material_id from info in sampler
    std::vector<Array<int>> material_id(nLevels);
    if (!myid && verbose) std::cout << "-- Agglomerating the meshes\n";
    {
        Timer agg = TimeManager::AddTimer("Mesh Agglomeration -- Total");
        if (unstructured)
            EmbeddedBuildTopologyAlgebraic(pmesh, pembedmesh, coarsening_factor,
                topology, embed_topology, material_id);
        else
            EmbeddedBuildTopologyGeometric(pmesh, pembedmesh, 
                level_nElements, embed_level_nElements,
                topology, embed_topology, material_id);
    }
    
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    Vector xi, coef;
    if (!myid && verbose) std::cout << "-- EmbeddedPDESampler" << std::endl;
    EmbeddedPDESampler embedsampler(pmesh, pembedmesh, 
            dist, material_id, *master_list);

    if (!myid && verbose) std::cout << "-- BuildHierarchy" << std::endl;
    {
        Timer timer = TimeManager::AddTimer("BuildHierarchy -- Total");
        embedsampler.BuildDeRhamSequence(embed_topology);
        embedsampler.BuildHierarchy();
    }

    std::vector< std::unique_ptr<HypreParVector > > embed_chi(nLevels);
    embed_chi[0].reset(chi_center_of_mass(pembedmesh.get()));

    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        embed_chi[ilevel+1] = make_unique<HypreParVector>(
            *embedsampler.GetTrueP(ilevel), 0 );
        embedsampler.GetTrueP(ilevel)->MultTranspose(
            *embed_chi[ilevel], *embed_chi[ilevel+1]);
    }

    if (visualize) embedsampler.SaveMeshGLVis("mesh");
    
    // Realization computation
    if (!myid && verbose) std::cout << "-- Realization computation \n";
    embedsampler.Sample(0,xi);
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        embedsampler.EmbedEval(ilevel, xi, coef);
        if (visualize) embedsampler.SaveFieldGLVis(ilevel, coef, 
            "embed_realization");
    }

    // Error 
    Vector exp_error(nLevels);
    Vector var_error(nLevels);
    const double exact_expectation(lognormal ? std::exp(variance/2.) : 0.0);
    const double exact_variance(lognormal ? 
        std::exp(variance)*(std::exp(variance)-1.) : variance);
    // Dof stats 
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);

    if (!myid && verbose) std::cout << "-- RF Statistics computation\n";
    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        int s_size = embedsampler.EmbedSampleSize(ilevel);

        Vector embed_expectation(s_size);
        Vector embed_marginal_variance(s_size);
        Vector embed_chi_cov(s_size);
        embed_expectation = 0.0;
        embed_marginal_variance = 0.0;
        embed_chi_cov = 0.0;
        Vector & embed_chi_level(*embed_chi[ilevel]);
        {
            Timer mc_timer = TimeManager::AddTimer(
                std::string("Sample Generation -- Level ")
                .append(std::to_string(ilevel)));
            if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";
            for(int i(0); i < nsamples; ++i)
            {
                embedsampler.Sample(ilevel, xi);
                embedsampler.EmbedEval(ilevel, xi, coef);
                embed_expectation.Add(1., coef);
                for(int k = 0; k < s_size; ++k)
                    embed_marginal_variance(k) += coef(k)*coef(k);
                double embed_chi_coef = dot(coef, embed_chi_level, comm);
                embed_chi_cov.Add(embed_chi_coef, coef);
            }
        }
        stoch_size_l[ilevel] = embedsampler.SampleSize(ilevel); 
        stoch_size_g[ilevel] = embedsampler.GlobalSampleSize(ilevel); 
        ndofs_l[ilevel] = embedsampler.GetNumberOfDofs(ilevel);
        ndofs_g[ilevel] = embedsampler.GetGlobalNumberOfDofs(ilevel);   
        embed_expectation *= 1./static_cast<double>(nsamples);
        embed_marginal_variance *= 1./static_cast<double>(nsamples);
        embed_chi_cov *= 1./static_cast<double>(nsamples);

        if (visualize)
        {
            embedsampler.SaveFieldGLVis(ilevel, embed_expectation, 
                "embed_expectation");
            embedsampler.SaveFieldGLVis(ilevel, embed_marginal_variance, 
                "embed_marginal_variance");
            embedsampler.SaveFieldGLVis(ilevel, embed_chi_cov,
                    "embed_cov_chi");
        }
        
        exp_error[ilevel] = embedsampler.ComputeL2Error(
                ilevel, embed_expectation, exact_expectation);
        var_error[ilevel] = embedsampler.ComputeL2Error(
                ilevel, embed_marginal_variance, exact_variance);
    }

    if (myid == 0)
    {
        std::cout << "\nSampler Error: Expected E[u] = " << exact_expectation
        << ", Expected V[u] = " << exact_variance << '\n'
        << "\n L2 Error Embedded PDE Sampler \n";
    }
    ReduceAndOutputRandomFieldErrors(exp_error, var_error);

    ReduceAndOutputBothInfo(ndofs_l, ndofs_g, 
        stoch_size_l, stoch_size_g);

    if (print_time) TimeManager::Print(std::cout);
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
