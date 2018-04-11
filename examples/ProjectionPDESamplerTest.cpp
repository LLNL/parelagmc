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
#include "L2ProjectionPDESampler.hpp"
#include "Utilities.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateSamplerParameterList.hpp"

using namespace parelagmc;
using namespace parelag;
using namespace mfem;
using std::cout;

// Computes various statistics and realizations of the SPDE sampler            
// with non-matching mesh embedding solving the saddle point linear 
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
        prob_list.Get("Embedded mesh file", meshfile);

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
                 << "*  L2ProjectionPDESampler.exe \n" 
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << "\n" 
                 << "*  Embedded Mesh: " << embedfile << "\n"
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Number of Samples: " << nsamples << '\n'
                 << "*\n";
        if (unstructured)
        {
            std::cout << "*  Unstructured (algebraic) coarsening \n"
                    << "*  Coarsening Factor: " << coarsening_factor << '\n';
        }
        else
            std::cout << "*  Structured (geometric) coarsening \n";
        std::cout << std::string(50,'*') << '\n';
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

        pmesh = std::make_shared<ParMesh>(comm, *mesh);

        std::ifstream emesh(embedfile.c_str());
        if (meshfile == "BuildEmbedHexMesh" || !emesh)
        {
            if (!myid && embedfile != "BuildEmbedHexMesh")
                std::cerr << std::endl << "-- Cannot open embedded mesh file: "
                              << embedfile << std::endl;
            if (!myid)
                std::cout << "-- Generating enlarged cube mesh with 216 hexahedral elements.\n";
            embedmesh = examplehelpers::Build3DHexEnlargedMesh();
        }
        else
        {
            embedmesh = make_unique<mfem::Mesh>(emesh, 1, 1);
            emesh.close();
        }

        for (int l = 0; l < ser_ref_levels; l++)
            embedmesh->UniformRefinement();

        pembedmesh = std::make_shared<ParMesh>(comm, *embedmesh);
    }

    const int nDimensions = pmesh->Dimension();    

    if (!myid && verbose) std::cout << "-- Parallel mesh refinement " << std::endl;
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
    if (!myid && verbose) std::cout << "-- Agglomerating the meshes " << std::endl;
    {
        Timer agg = TimeManager::AddTimer("Mesh Agglomeration -- Total");
        if (unstructured)
        {
            BuildTopologyAlgebraic(pmesh, coarsening_factor, topology);
            BuildTopologyAlgebraic(pembedmesh, coarsening_factor, embed_topology);
        }
        else
        {
            BuildTopologyGeometric(pmesh, level_nElements, topology);
            BuildTopologyGeometric(pembedmesh, embed_level_nElements, embed_topology);
        }
    }
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    if (!myid && verbose) std::cout << "-- Construct L2ProjectionPDESampler" << std::endl;
    L2ProjectionPDESampler projsampler(pmesh, pembedmesh, 
            dist, *master_list);
    if (!myid && verbose) std::cout << "-- Sampler BuildHierarchy" << std::endl;
    {
        Timer timer = TimeManager::AddTimer("BuildHierarchy -- Total");
        projsampler.BuildDeRhamSequence(topology, embed_topology);
        projsampler.BuildHierarchy();
    }

    std::vector< std::unique_ptr<HypreParVector> > projchi(nLevels);
    projchi[0].reset(chi_center_of_mass(pmesh.get()));

    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        projchi[ilevel+1] = make_unique<HypreParVector>( *projsampler.GetTrueP(ilevel), 0 );
        projsampler.GetTrueP(ilevel)->MultTranspose(*projchi[ilevel], *projchi[ilevel+1]);
    }

    if (visualize)
    { 
        projsampler.EmbedSaveMeshGLVis("embed_mesh");
        projsampler.SaveMeshGLVis("mesh");
    }
    Vector xi, proj_coef; 
   
    if (!myid && verbose) std::cout << "-- Realization calculation" << std::endl;
    projsampler.Sample(0, xi);
    // Realization
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        projsampler.Eval(ilevel, xi, proj_coef);
        if (visualize) projsampler.SaveFieldGLVis(ilevel, proj_coef, "proj_realization");
    }

    // Error 
    Vector proj_exp_error(nLevels);
    Vector proj_var_error(nLevels);
    const double exact_expectation(lognormal ? std::exp(variance/2.) : 0.0);
    const double exact_variance(lognormal ?
        std::exp(variance)*(std::exp(variance)-1.) : variance);

    // Dof stats
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);
    Vector nnz(nLevels); 
    if (!myid && verbose) std::cout << "-- Computing sample statistics" << std::endl;
    
    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        int s_size = projsampler.SampleSize(ilevel);
        Vector projexpectation(s_size);
        Vector projchi_cov(s_size);
        Vector projmarginal_variance(s_size);
        Vector projtruemarginal_variance(s_size);
        projexpectation = 0.0;
        projchi_cov = 0.0;
        projmarginal_variance = 0.0;
        projtruemarginal_variance = 0.0;
        Vector & projchi_level(*projchi[ilevel]);
        {
            Timer mc_timer = TimeManager::AddTimer(
                std::string("Sample Generation -- Level ")
                .append(std::to_string(ilevel)));
        }    
        if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";
        for(int i(0); i < nsamples; ++i)
        {
            {
                Timer mc_timer = TimeManager::GetTimer(
                    std::string("Sample Generation -- Level ")
                    .append(std::to_string(ilevel)));
                projsampler.Sample(ilevel, xi);
                projsampler.Eval(ilevel, xi, proj_coef);
            }
            double projchi_coef = dot(proj_coef, projchi_level, comm);
            projchi_cov.Add(projchi_coef, proj_coef);
            projexpectation.Add(1., proj_coef);
            for(int k = 0; k < s_size; ++k)
            {
                projmarginal_variance(k) += proj_coef(k)*proj_coef(k);
                projtruemarginal_variance(k) += proj_coef(k)*proj_coef(k);
            }
        }
        
        nnz[ilevel] = projsampler.GetNNZ(ilevel);
        ndofs_l[ilevel] = projsampler.GetNumberOfDofs(ilevel); 
        ndofs_g[ilevel] = projsampler.GetGlobalNumberOfDofs(ilevel);    
        stoch_size_l[ilevel] = projsampler.SampleSize(ilevel); 
        stoch_size_g[ilevel] = projsampler.GlobalSampleSize(ilevel);    

        projchi_cov *= 1./static_cast<double>(nsamples);
        projtruemarginal_variance *= 1./static_cast<double>(nsamples-1);
        for (int k = 0; k < s_size; ++k)
            projtruemarginal_variance(k) -= projexpectation(k)*projexpectation(k)
                /(static_cast<double>(nsamples)*(static_cast<double>(nsamples)-1.)); 
        projexpectation *= 1./static_cast<double>(nsamples);
        projmarginal_variance *= 1./static_cast<double>(nsamples);
        
        // this is actually GridFunction::ComputeL2Error ** 2! fyi
        proj_exp_error[ilevel] = projsampler.ComputeL2Error(
                ilevel, projexpectation, exact_expectation);
        proj_var_error[ilevel] = projsampler.ComputeL2Error(
                ilevel, projmarginal_variance, exact_variance);

        if (visualize)
        {
            projsampler.SaveFieldGLVis(ilevel, projexpectation, 
                    "proj_expectation");
            projsampler.SaveFieldGLVis(ilevel, projchi_cov, 
                    "proj_cov_chi");
            projsampler.SaveFieldGLVis(ilevel, projmarginal_variance, 
                    "proj_marginal_variance");
            projsampler.SaveFieldGLVis(ilevel, projtruemarginal_variance, 
                    "proj_true_marginal_variance");
        }
    }
    if (myid == 0)                                                             
    {                                                                          
        std::cout << "\nSampler Error: Expected E[u] = " << exact_expectation  
        << ", Expected V[u] = " << exact_variance << '\n'                      
        << "\n L2 Error Projection PDE Sampler \n";                                       
    }         
    ReduceAndOutputRandomFieldErrors(proj_exp_error, proj_var_error);
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
