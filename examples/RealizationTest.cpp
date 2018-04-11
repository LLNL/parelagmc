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
#include <iostream>
#include <memory>
#include <cstring>
#include <elag.hpp>

#include "NormalDistributionSampler.hpp"
#include "L2ProjectionPDESampler.hpp"
#include "PDESampler.hpp"
#include "MLSampler.hpp"
#include "KLSampler.hpp"
#include "MaternCovariance.hpp"
#include "AnalyticExponentialCovariance.hpp"
#include "Utilities.hpp"
#include "MeshUtilities.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateSamplerParameterList.hpp"

// Computes a realization of a random field (Gaussian or log-normal) using a  
// KL expansion with an exponential analytic covariance and a Matern covariance
// function, and the SPDE sampler (with and without non-matching mesh embedding). 

using namespace parelagmc;
using namespace parelag;
using namespace mfem;

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

    // The files from which to read the meshes
    const std::string meshfile =
        prob_list.Get("Mesh file", "no mesh name found");
    const std::string embedfile =
        prob_list.Get("Embedded mesh file", "no mesh name found");

    // Number of mesh refinements
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);

    // AMGe hierarchy parameters
    int nLevels = prob_list.Get("Number of levels", 2);
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);

    // Uncertainty parameters
    const double variance = prob_list.Get("Variance", 1.0);
    const int nsamples = prob_list.Get("Number of samples", 10);

    if (!unstructured)
        nLevels = par_ref_levels + 1;

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
            << "*  RealizationTest.exe: Computes a realization using \n"
            << "*  analytic exponential KLE, matern KLE, SPDE sampler \n"
            << "*  (with and without non-matching mesh embedding) \n"
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

    Array<int> level_nElements(nLevels);
    Array<int> embed_level_nElements(nLevels);
    if (!myid) std::cout << "-- Parallel mesh refinement " << std::endl;    
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
        pmesh->ReorientTetMesh();
        pembedmesh->ReorientTetMesh();
    }
    
    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);
    std::vector< std::shared_ptr< AgglomeratedTopology > > embed_topology(nLevels);
    if (!myid) std::cout << "-- Agglomerating the meshes " << std::endl;
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

    
    if (!myid) std::cout << "-- Generate the uniform distribution object" << std::endl;
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    // Generate AnalyticExponentialCovariance operator
    if (!myid) std::cout << "-- Construct AnalyticExponentialCovariance" << std::endl;
    AnalyticExponentialCovariance analytic_cov(
            pmesh, *master_list);
    // Create KL Sampler with analytic covariance function
    KLSampler analytic_sampler(pmesh, dist, analytic_cov, *master_list);
    analytic_sampler.BuildDeRhamSequence(topology);
    analytic_sampler.BuildHierarchy();
    if (!myid) std::cout << "-- Construct MaternCovariance" << std::endl;
    MaternCovariance matern_cov(
            pmesh, *master_list);
    // Create KL Sampler with matern covariance function
    KLSampler matern_sampler(pmesh, dist, matern_cov, *master_list); 
    matern_sampler.BuildDeRhamSequence(topology);
    matern_sampler.BuildHierarchy();

    if (!myid) std::cout << "-- Construct PDESampler" << std::endl;
    PDESampler pde_sampler(pmesh, dist, *master_list); 
    pde_sampler.BuildDeRhamSequence(topology);
    pde_sampler.BuildHierarchy();
    
    if (!myid) std::cout << "-- Construct L2ProjectionPDESampler" << std::endl;
    L2ProjectionPDESampler projpde_sampler(pmesh, pembedmesh,
            dist, *master_list);
    projpde_sampler.BuildDeRhamSequence(topology, embed_topology);
    projpde_sampler.BuildHierarchy();

    Vector xi, embed_xi, analytic_coef, matern_coef, pde_coef, projpde_coef;
    
    pde_sampler.SaveMeshGLVis("mesh");

    // Random input on level 0 for enlarged mesh
    projpde_sampler.Sample(0, embed_xi);
    // Transfer xi to original mesh
    projpde_sampler.Transfer(0, embed_xi, xi);
    std::cout << "-- Saving realizations as matern_realization, analytic_realization, pde_realization, projpde_realization + _L(mylevel)\n";
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    { 
        projpde_sampler.Eval(ilevel, embed_xi, projpde_coef);
        pde_sampler.Eval(ilevel, xi, pde_coef);
        analytic_sampler.Eval(ilevel, xi, analytic_coef);
        matern_sampler.Eval(ilevel, xi, matern_coef);
        
        matern_sampler.SaveFieldGLVis(ilevel, matern_coef, "matern_realization");
        analytic_sampler.SaveFieldGLVis(ilevel, analytic_coef, "analytic_realization");
        projpde_sampler.SaveFieldGLVis(ilevel, pde_coef, "pde_realization");
        projpde_sampler.SaveFieldGLVis(ilevel, projpde_coef, "proj_pde_realization");
    }
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
