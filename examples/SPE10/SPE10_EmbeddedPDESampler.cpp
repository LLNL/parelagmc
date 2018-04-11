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

using namespace parelagmc;
using namespace parelag;
using namespace mfem;

// Computes realizations and statistics of the SPDE sampler            
// with matching mesh embedding solving the saddle point linear                
// system with solver specified in XML parameter list on SPE10 domain.

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
    std::vector<int> v_N = spe_list.Get("Number of elements", std::vector<int>{60,220,85});
    std::vector<double> v_h = spe_list.Get("Element sizes", std::vector<double>{20,10,2});
    std::vector<int> v_n = spe_list.Get("Number of embedded elements", std::vector<int>{2,2,2});

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
                 << "*  SPE10_EmbeddedPDESampler.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  nDimensions: " << nDimensions << '\n';
                 if (nDimensions == 2)
                    std::cout << "*  Slice: " << par_ref_levels << '\n';
                 std::cout << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Samples: " << nsamples << '\n'
                 << "*  Number of elements: ";
        for (const auto i: v_N) std::cout << i << ' ';
        std::cout << "\n*  Element sizes: "; 
        for (const auto i: v_h) std::cout << i << ' ';
        std::cout << "\n*  Number of added elements: ";             
        for (const auto i: v_n) std::cout << i << ' ';
        std::cout << '\n';

        std::cout << std::string(50,'*') << '\n';
    }

    // Create the finite element mesh
    std::unique_ptr<Mesh> mesh = Create_SPE10_Mesh(nDimensions, v_N, v_h);
    std::unique_ptr<Mesh> embedmesh = 
        Create_Embedded_SPE10_Mesh(nDimensions, v_N, v_h, v_n);;
    
    for (int l = 0; l < ser_ref_levels; l++)
    {
        mesh->UniformRefinement();
        embedmesh->UniformRefinement();
    }
   
    std::shared_ptr<ParMesh> pmesh;
    std::shared_ptr<ParMesh> pembedmesh;

    if (myid == 0) std::cout << "-- Create parallel domain splits using METIS " << '\n';
    // Build parallel meshes with same domain partitioning for embedded mesh
    Create_Parallel_Domain_Partitioning(comm, mesh, embedmesh, pmesh, pembedmesh);

    // Free the serial mesh
    mesh.reset();
    embedmesh.reset();

    Array<int> level_nElements(nLevels);
    Array<int> embed_level_nElements(nLevels);
    if (myid == 0) std::cout << "-- Parallel refinement" << std::endl;
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

    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);
    std::vector< std::shared_ptr< AgglomeratedTopology > > embed_topology(nLevels);

    // Used to store material_id from info in sampler
    std::vector<Array<int>> material_id(nLevels);

    if (unstructured)
        EmbeddedBuildTopologyAlgebraic(pmesh, pembedmesh, coarsening_factor,
            topology, embed_topology, material_id);
    else
        EmbeddedBuildTopologyGeometric(pmesh, pembedmesh,
            level_nElements, embed_level_nElements,
            topology, embed_topology, material_id);

    if (visualize)
    {
        for (int ilevel = 0; ilevel < nLevels; ++ilevel)
        {
            // Writes agglomerates to a vector  
            std::ostringstream name;
            name << "agglomerates_L" << ilevel; 
            std::ofstream name_ofs(name.str().c_str());
            ShowTopologyAgglomeratedElements(topology[ilevel].get(), pmesh.get(), &name_ofs);
        }
    }

    if (myid == 0) std::cout << "-- NormalDistributionSampler" << std::endl;
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    Vector xi, embedcoef, coef, origcoef;
    if (myid == 0) std::cout << "-- EmbeddedPDESampler" << std::endl;
    EmbeddedPDESampler embedsampler(pmesh, pembedmesh,
            dist, material_id, *master_list);
    embedsampler.BuildDeRhamSequence(embed_topology);
    embedsampler.BuildHierarchy();

    std::vector< std::unique_ptr<HypreParVector > > embedchi(nLevels);
    embedchi[0].reset(chi_center_of_mass(pembedmesh.get()));

    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        embedchi[ilevel+1] = make_unique<HypreParVector>( *embedsampler.GetTrueP(ilevel), 0 );
        embedsampler.GetTrueP(ilevel)->MultTranspose(*embedchi[ilevel], *embedchi[ilevel+1]);
    }
    
    if (myid == 0) std::cout << "-- Save Meshes " << std::endl; 
    if (visualize) embedsampler.EmbedSaveMeshGLVis("embed_mesh");
    if (visualize) embedsampler.SaveMeshGLVis("mesh");

    // Realization
    embedsampler.Sample(0,xi);
    if (myid == 0) std::cout << "-- Realization computation" << std::endl;
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        embedsampler.EmbedEval(ilevel, xi, embedcoef);
        embedsampler.EmbedSaveFieldGLVis(ilevel, embedcoef, "embed_realization");
        embedsampler.SaveFieldGLVis(ilevel, embedcoef, "proj_realization");
    }

    // Error 
    Vector exp_error(nLevels);
    Vector var_error(nLevels);
    const double exact_expectation(lognormal ? std::exp(variance/2.) : 0.0);
    const double exact_variance(lognormal ?
        std::exp(variance)*(std::exp(variance)-1.) : variance);

    Vector iters(nLevels);
    iters = 0.0;
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector nnz(nLevels);
    if (myid == 0) std::cout << "-- Computing sample statistics" << std::endl;
    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        const int embed_s_size = embedsampler.EmbedSampleSize(ilevel);
        Vector embedexpectation(embed_s_size);
        Vector embedchi_cov(embed_s_size);
        Vector embedmarginal_variance(embed_s_size);
        embedexpectation = 0.0;
        embedchi_cov = 0.0;
        embedmarginal_variance = 0.0;
        Vector & embedchi_level(*embedchi[ilevel]);

        if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";
        {
            for(int i(0); i < nsamples; ++i)
            {
                embedsampler.Sample(ilevel, xi);
                embedsampler.EmbedEval(ilevel, xi, embedcoef);
                double embedchi_coef = dot(embedcoef, embedchi_level, comm);
                embedchi_cov.Add(embedchi_coef, embedcoef);
                embedexpectation.Add(1., embedcoef);
                for(int k = 0; k < embed_s_size; ++k)
                    embedmarginal_variance(k) += embedcoef(k)*embedcoef(k);
            }
        }
        ndofs_l[ilevel] = embed_s_size;
        ndofs_g[ilevel] = embedchi[ilevel]->GlobalSize();
        nnz[ilevel] = embedsampler.GetNNZ(ilevel);
        embedchi_cov *= 1./static_cast<double>(nsamples);
        embedexpectation *= 1./static_cast<double>(nsamples);
        embedmarginal_variance *= 1./static_cast<double>(nsamples);

        // this is actually GridFunction::ComputeL2Error ** 2! fyi
        exp_error[ilevel] = embedsampler.ComputeL2Error(
                ilevel, embedexpectation, exact_expectation);
        var_error[ilevel] = embedsampler.ComputeL2Error(
                ilevel, embedmarginal_variance, exact_variance);

        if (visualize)
        {
            embedsampler.EmbedSaveFieldGLVis(ilevel, embedexpectation,
                    "embed_expectation");
            embedsampler.EmbedSaveFieldGLVis(ilevel, embedchi_cov,
                    "embed_chi_cov");
            embedsampler.EmbedSaveFieldGLVis(ilevel, embedmarginal_variance,
                    "embed_marginal_variance");
            embedsampler.SaveFieldGLVis(ilevel, embedexpectation,
                    "expectation");
            embedsampler.SaveFieldGLVis(ilevel, embedchi_cov,
                    "chi_cov");
            embedsampler.SaveFieldGLVis(ilevel, embedmarginal_variance,
                    "marginal_variance");
        }
    }
   
    if (myid == 0)                                                             
    {                                                                          
        std::cout << "\nSampler Error: Expected E[u] = " << exact_expectation  
        << ", Expected V[u] = " << exact_variance << '\n'                      
        << "\n L2 Error Mesh Embedded PDE Sampler \n";                                       
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
