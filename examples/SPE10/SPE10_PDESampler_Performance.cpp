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
#include "PDESampler_Legacy.hpp"
#include "Utilities.hpp"
#include "MeshUtilities.hpp"

using namespace parelagmc;
using namespace parelag;
using namespace mfem;

// Computes realizations of the SPDE sampler (useful for timing)           
// without mesh embedding (the variance will be artificially inflated          
// along the boundary, especially near corners) solving the saddle point linear 
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

    // Uncertainty parameters
    const int nsamples = prob_list.Get("Number of samples", 10);
    const double variance = prob_list.Get("Variance", 1.0);
    std::string sampler_name = prob_list.Get("Sampler name", "pde");

    nLevels = par_ref_levels + 1;

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  SPE10_PDESampler_Performance.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  nDimensions: " << nDimensions << '\n';
                 if (nDimensions == 2)
                    std::cout << "*  Slice: " << par_ref_levels << '\n';
                 std::cout << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Samples: " << nsamples << '\n';
        std::cout << "*  Number of elements: ";
        for (const auto i: v_N) std::cout << i << ' ';
        std::cout << "\n*  Element sizes: "; 
        for (const auto i: v_h) std::cout << i << ' ';
        std::cout << '\n' << std::string(50,'*') << '\n';
    }

    // Create the finite element mesh
    std::unique_ptr<Mesh> mesh =  Create_SPE10_Mesh(nDimensions, v_N, v_h);
    
    for (int l = 0; l < ser_ref_levels; l++)
        mesh->UniformRefinement();

    auto pmesh = std::make_shared<ParMesh>(comm, *mesh);

    // Free the serial mesh
    mesh.reset();

    Array<int> level_nElements(nLevels);
    if (myid == 0) std::cout << "-- Parallel refinement" << std::endl;
    for (int l = 0; l < par_ref_levels; l++)
    {
        if (!unstructured) level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    
    if (!unstructured) level_nElements[0] = pmesh->GetNE();

    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);

    // Used to store material_id from info in sampler
    if(myid == 0) std::cout << "-- Coarsen topology\n";
    if (unstructured)
        BuildTopologyAlgebraic(pmesh, coarsening_factor, topology);
    else
        BuildTopologyGeometric(pmesh, level_nElements, topology);


    if (myid == 0) std::cout << "-- NormalDistributionSampler" << std::endl;
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    Vector xi, coef;
    if (myid == 0) std::cout << "-- PDESampler" << std::endl;
    PDESampler_Legacy sampler(pmesh, dist, *master_list);
    if (myid == 0) std::cout << "-- BuildHierarchy" << std::endl;
    sampler.BuildDeRhamSequence(topology);
    sampler.BuildHierarchy();
    
    // Error 
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector nnz(nLevels);
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);

    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sample Generation -- Level ")
                .append(std::to_string(ilevel)));
            for(int i(0); i < nsamples; ++i)
            {
                sampler.Sample(ilevel, xi);
                sampler.Eval(ilevel, xi, coef);
            }
        }

        nnz[ilevel] = sampler.GetNNZ(ilevel);
        ndofs_l[ilevel] = sampler.GetNumberOfDofs(ilevel);
        ndofs_g[ilevel] = sampler.GetGlobalNumberOfDofs(ilevel);
        stoch_size_l[ilevel] = sampler.SampleSize(ilevel);
        stoch_size_g[ilevel] = sampler.GlobalSampleSize(ilevel);
    }

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
