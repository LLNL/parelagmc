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

#include "NormalDistributionSampler.hpp"
#include "AnalyticExponentialCovariance.hpp"
#include "Utilities.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateSamplerParameterList.hpp"

// Example of computing eigenvalues and eigenvectors of 
// AnalyticExponentialCovariance function for a particular mesh. 

using namespace parelagmc;
using namespace parelag;
using namespace mfem;
using std::cout;

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

    // The file from which to read the mesh
    const std::string meshfile =
        prob_list.Get("Mesh file", "no mesh name found");

    // The number of times to refine in parallel
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);

    // The number of times to refine in serial
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);
    const bool print_time = prob_list.Get("Print timings",true);
    const double variance = prob_list.Get("Variance", 1.0);
    const int nsamples = prob_list.Get("Number of samples", 10);
    std::vector<int> v_nmodes = prob_list.Get("Number of modes", 
        std::vector<int>{10, 10});
    int nmodes = v_nmodes[0];

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  Computes and prints AnalyticCovariance ews \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << '\n'
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Modes: " << nmodes << '\n'
                 << "*  Number of Samples: " << nsamples << '\n';
        std::cout << std::string(50,'*') << '\n';
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

    AnalyticExponentialCovariance cov_func(pmesh, *master_list);
    {
        Timer timer = TimeManager::AddTimer("SolveEigenvalue() Time");
        cov_func.SolveEigenvalue();
    }
    
    cov_func.ShowMe();
    DenseMatrix evects;
    Vector eigs(nmodes);
    eigs = 0.;
    eigs = cov_func.Eigenvalues();
    
    if (myid == 0)
    {
        std::cout << "Analytic: Dominant Eigenvectors: " << std::endl;
        for (int n = 0; n < nmodes; n++)
            std::cout << eigs[n] << std::endl;
    }

    if (print_time) TimeManager::Print(std::cout); 

  }
  catch (std::exception &e)
  {  
    if (myid == 0) std::cout << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}
