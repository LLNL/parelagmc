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

#include "DarcySolver_Legacy.hpp"
#include "Utilities.hpp"
#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateDarcyParameterList.hpp"

using namespace parelagmc;
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

// Deterministic example of DarcySolver_Legacy to solve a mixed Darcy 
// problem where the boundary conditions are specified in the XML parameter 
// list using L2-H1 preconditioned MINRES to solve the saddle point system.

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
        master_list = examplehelpers::CreateDarcyTestParameters();
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
    const bool verbose = prob_list.Get("Verbosity",false);
    const bool print_time = prob_list.Get("Print timings",true);
    const bool visualize = prob_list.Get("Visualize",false);
    
    // AMGe hierarchy parameters
    int nLevels = prob_list.Get("Number of levels", 2);
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);
    
    // Boundary conditions and QoI specifications
    const int n_bdr_attributes = prob_list.Get("Number boundary attributes", 6);
    std::vector<int> v_ess_attr = prob_list.Get(
        "Essential attributes", std::vector<int>());
    std::vector<int> v_obs_attr = prob_list.Get(
        "Observational attributes", std::vector<int>());
    std::vector<int> v_inflow_attr = prob_list.Get(
        "Inflow attributes", std::vector<int>());
    // Create mfem::Array from std::vector 
    Array<int> ess_attr(v_ess_attr.data(), n_bdr_attributes);
    Array<int> obs_attr(v_obs_attr.data(), n_bdr_attributes);
    Array<int> inflow_attr(v_inflow_attr.data(), n_bdr_attributes);

    if (!unstructured)
        nLevels = par_ref_levels + 1;

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  DarcySolver_Legacy example with permeability coeff == 1 \n" 
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << '\n'
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Quantity of interest: eff_perm \n";
        if (unstructured)
        {
            std::cout << "*  Unstructured (algebraic) coarsening \n"
                    << "*  Coarsening Factor: " << coarsening_factor << '\n';
        }
        else
            std::cout << "*  Structured (geometric) coarsening \n";
        ess_attr.Print(std::cout << "*  EssentialAttributes: ", n_bdr_attributes);
        obs_attr.Print(std::cout << "*  ObservationAttributes: ", n_bdr_attributes);
        inflow_attr.Print(std::cout << "*  InflowAttributes: ", n_bdr_attributes);
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

    const int nDimensions = pmesh->Dimension();

    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        if (!unstructured)
            level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    if (!unstructured) level_nElements[0] = pmesh->GetNE();
    if(nDimensions == 3)
        pmesh->ReorientTetMesh();

    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);

    if (unstructured)
        BuildTopologyAlgebraic(pmesh, coarsening_factor, topology);
    else
        BuildTopologyGeometric(pmesh, level_nElements, topology);

    ConstantCoefficient zero(0.);
    ConstantCoefficient one(1.);
    ConstantCoefficient minus_one(-1.);
    RestrictedCoefficient obs_coeff(one, obs_attr);
    RestrictedCoefficient pinflow_coeff(minus_one, inflow_attr);
    Vector zeros_nDim(nDimensions);
    zeros_nDim = 0.;
    VectorConstantCoefficient zero_vcoeff(zeros_nDim);
    auto solver = make_unique<DarcySolver_Legacy>(pmesh, *master_list); 
    solver->BuildHierachySpaces(topology, 
        make_unique<VectorFEMassIntegrator>(one));
    // Set QoI to be effective permeability along observational boundary
    solver->BuildBdrObservationFunctional(
        new VectorFEBoundaryFluxLFIntegrator(obs_coeff) );
    solver->SetEssBdrConditions(ess_attr, zero_vcoeff);
    solver->BuildForcingTerms(zero_vcoeff, pinflow_coeff, zero);

    // Get iters, dofs for each level
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector iters(nLevels);
    Vector nnz(nLevels);
    iters = 0.;
    double Q,C;
    Vector ones(solver->GetSizeOfStochasticData(0));
    ones = 1.;
    if (visualize) solver->SaveMeshGLVis("darcy_mesh");
    
    std::stringstream msg;
    if (!myid) msg << "\nL  "
        << std::setw(8) << std::left << "QoI  "
        << std::setw(8) << std::left << "  Cost (dofs)" << std::endl;
    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        ones.SetSize(solver->GetSizeOfStochasticData(ilevel));
        if (!myid && verbose) std::cout << "\nLevel " << ilevel << ":" << std::endl;
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Darcy: SolveFwd -- Level ")
                .append(std::to_string(ilevel)));
            solver->SolveFwd(ilevel, ones, Q, C);
        }
        if (!myid) msg << ilevel << "  " 
            << std::setw(8) << std::left << Q << "  "
            << std::setw(8) << std::left << C << std::endl;
        ndofs_l[ilevel] = solver->GetNumberOfDofs(ilevel);
        ndofs_g[ilevel] = solver->GetGlobalNumberOfDofs(ilevel);
        iters[ilevel] = solver->GetNumIters();
        nnz[ilevel] = solver->GetNNZ(ilevel);
    }

    if (!myid) std::cout << msg.str() << std::flush; 
    ReduceAndOutputDofsInfo(ndofs_l, ndofs_g, nnz, iters);

    if (print_time) TimeManager::Print(std::cout); 
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
