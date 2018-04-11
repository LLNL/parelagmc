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
#include "DarcySolver.hpp"
#include "MLSampler.hpp"
#include "Utilities.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateMLMCParameterList.hpp"

using namespace parelagmc;
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

// Solve a mixed Darcy problem with linear solver specified in XML parameter 
// list with a random permeability coefficient generated with the SPDE sampler
// with non-matching mesh embedding (using L2ProjectionPDESampler with linear 
// solver for the sampler saddle point system specified in XML parameter list).

// If visualize=true, a random permeability realization is saved, along 
// with the Darcy velocity and pressure for GLVis visualization.

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
        master_list = examplehelpers::CreateMLMCTestParameters();
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
    
    // Boundary conditions and QoI specifications
    const std::string qoi = prob_list.Get("Quantity of interest", "eff_perm");
    std::vector<double> v_qoi_p_data_point = prob_list.Get(                    
        "Local average pressure QoI spatial point",                            
        std::vector<double>{0.5, 0.5, 0.5});                                   
    const double eps_p_qoi = prob_list.Get(                                    
        "Epsilon for local average pressure QoI", 0.1); 
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
                 << "*  DarcySolver example with random input from L2ProjectionPDESampler \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << "\n"
                 << "*  Embedded Mesh: " << embedfile << "\n"
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Quantity of interest: " << qoi << '\n';
        if (qoi.compare("local_avg_p") == 0)
        {
            std::cout << "*  Pressure QoI spatial point: (";
            for (auto i:v_qoi_p_data_point)
                std::cout << i << " ";
            std::cout << ") with epsilon = " << eps_p_qoi << '\n';
        }
        std::cout << "*  Variance: " << variance << '\n'
                 << "*  Number of Samples: " << nsamples << '\n';
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

    if (!myid) std::cout << "-- Parallel mesh refinement " << std::endl;
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
    if (!myid) std::cout << "-- Agglomerating the meshes " << std::endl;

    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);
    std::vector< std::shared_ptr< AgglomeratedTopology > > embed_topology(nLevels);
    // Build topology for both meshes either using derefining (Geometric) 
    // or Metis (Algrebraic)
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

    ConstantCoefficient zero(0.);
    ConstantCoefficient one(1.);
    ConstantCoefficient minus_one(-1.);
    RestrictedCoefficient obs_coeff(one, obs_attr);
    RestrictedCoefficient pinflow_coeff(minus_one, inflow_attr);
    Vector zeros_nDim(nDimensions);
    zeros_nDim = 0.;
    VectorConstantCoefficient zero_vcoeff(zeros_nDim);
     
    std::unique_ptr<DarcySolver> solver =
        make_unique<DarcySolver>(pmesh, *master_list);

    if (!myid) std::cout << "-- Darcy BuildHierarchy" << std::endl;
    {
        Timer timer = TimeManager::AddTimer("Darcy: BuildHierarchy -- Total");
        solver->BuildHierachySpaces(topology,make_unique<VectorFEMassIntegrator>(one));
    }

    if (!myid) std::cout << "-- Build functional for QoI evaluation" << std::endl;
    if (qoi.compare("local_avg_p") == 0)                                       
        solver->BuildPWObservationFunctional_p(v_qoi_p_data_point, eps_p_qoi); 
    else if (qoi.compare("p_int") == 0)                                        
        solver->BuildVolumeObservationFunctional(                              
                new VectorFEDomainLFIntegrator(zero_vcoeff),                   
                new DomainLFIntegrator(one));                                  
    else // effective permeability                                             
        solver->BuildBdrObservationFunctional(                                 
                new VectorFEBoundaryFluxLFIntegrator(obs_coeff) );

    solver->SetEssBdrConditions(ess_attr, zero_vcoeff);
    solver->BuildForcingTerms(zero_vcoeff, pinflow_coeff, zero);

    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    double Q, C, EQ;

    if (!myid) std::cout << "-- Construct L2ProjectionPDESampler" << std::endl;
    
    auto sampler = make_unique<L2ProjectionPDESampler>
        (pmesh, pembedmesh, dist, *master_list);
    
    if (!myid) std::cout << "-- Sampler BuildHierarchy" << std::endl;
    {
        Timer timer = TimeManager::AddTimer("Sampler: BuildHierarchy -- Total");
        sampler->SetDeRhamSequence(solver->GetSequence());
        sampler->BuildDeRhamSequence(topology, embed_topology);
        sampler->BuildHierarchy();
    }

    if (visualize)
        solver->SaveMeshGLVis("mesh");

    Vector xi, coef, soln, c_soln;


    // Create various timers
    for (int i = 0; i < nLevels; i++)
    {
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Darcy: SolveFwd -- Level ")
                .append(std::to_string(i)));
        }
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sample Generation -- Level ")
                .append(std::to_string(i)));
        }
        {
            Timer timer = TimeManager::AddTimer(
                std::string("MC Sample -- Level ")
                .append(std::to_string(i)));
        }
    }

    // Get iters, dofs for each level
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector darcy_nnz(nLevels);
    Vector sampler_nnz(nLevels);
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);
    if (!myid) std::cout << "-- Sample Generation" << std::endl;
        std::stringstream msg;
    if (!myid) msg << "\nL  " << std::setw(8) << std::left << "E[Q]  "
        << std::setw(8) << std::left << "  Cost (dofs)\n";
    for (int ilevel = 0; ilevel < nLevels; ++ilevel)
    {
        EQ = 0.;
        if (myid == 0 && verbose) std::cout << "Level " << ilevel << ":\n";
        for (int i = 0; i < nsamples; i++)
        {
            Timer mc_timer = TimeManager::GetTimer(                        
                std::string("MC Sample -- Level ")                    
                .append(std::to_string(ilevel)));           
            if (myid == 0 && verbose) std::cout << "\nSampler:";
            {
                Timer sampler_timer = TimeManager::GetTimer(               
                            std::string("Sample Generation -- Level ")
                            .append(std::to_string(ilevel)));  
                sampler->Sample(ilevel, xi);
                sampler->Eval(ilevel, xi, coef);
            }
            if (myid == 0 && verbose) std::cout << "Darcy:\n";
            {
                Timer darcy_timer = TimeManager::GetTimer(                 
                        std::string("Darcy: SolveFwd -- Level ")      
                        .append(std::to_string(ilevel)));
                solver->SolveFwd(ilevel, coef, Q, C);
            }    
            EQ += Q;
        }
        EQ /= static_cast<double>(nsamples);
        if (!myid) msg << ilevel << "  "
            << std::setw(8) << std::left << EQ << "  "
            << std::setw(8) << std::left << C << "\n";
    }  
 
    for (int ilevel = 0; ilevel < nLevels; ++ilevel)
    {
        darcy_nnz[ilevel] = sampler->GetNNZ(ilevel);
        sampler_nnz[ilevel] = solver->GetNNZ(ilevel);
        ndofs_l[ilevel] = solver->GetNumberOfDofs(ilevel);
        ndofs_g[ilevel] = solver->GetGlobalNumberOfDofs(ilevel);
        stoch_size_l[ilevel] = sampler->SampleSize(ilevel);
        stoch_size_g[ilevel] = sampler->GlobalSampleSize(ilevel);
    }

    // Generate some pictures: realization, then DarcySolver will save u,p with uncertain coef
    if (visualize)
    {
        if (myid == 0)
        {
             std::cout << "-- Saving a random input realization," 
             << " and resulting Darcy pressure and velocity.\n"; 
        }
        sampler->Sample(0,xi);
        for (int i = 0; i < nLevels; i++)
        {
            sampler->Eval(i, xi, coef);
            solver->SolveFwd(i, coef, Q, C);
            sampler->SaveFieldGLVis(i, coef, "realization");
        }
    }

    if (!myid) std::cout << msg.str() << std::flush;
    
    ReduceAndOutputDofsInfo(ndofs_l, ndofs_g, darcy_nnz);
    ReduceAndOutputStochInfo(stoch_size_l, stoch_size_g, sampler_nnz);
    
    if (print_time) TimeManager::Print(std::cout);
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
                  
  return EXIT_SUCCESS;
}
