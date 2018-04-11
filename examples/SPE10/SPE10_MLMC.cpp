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
#include "Utilities.hpp"
#include "MeshUtilities.hpp"
#include "DarcySolver.hpp"
#include "MLSampler.hpp"
#include "KLSampler.hpp"
#include "PDESampler.hpp"
#include "MaternCovariance.hpp"
#include "AnalyticExponentialCovariance.hpp"
#include "MLMC_Manager.hpp"

using namespace parelagmc;
using namespace parelag;
using namespace mfem;

// MLMC simulation using SPE10 dataset for the mixed Darcy problem with
// an uncertain permeability coefficient sampled with a 
// KL expansion: analytic exponential or matern
// covariance function, or the SPDE sampler without mesh embedding. 

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
    const int slice = spe_list.Get("SPE10 slice", -1);
    std::vector<int> v_N = spe_list.Get("Number of elements", std::vector<int>{60,220,85});
    std::vector<double> v_h = spe_list.Get("Element sizes", std::vector<double>{20,10,2});

    // Number of mesh refinements
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);
   
    // AMGe hierarchy specifications
    int nLevels = prob_list.Get("Number of levels", 2);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);
   
    // Output options
    const bool print_time = prob_list.Get("Print timings",true);
    const bool visualize = prob_list.Get("Visualize",true);
   
    // Uncertainty parameters
    const double variance = prob_list.Get("Variance", 1.0);
    const std::string sampler_name = prob_list.Get("Sampler name", "pde");
 
    // Boundary conditions and QoI specifications
    const std::string qoi = prob_list.Get("Quantity of interest", "eff_perm");
    std::vector<double> v_qoi_p_data_point = prob_list.Get(
        "Local average pressure QoI spatial point",              
        std::vector<double>{0.5, 0.5, 0.5});   
    const double eps_p_qoi = prob_list.Get(                                    
        "Epsilon for local average pressure QoI", 0.1);
    const int n_bdr_attributes = prob_list.Get(
        "Number boundary attributes", 6);
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
                 << "*  SPE10_MLMC.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  nDimensions: " << nDimensions << '\n';
                 if (nDimensions == 2)
                    std::cout << "*  Slice: " << par_ref_levels << '\n';
                 std::cout << "*  Number of Levels: " << nLevels << '\n'
                << "*  Quantity of interest: " << qoi << '\n';
        std::cout << "*  Sampler type:" ;
        if (sampler_name == "analytic" || sampler_name == "matern")            
            std::cout << " KL with " << sampler_name << " covariance " << std::endl;
        else                                                                   
            std::cout << " PDE sampler " << std::endl;     
        if (qoi.compare("local_avg_p") == 0)
        {
            std::cout << "*  Pressure QoI spatial point: (";
            for (auto i:v_qoi_p_data_point)
                std::cout << i << " ";
            std::cout << ") with epsilon = " << eps_p_qoi << '\n';
        }
        std::cout << "*  Variance: " << variance << '\n';
        if (unstructured)
        {
            std::cout << "*  Unstructured (algebraic) coarsening \n"
                    << "*  Coarsening Factor: " << coarsening_factor << '\n';
        }
        else
            std::cout << "*  Structured (geometric) coarsening \n";
        std::cout << "*  Number of elements: ";
        for (const auto i: v_N) std::cout << i << ' ';
        std::cout << "\n*  Element sizes: "; 
        for (const auto i: v_h) std::cout << i << ' ';
        std::cout << '\n';

        ess_attr.Print(std::cout << "*  EssentialAttributes: ", n_bdr_attributes);
        obs_attr.Print(std::cout << "*  ObservationAttributes: ", n_bdr_attributes);
        inflow_attr.Print(std::cout << "*  InflowAttributes: ", n_bdr_attributes);
        std::cout << std::string(50,'*') << '\n';
    }

    InversePermeabilityFunction::ReadPermeabilityFile(permFile, comm);

    if(nDimensions == 2)
        InversePermeabilityFunction::Set2DSlice(InversePermeabilityFunction::XY, slice);

    VectorFunctionCoefficient kinv(nDimensions, 
        InversePermeabilityFunction::InversePermeability);

    // Create the finite element mesh
    std::unique_ptr<Mesh> mesh = Create_SPE10_Mesh(nDimensions, v_N, v_h);
    for (int l = 0; l < ser_ref_levels; l++)
        mesh->UniformRefinement();
    
    auto pmesh = std::make_shared<ParMesh>(comm, *mesh);
   
    // Free the serial mesh
    mesh.reset();
    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        if (!unstructured) level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    if (!unstructured) level_nElements[0] = pmesh->GetNE();

    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);
   
    if (unstructured)
        BuildTopologyAlgebraic(pmesh, coarsening_factor, topology);
    else
        BuildTopologyGeometric(pmesh, level_nElements, topology);
 
    ConstantCoefficient zero(0.);
    ConstantCoefficient one(1.);
    RestrictedCoefficient obs_coeff(one, obs_attr);
    RestrictedCoefficient pinflow_coeff(one, inflow_attr);
    Vector zeros_nDim(nDimensions);
    zeros_nDim = 0.;
    VectorConstantCoefficient zero_vcoeff(zeros_nDim);

    auto solver = make_unique<DarcySolver>(pmesh, *master_list);
    if (!myid) std::cout << "-- Darcy BuildHierarchy" << std::endl;
    {
        Timer timer = TimeManager::AddTimer("BuildHierarchy -- Total");
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

    if (visualize) solver->SaveMeshGLVis("mesh");
    
    // Generate the uniform distribution object
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    // Generate CovarianceFunction
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
    sampler->SetDeRhamSequence(solver->GetSequence());
    sampler->BuildHierarchy();
    
    MLMC_Manager manager(comm, nLevels, *solver, *sampler, *master_list);
    manager.Run();

    if (print_time) TimeManager::Print(std::cout);     
    // Free the used memory.
    InversePermeabilityFunction::ClearMemory();
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
                  
  return EXIT_SUCCESS;
}
