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
#include "DarcySolver_Legacy.hpp"
#include "BayesianInverseProblem.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateBayesianParameterList.hpp"

using namespace parelagmc;
using namespace mfem;
using namespace parelag;

// Computes the single-level ratio estimate of the Bayesian posterior          
// expectation of a QoI (eff_perm) using single-level MC estimators for a      
// variety of spatial discretizations with a fixed number of samples to         
// examine the spatial discretization error.

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
        master_list = examplehelpers::CreateBayesianTestParameters();
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
    const int nDimensions = prob_list.Get("nDimensions", 2);                   
                                                                               
    // Number of mesh refinements                                              
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2); 
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);   
                                                                               
    // Output options                                                          
    const bool print_time = prob_list.Get("Print timings",true);               
                                                                               
    // AMGe hierarchy parameters                                               
    int nLevels = prob_list.Get("Number of levels", 2);                        
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);       
    const bool unstructured = prob_list.Get("Unstructured coarsening", false); 
                                                                               
    // Uncertainty parameters                                                  
    const double variance = prob_list.Get("Variance", 1.0);                    
    const int nsamples = prob_list.Get("Number of samples", 10);               
                                                                               
    // Boundary conditions and QoI specifications                              
    const int n_bdr_attributes = prob_list.Get("Number boundary attributes", 6);
    std::vector<int> v_ess_attr = prob_list.Get(                               
        "Essential attributes", std::vector<int>());                           
    std::vector<int> v_obs_attr = prob_list.Get(                               
        "Observational attributes", std::vector<int>());                       
    std::vector<int> v_inflow_attr = prob_list.Get(                            
        "Inflow attributes", std::vector<int>());                              
                                                                               
    // Bayesian parameters                                                     
    ParameterList& bayesian_list = master_list->Sublist(                       
        "Bayesian inverse problem parameters",true);                           
    // Use independent samples                                                 
    const bool independent = bayesian_list.Get(                                
        "Use independent samples", false);       

    // Create mfem::Array from std::vector 
    Array<int> ess_attr(v_ess_attr.data(), n_bdr_attributes);
    Array<int> obs_attr(v_obs_attr.data(), n_bdr_attributes);
    Array<int> inflow_attr(v_inflow_attr.data(), n_bdr_attributes);

    if (!unstructured)
        nLevels = par_ref_levels + 1;

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  RatioEstimator_MC_Spatial_Error.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << '\n'
                 << "*  Embedded Mesh: " << embedfile << '\n'
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Number of Levels: " << nLevels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Quantity of interest: eff_perm" << '\n'
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
        pmesh->ReorientTetMesh();
        pembedmesh->ReorientTetMesh();
    }

    // topology for embedmesh and mesh
    std::vector< std::shared_ptr< AgglomeratedTopology > > topology(nLevels);
    std::vector< std::shared_ptr< AgglomeratedTopology > > embed_topology(nLevels);

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
    // Build deterministic solver
    ConstantCoefficient zero(0.);
    ConstantCoefficient one(1.);
    ConstantCoefficient minus_one(-1.);
    RestrictedCoefficient obs_coeff(one, obs_attr);
    RestrictedCoefficient pinflow_coeff(minus_one, inflow_attr);
    Vector zeros_nDim(nDimensions);
    zeros_nDim = 0.;
    VectorConstantCoefficient zero_vcoeff(zeros_nDim);

    auto solver = make_unique<DarcySolver_Legacy>(pmesh, *master_list);
    solver->BuildHierachySpaces(topology, make_unique<VectorFEMassIntegrator>(one));

    if (!myid) std::cout << "-- Build functional for QoI evaluation" << std::endl;
    // effective permeability
    solver->BuildBdrObservationFunctional(new VectorFEBoundaryFluxLFIntegrator(obs_coeff) );

    solver->SetEssBdrConditions(ess_attr, zero_vcoeff);
    solver->BuildForcingTerms(zero_vcoeff, pinflow_coeff, zero);

    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    if (!myid) std::cout << "-- Construct L2ProjectionPDESampler" << std::endl;
    auto sampler = make_unique<L2ProjectionPDESampler>
        (pmesh, pembedmesh, dist, *master_list);
    if (!myid) std::cout << "-- Sampler BuildHierarchy" << std::endl;
    sampler->SetDeRhamSequence(solver->GetSequence());
    sampler->BuildDeRhamSequence(topology, embed_topology);
    sampler->BuildHierarchy();

    if (!myid) std::cout << "-- Building BayesianInverseProblem" << std::endl;
    auto bayesian_problem = make_unique<BayesianInverseProblem>(
        pmesh, *solver, *sampler, *master_list);

    if (!myid) std::cout << "-- Generate observational data" << std::endl;
    bayesian_problem->GenerateObservationalData();

    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);

    Vector zxi, xi, zcoef, coef;
    double R, Z, c; 
    const double n = static_cast<double>(nsamples);
    double form_var_r, form_var_z;
 
    if (!myid) std::cout << "#Refine" << std::setw(10) << "E[R] "           
                << std::setw(12) << "Var[R] " << std::setw(12) << "E[Z] "      
                << std::setw(12) << "Var[Z] " << std::setw(12) << "E[Q] \n";

    // Solve on "fine" level (where fine level is getting more coarse)
    for (int ilevel = 0; ilevel < nLevels; ++ilevel)
    {
        double c_tot=0.;
        double r=0., r2=0., z=0., z2=0.;
        {
            Timer mc_timer = TimeManager::AddTimer(
                std::string("Compute Estimator -- Level ")
                .append(std::to_string(ilevel)));
            for (int i = 0; i < nsamples; i++)
            {
                // Compute Z
                bayesian_problem->SamplePrior(ilevel, zxi);
                bayesian_problem->EvalPrior(ilevel, zxi, zcoef);
                bayesian_problem->ComputeLikelihood(ilevel, zcoef, Z, c);
                c_tot += c;
                // Compute R = Q*pi_like
                if (independent)
                {
                    bayesian_problem->SamplePrior(ilevel, xi);
                    bayesian_problem->EvalPrior(ilevel, xi, coef);
                }
                else 
                    coef = zcoef;
                bayesian_problem->ComputeR(ilevel, coef, R, c); c_tot += c; 
                r += R; r2 += R*R; 
                z += Z; z2 += Z*Z; 
            }
        }
        r /= n;                                                                
        r2 /= n;                                                               
                                                                               
        z /= n;                                                                
        z2 /= n;        
 
        // Compute variance 
        form_var_r = n*(r2 - r*r)/(n - 1.);                                    
        form_var_z = n*(z2 - z*z)/(n - 1.);              

        ndofs_l[ilevel] = solver->GetNumberOfDofs(ilevel);
        ndofs_g[ilevel] = solver->GetGlobalNumberOfDofs(ilevel);
        stoch_size_l[ilevel] = solver->GetSizeOfStochasticData(ilevel);
        stoch_size_g[ilevel] = solver->GetSizeOfGlobalStochasticData(ilevel);

        if (!myid) std::cout << ilevel << std::setw(15) << r     
            << std::setw(12) << form_var_r << std::setw(12) << z           
            << std::setw(12) << form_var_z << std::setw(12) << r/z << '\n';

    }
    
    ReduceAndOutputBothInfo(ndofs_l, ndofs_g, stoch_size_l, stoch_size_g);

    if (print_time) TimeManager::Print(std::cout);
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
