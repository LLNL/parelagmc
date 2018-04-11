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
#include "MLSampler.hpp"
#include "KLSampler.hpp"
#include "PDESampler.hpp"
#include "L2ProjectionPDESampler.hpp"
#include "MaternCovariance.hpp"
#include "AnalyticExponentialCovariance.hpp"
#include "Utilities.hpp"
#include "MeshUtilities.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateSamplerParameterList.hpp"

// Computes statistics of many realizations of a random field (Gaussian or 
// log-normal) using analytic exponential covariance KLE, Matern covariance 
// KLE, and SPDE sampler (with and without non-matching mesh embedding). 

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

    // The file from which to read the mesh
    const std::string meshfile =
        prob_list.Get("Mesh file", "no mesh name found");
    const std::string embedfile =
        prob_list.Get("Embedded mesh file", "no mesh name found");

    // Number of mesh refinements
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);

    // Output options
    const bool visualize = prob_list.Get("Visualize",false);

    // AMGe hierarchy parameters
    int nLevels = prob_list.Get("Number of levels", 2);
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);

    // Uncertainty parameters
    const double variance = prob_list.Get("Variance", 1.0);
    const int nsamples = prob_list.Get("Number of samples", 10);
    const std::string sampler_name = prob_list.Get("Sampler name", "analytic");
    
    if (!unstructured)
        nLevels = par_ref_levels + 1;
    
    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
            << "*  SamplerTest.exe: Computes realizations using analytic\n" 
            << "*  exponenetial covariance KLE, matern covariance KLE, and \n"           
            << "*  SPDE sampler (with and without non-matching mesh embedding) \n*\n"
            << "*  XML filename: " << xml_file << "\n"
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
  
    // Generate the uniform distribution object
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    if (!myid) std::cout << "-- Construct AnalyticCovariance based KLSampler" << std::endl;       
    // Create KL Sampler with analytic covariance function                      
    AnalyticExponentialCovariance analytic_cov(
        pmesh, *master_list);
    KLSampler analytic_sampler(pmesh, dist, analytic_cov, *master_list);
    analytic_sampler.BuildDeRhamSequence(topology);
    analytic_sampler.BuildHierarchy();

    if (!myid) std::cout << "-- Construct MaternCovariance based KLSampler" << std::endl;       
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
    if (!myid) std::cout << "-- Construct solution vectors" << std::endl; 
    std::vector<std::unique_ptr<HypreParVector>> a_chi(nLevels);
    a_chi[0].reset(chi_center_of_mass(pmesh.get()));
    std::vector<std::unique_ptr<HypreParVector> > m_chi(nLevels);
    m_chi[0].reset(chi_center_of_mass(pmesh.get()));
    std::vector<std::unique_ptr<HypreParVector >> p_chi(nLevels);
    p_chi[0].reset(chi_center_of_mass(pmesh.get()));
    std::vector<std::unique_ptr<HypreParVector >> pp_chi(nLevels);
    pp_chi[0].reset(chi_center_of_mass(pmesh.get()));

    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        a_chi[ilevel+1] = make_unique<HypreParVector>( *analytic_sampler.GetTrueP(ilevel), 0 );
        analytic_sampler.GetTrueP(ilevel)->MultTranspose(*a_chi[ilevel], *a_chi[ilevel+1]);
        m_chi[ilevel+1] = make_unique<HypreParVector>( *matern_sampler.GetTrueP(ilevel), 0 );
        matern_sampler.GetTrueP(ilevel)->MultTranspose(*m_chi[ilevel], *m_chi[ilevel+1]);
        p_chi[ilevel+1] = make_unique<HypreParVector>( *pde_sampler.GetTrueP(ilevel), 0 );
        pde_sampler.GetTrueP(ilevel)->MultTranspose(*p_chi[ilevel], *p_chi[ilevel+1]);
        pp_chi[ilevel+1] = make_unique<HypreParVector>( *pde_sampler.GetTrueP(ilevel), 0 );
        pde_sampler.GetTrueP(ilevel)->MultTranspose(*pp_chi[ilevel], *pp_chi[ilevel+1]);
    }

    if (visualize) pde_sampler.SaveMeshGLVis("mesh");
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        int s_size = matern_sampler.SampleSize(ilevel);
        Vector a_expectation(s_size);
        Vector a_chi_cov(s_size);
        Vector a_marginal_variance(s_size);
        a_expectation = 0.0;
        a_chi_cov = 0.0;
        a_marginal_variance = 0.0;
        Vector & a_chi_level(*a_chi[ilevel]);
        
        Vector m_expectation(s_size);
        Vector m_chi_cov(s_size);
        Vector m_marginal_variance(s_size);
        m_expectation = 0.0;
        m_chi_cov = 0.0;
        m_marginal_variance = 0.0;
        Vector & m_chi_level(*m_chi[ilevel]);

        Vector p_expectation(s_size);
        Vector p_chi_cov(s_size);
        Vector p_marginal_variance(s_size);
        p_expectation = 0.0;
        p_chi_cov = 0.0;
        p_marginal_variance = 0.0;
        Vector & p_chi_level(*p_chi[ilevel]);
        
        Vector pp_expectation(s_size);
        Vector pp_chi_cov(s_size);
        Vector pp_marginal_variance(s_size);
        pp_expectation = 0.0;
        pp_chi_cov = 0.0;
        pp_marginal_variance = 0.0;
        Vector & pp_chi_level(*pp_chi[ilevel]);

        if (!myid) std::cout << "-- Sample on level " << ilevel << std::endl;
        for (int i = 0; i < nsamples; i++)
        {
            projpde_sampler.Sample(ilevel, embed_xi); 
            projpde_sampler.Transfer(ilevel, embed_xi, xi);
            
            projpde_sampler.Eval(ilevel, embed_xi, projpde_coef);
            double pp_chi_coef = dot(projpde_coef, pp_chi_level, comm);
            pp_chi_cov.Add(pp_chi_coef, projpde_coef);
            pp_expectation.Add(1., projpde_coef);
            for(int k = 0; k < s_size; ++k)
                pp_marginal_variance(k) += projpde_coef(k)*projpde_coef(k);
            
            pde_sampler.Eval(ilevel, xi, pde_coef);
            double p_chi_coef = dot(pde_coef, p_chi_level, comm);
            p_chi_cov.Add(p_chi_coef, pde_coef);
            p_expectation.Add(1., pde_coef);
            for(int k = 0; k < s_size; ++k)
                p_marginal_variance(k) += pde_coef(k)*pde_coef(k);
        
            analytic_sampler.Eval(ilevel, xi, analytic_coef);
            double a_chi_coef = dot(analytic_coef, a_chi_level, comm);
            a_chi_cov.Add(a_chi_coef, analytic_coef);
            a_expectation.Add(1., analytic_coef);
            for(int k = 0; k < s_size; ++k)
                a_marginal_variance(k) += analytic_coef(k)*analytic_coef(k);
            
            matern_sampler.Eval(ilevel, xi, matern_coef);
            double m_chi_coef = dot(matern_coef, m_chi_level, comm);
            m_chi_cov.Add(m_chi_coef, matern_coef);
            m_expectation.Add(1., matern_coef);
            for(int k = 0; k < s_size; ++k)
                m_marginal_variance(k) += matern_coef(k)*matern_coef(k);
        }
        
        if (!myid) std::cout << "-- Save solution vectors" << std::endl;
        a_chi_cov *= 1./static_cast<double>(nsamples);
        a_expectation *= 1./static_cast<double>(nsamples);
        a_marginal_variance *= 1./static_cast<double>(nsamples); 

        m_chi_cov *= 1./static_cast<double>(nsamples);
        m_expectation *= 1./static_cast<double>(nsamples);
        m_marginal_variance *= 1./static_cast<double>(nsamples);

        p_chi_cov *= 1./static_cast<double>(nsamples);
        p_expectation *= 1./static_cast<double>(nsamples);
        p_marginal_variance *= 1./static_cast<double>(nsamples);

        pp_chi_cov *= 1./static_cast<double>(nsamples);
        pp_expectation *= 1./static_cast<double>(nsamples);
        pp_marginal_variance *= 1./static_cast<double>(nsamples);
        
        if (visualize)
        { 
            analytic_sampler.SaveFieldGLVis(ilevel, a_expectation, 
                "analytic_expectation");
            analytic_sampler.SaveFieldGLVis(ilevel, a_marginal_variance, 
                "analytic_marginal_variance");
            analytic_sampler.SaveFieldGLVis(ilevel, a_chi_cov, 
                "analytic_chi_cov");
            
            matern_sampler.SaveFieldGLVis(ilevel, m_expectation, 
                "matern_expectation");
            matern_sampler.SaveFieldGLVis(ilevel, m_marginal_variance, 
                "matern_marginal_variance");
            matern_sampler.SaveFieldGLVis(ilevel, m_chi_cov, 
                "matern_chi_cov");

            pde_sampler.SaveFieldGLVis(ilevel, p_expectation, 
                "pde_expectation");
            pde_sampler.SaveFieldGLVis(ilevel, p_marginal_variance, 
                "pde_marginal_variance");
            pde_sampler.SaveFieldGLVis(ilevel, p_chi_cov, 
                "pde_chi_cov"); 

            projpde_sampler.SaveFieldGLVis(ilevel, pp_expectation, 
                "proj_pde_expectation");
            projpde_sampler.SaveFieldGLVis(ilevel, pp_marginal_variance, 
                "proj_pde_marginal_variance");
            projpde_sampler.SaveFieldGLVis(ilevel, pp_chi_cov, 
                "proj_pde_chi_cov"); 
        }
    }
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
