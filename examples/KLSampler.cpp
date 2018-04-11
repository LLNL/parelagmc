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
#include <cstring>
#include <elag.hpp>

#include "NormalDistributionSampler.hpp"
#include "MLSampler.hpp"
#include "KLSampler.hpp"
#include "MaternCovariance.hpp"
#include "AnalyticExponentialCovariance.hpp"
#include "Utilities.hpp"

#include "example_helpers/Build3DMesh.hpp"
#include "example_helpers/CreateSamplerParameterList.hpp"

// Example that samples a truncated KL Expansion of a random field. 
// The underlying covariance function is either a Matern Covariance
// or an AnalyticExponentialCovariance, and various statistics are 
// computed.
  
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

    // Number of mesh refinements
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);
    const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);

    // Output options
    const bool visualize = prob_list.Get("Visualize",false);
    const bool print_time = prob_list.Get("Print timings",true);
    
    // AMGe hierarchy parameters
    int nLevels = prob_list.Get("Number of levels", 2);
    const int coarsening_factor = prob_list.Get("Coarsening factor", 8);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);
    
    // Uncertainty parameters
    const double variance = prob_list.Get("Variance", 1.0);
    const int nsamples = prob_list.Get("Number of samples", 10);
    const std::string sampler_name = prob_list.Get("Sampler name", "analytic");

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  KLSampler.exe \n"
                 << "*  XML filename: " << xml_file << "\n*\n"
                 << "*  Mesh: " << meshfile << '\n'
                 << "*  Serial refinements: " << ser_ref_levels << '\n'
                 << "*  Parallel refinements: " << par_ref_levels << '\n'
                 << "*  Variance: " << variance << '\n'
                 << "*  Number of Samples: " << nsamples << '\n';
        std::cout << "*  KL exansion with ";
        if (sampler_name.compare("analytic") == 0)  
            std::cout << sampler_name;
        else
            std::cout << "matern";
        std::cout << " covariance \n";
        std::cout << std::string(50,'*') << '\n';
    }

    if (!unstructured)
        nLevels = par_ref_levels + 1;

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
 
    // Generate the uniform distribution object
    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    // Generate CovarianceFunction
    std::unique_ptr<CovarianceFunction>  cov_func; 
    if (sampler_name.compare("analytic") == 0)
    {
        if (!myid) std::cout << "-- AnalyticExponentialCovariance " << std::endl;
        cov_func = make_unique<AnalyticExponentialCovariance>( 
            pmesh, *master_list);
    }
    else  
    {
        if (!myid) std::cout << "-- MaternCovariance " << std::endl;
        cov_func = make_unique<MaternCovariance>(pmesh, *master_list);
    } 

    // KL Sampler
    Vector xi, coef;
    auto sampler = make_unique<KLSampler>(pmesh, dist, *cov_func, *master_list);
  
    sampler->BuildDeRhamSequence(topology);
    sampler->BuildHierarchy();
    
    cov_func->ShowMe();
   
    std::vector< std::unique_ptr<HypreParVector >> chi(nLevels);
    chi[0].reset(chi_center_of_mass(pmesh.get()));
    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        chi[ilevel+1] = make_unique<HypreParVector>( *sampler->GetTrueP(ilevel), 0 );
        sampler->GetTrueP(ilevel)->MultTranspose(*chi[ilevel], *chi[ilevel+1]);
    }

    // Error 
    Vector exp_error(nLevels);
    Vector var_error(nLevels);
    Vector m_exp_error(nLevels);
    Vector m_var_error(nLevels);
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);

    if (visualize) sampler->SaveMeshGLVis("mesh");

    // realization computation
    sampler->Sample(0, xi);
    for (int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        sampler->Eval(ilevel, xi, coef);
        if (visualize) sampler->SaveFieldGLVis(ilevel, coef, "kl_realization");
    }
    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        int s_size = sampler->SampleSize(ilevel);
        Vector expectation(s_size);
        Vector chi_cov(s_size);
        Vector marginal_variance(s_size);
        expectation = 0.0;
        chi_cov = 0.0;
        marginal_variance = 0.0;
        Vector & chi_level(*chi[ilevel]);
        {
            Timer mc_timer = TimeManager::AddTimer(
                std::string("KLSampler -- Level ").append(std::to_string(ilevel)));
            for(int i(0); i < nsamples; ++i)
            {
                sampler->Sample(ilevel, xi);
                xi = 1.0;
                sampler->Eval(ilevel, xi, coef);
                double chi_coef = dot(coef, chi_level, comm);
                chi_cov.Add(chi_coef, coef);
                expectation.Add(1., coef);
                for(int k = 0; k < s_size; ++k)
                    marginal_variance(k) += coef(k)*coef(k);
            }
        }
        ndofs_l[ilevel] = s_size;
        ndofs_g[ilevel] = chi[ilevel]->GlobalSize();

        chi_cov *= 1./static_cast<double>(nsamples);
        expectation *= 1./static_cast<double>(nsamples);
        marginal_variance *= 1./static_cast<double>(nsamples);
        if (visualize) sampler->SaveFieldGLVis(ilevel, expectation, "expectation");
        if (visualize) sampler->SaveFieldGLVis(ilevel, chi_cov, "cov_chi");
        if (visualize) sampler->SaveFieldGLVis(ilevel, marginal_variance, "marginal_variance");
         
        exp_error[ilevel] = sampler->ComputeL2Error(
                ilevel, expectation, 0.0);
        var_error[ilevel] = sampler->ComputeL2Error(
                ilevel, marginal_variance, variance);
        m_exp_error[ilevel] = sampler->ComputeMaxError(
                ilevel, expectation, 0.0);
        m_var_error[ilevel] = sampler->ComputeMaxError(
                ilevel, marginal_variance, variance);

    }

    if (myid == 0) std::cout << "\n L2 Sampler Error " << std::endl;
    ReduceAndOutputRandomFieldErrors(exp_error, var_error);
    if (myid == 0) std::cout << "\n Max Sampler Error " << std::endl;
    ReduceAndOutputRandomFieldErrors(m_exp_error, m_var_error);
    ReduceAndOutputStochInfo(ndofs_l, ndofs_g);

    if (print_time) TimeManager::Print(std::cout);
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
