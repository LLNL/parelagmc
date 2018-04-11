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
#include "L2ProjectionPDESampler.hpp"
#include "MeshUtilities.hpp"

using namespace parelagmc;
using namespace parelag;
using namespace mfem;

// Computes various statistics and realizations of the SPDE sampler            
// with non-matching mesh embedding solving the saddle point linear 
// system with solver specified in XML parameter list on SPE10 domain.

// Also saves *many* GLVis figures of the SPE10 dataset itself.

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
    const bool lognormal = prob_list.Get("Lognormal", false);

    if (!unstructured)
        nLevels = par_ref_levels + 1;

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                 << "*  SPE10_L2ProjectionPDESampler.exe \n"
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
        std::cout << "\n*  Number of added elements: ";
        for (const auto i: v_n) std::cout << i << ' ';
        std::cout << '\n';
        std::cout << std::string(50,'*') << '\n';
    }

    InversePermeabilityFunction::ReadPermeabilityFile(permFile, comm);
    
    if(nDimensions == 2)
        InversePermeabilityFunction::Set2DSlice(InversePermeabilityFunction::XY, slice);

    VectorFunctionCoefficient kinv(nDimensions, InversePermeabilityFunction::InversePermeability);
    // Create the finite element mesh
    std::unique_ptr<Mesh> mesh = Create_SPE10_Mesh(nDimensions, v_N, v_h);
    std::unique_ptr<Mesh> embedmesh =  Create_Embedded_SPE10_Mesh(nDimensions, v_N, v_h, v_n);;
    
    if (!myid) std::cout << "-- Serial refinement\n";
    for (int l = 0; l < ser_ref_levels; l++)
        mesh->UniformRefinement();
    for (int l = 0; l < ser_ref_levels; l++) 
       embedmesh->UniformRefinement();
   
    auto pmesh = std::make_shared<ParMesh>(comm, *mesh);
    auto pembedmesh = std::make_shared<ParMesh>(comm, *embedmesh);
 
    // Free the serial mesh
    mesh.reset();
    embedmesh.reset();

    Array<int> level_nElements(nLevels);
    Array<int> embed_level_nElements(nLevels);

    if (!myid) std::cout << "-- Parallel refinement\n";
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

    if (!myid) std::cout << "-- Agglomerating the meshes " << std::endl;
    if (unstructured)
    {
        Timer agg = TimeManager::AddTimer("Mesh Agglomeration (algebraic) -- Total");
        BuildTopologyAlgebraic(pmesh, coarsening_factor, topology);
        BuildTopologyAlgebraic(pembedmesh, coarsening_factor, embed_topology);
    }
    else
    {
        Timer agg = TimeManager::AddTimer("Mesh Agglomeration (geometric) -- Total");
        BuildTopologyGeometric(pmesh, level_nElements, topology);
        BuildTopologyGeometric(pembedmesh, embed_level_nElements, embed_topology);
    }

    NormalDistributionSampler dist(0, variance);
    dist.Split(num_procs, myid);

    if (!myid) std::cout << "-- Construct L2ProjectionPDESampler" << std::endl;
    L2ProjectionPDESampler projsampler(pmesh, pembedmesh, dist, *master_list);
    if (!myid) std::cout << "-- Sampler BuildHierarchy" << std::endl;
    {
        Timer timer = TimeManager::AddTimer("BuildHierarchy -- Total");
        projsampler.BuildDeRhamSequence(topology, embed_topology);
        projsampler.BuildHierarchy();
    }

    std::vector< std::unique_ptr<HypreParVector > > projchi(nLevels);
    projchi[0].reset(chi_center_of_mass(pmesh.get()));

    for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
    {
        projchi[ilevel+1] = make_unique<HypreParVector>( *projsampler.GetTrueP(ilevel), 0 );
        projsampler.GetTrueP(ilevel)->MultTranspose(*projchi[ilevel], *projchi[ilevel+1]);
    }
   
    if (visualize)
        projsampler.SaveMeshGLVis("mesh");
 
    Vector xi, proj_coef;

    // Visualize the nodal values of kinv (SPE10 dataset)  
    Vector m;
    if (visualize)
    {
        int num_v = pmesh->GetNV();
        Vector mx, my, mz;
        mx.SetSize(num_v);
        my.SetSize(num_v);
        mz.SetSize(num_v);
       
        FiniteElementCollection * fec = new L2_FECollection(0,pmesh->Dimension());
        FiniteElementSpace * fespace = new FiniteElementSpace(pmesh.get(), fec);

        GridFunction x(fespace);
        x.MakeOwner(fec);
        x.ProjectCoefficient(kinv);

        x.GetNodalValues(mx, 1); 
        x.GetNodalValues(my, 2); 
        x.GetNodalValues(mz, 3); 
        projsampler.SaveFieldGLVis_H1(0, mx, "kinv_x_nodal");
        projsampler.SaveFieldGLVis_H1(0, my, "kinv_y_nodal");
        projsampler.SaveFieldGLVis_H1(0, mz, "kinv_z_nodal");
    
        SaveFieldGLVis_H1(pmesh.get(), mx, "kinv_test");

    }

    // This only works in serial
    if (num_procs == 1 && visualize)
    {
        Vector mx, my, mz, xy, val;
        const int nx = std::pow(2, ser_ref_levels + par_ref_levels)*v_N[0] + 1;
        const int ny = std::pow(2, ser_ref_levels + par_ref_levels)*v_N[1] + 1;
        const int nz = std::pow(2, ser_ref_levels + par_ref_levels)*v_N[2] + 1;
        m.SetSize(nx*ny*nz); m = 0.;
        mx.SetSize(nx*ny*nz);
        my.SetSize(nx*ny*nz);
        mz.SetSize(nx*ny*nz);
        int v;
        double * vertex;
        xy.SetSize(3);
        val.SetSize(3);
        for (int i_z = 0; i_z < nz; i_z++)
          for (int i_y = 0; i_y < ny; i_y++)
            for (int i_x = 0; i_x < nx; i_x++)
            {
                v = i_x + nx*i_y + i_z*nx*ny;
                vertex = pmesh->GetVertex(v);
                xy[0] = vertex[0];
                xy[1] = vertex[1];
                xy[2] = vertex[2]; 
                //m[v] = InversePermeabilityFunction::PermeabilityXY(xy);
                InversePermeabilityFunction::InversePermeability(xy, val);
                mx[v] = val[0];
                my[v] = val[1];
                mz[v] = val[2];
            }
        projsampler.SaveFieldGLVis_H1(0, mx, "kinv_x");
        projsampler.SaveFieldGLVis_H1(0, my, "kinv_y");
        projsampler.SaveFieldGLVis_H1(0, mz, "kinv_z");
        projsampler.SaveFieldGLVis_H1(0, m, "perm_xy");
    }
     
    projsampler.Sample(0, xi);
    
    if (!myid) std::cout << "-- Realization calculation" << std::endl;
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        projsampler.Eval(ilevel, xi, proj_coef);
        if (visualize) projsampler.SaveFieldGLVis(ilevel, proj_coef, "proj_realization");
        // Save mean + realization
        if (visualize && num_procs==1) projsampler.SaveFieldGLVis_H1Add(ilevel, proj_coef, "kappa", m);
    }

    // Error
    Vector proj_exp_error(nLevels);
    Vector proj_var_error(nLevels);
    const double exact_expectation(lognormal ? std::exp(variance/2.) : 0.0);
    const double exact_variance(lognormal ?
        std::exp(variance)*(std::exp(variance)-1.) : variance);

    // Dof stats
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector nnz(nLevels);
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);

    if (!myid) std::cout << "-- Computing sample statistics" << std::endl;
    for(int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        int s_size = projsampler.SampleSize(ilevel);
        Vector projexpectation(s_size);
        Vector projchi_cov(s_size);
        Vector projmarginal_variance(s_size);
        Vector projtruemarginal_variance(s_size);
        projexpectation = 0.0;
        projchi_cov = 0.0;
        projmarginal_variance = 0.0;
        projtruemarginal_variance = 0.0;
        Vector & projchi_level(*projchi[ilevel]);
        if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sample Generation -- Level ")
                .append(std::to_string(ilevel)));
            for(int i(0); i < nsamples; ++i)
            {
                projsampler.Sample(ilevel, xi);
                projsampler.Eval(ilevel, xi, proj_coef);
                double projchi_coef = dot(proj_coef, projchi_level, comm);
                projchi_cov.Add(projchi_coef, proj_coef);
                projexpectation.Add(1., proj_coef);
                for(int k = 0; k < s_size; ++k)
                {
                    projmarginal_variance(k) += proj_coef(k)*proj_coef(k);
                    projtruemarginal_variance(k) += proj_coef(k)*proj_coef(k);
                }
            }
        }
        nnz[ilevel] = projsampler.GetNNZ(ilevel);
        ndofs_l[ilevel] = projsampler.GetNumberOfDofs(ilevel);
        ndofs_g[ilevel] = projsampler.GetGlobalNumberOfDofs(ilevel);
        stoch_size_l[ilevel] = projsampler.SampleSize(ilevel);
        stoch_size_g[ilevel] = projsampler.GlobalSampleSize(ilevel);

        projchi_cov *= 1./static_cast<double>(nsamples);
        projtruemarginal_variance *= 1./static_cast<double>(nsamples-1);
        for (int k = 0; k < s_size; ++k)
            projtruemarginal_variance(k) -= projexpectation(k)*projexpectation(k)
                /(static_cast<double>(nsamples)*(static_cast<double>(nsamples)-1.));

        projexpectation *= 1./static_cast<double>(nsamples);
        projmarginal_variance *= 1./static_cast<double>(nsamples);
        // this is actually GridFunction::ComputeL2Error ** 2! fyi

        proj_exp_error[ilevel] = projsampler.ComputeL2Error(
                ilevel, projexpectation, exact_expectation);
        proj_var_error[ilevel] = projsampler.ComputeL2Error(
                ilevel, projmarginal_variance, exact_variance);

        if (visualize)
        {
            projsampler.SaveFieldGLVis(ilevel, projexpectation,
                    "proj_expectation");
            projsampler.SaveFieldGLVis(ilevel, projchi_cov,
                    "proj_cov_chi");
            projsampler.SaveFieldGLVis(ilevel, projmarginal_variance,
                    "proj_marginal_variance");
            projsampler.SaveFieldGLVis(ilevel, projtruemarginal_variance,
                    "proj_true_marginal_variance");
        }
    }

    if (myid == 0)                                                             
    {                                                                          
        std::cout << "\nSampler Error: Expected E[u] = " << exact_expectation  
        << ", Expected V[u] = " << exact_variance << '\n'                      
        << "\n L2 Error L2 Projection PDE Sampler \n";                                       
    }  
    ReduceAndOutputRandomFieldErrors(proj_exp_error, proj_var_error);
    ReduceAndOutputBothInfo(ndofs_l, ndofs_g,
            stoch_size_l, stoch_size_g, nnz);
    if (print_time) TimeManager::Print(std::cout);

    InversePermeabilityFunction::ClearMemory();
  }
  catch (std::exception &e)                                                    
  {                                                                            
    if (myid == 0) std::cout << e.what() << std::endl;                         
  }                                                                            
              
  return EXIT_SUCCESS;
}
