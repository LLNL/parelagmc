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
 
#include "BayesianInverseProblem.hpp"
#include "NormalDistributionSampler.hpp"
#include "MeshUtilities.hpp"

namespace parelagmc 
{

using namespace mfem;

BayesianInverseProblem::BayesianInverseProblem(
            const std::shared_ptr<ParMesh>& mesh_,
            PhysicalMLSolver & solver_,
            MLSampler & prior_,
            parelag::ParameterList& master_list_):
        mesh(mesh_),
        solver(solver_),
        solver_sequence(solver.GetSequence()),
        prior(prior_),
        nLevels(solver_sequence.size()),
        dim(mesh->Dimension()),
        bayesian_list(master_list_.Sublist("Bayesian inverse problem parameters", true)),
        noise(bayesian_list.Get("Noise", 0.1)),
        m(bayesian_list.Get("Number of observational data points", 0)),
        h(bayesian_list.Get("Epsilon for local average pressure", 0.1)),
        v_obs_data_coords(bayesian_list.Get("Observational data coordinates", std::vector<double>{0.5, 0.5})),
        size_obs_data(1),
        fec(nullptr),
        fespace(nullptr)
{
    MPI_Comm_rank(mesh->GetComm(), &myid);
    MPI_Comm_size(mesh->GetComm(), &num_procs);

    // If m==0 observational data is int_D(p), 
    // else observational data is the local average pressure at m interior points in mesh
    if (m > 0)
    {
        size_obs_data = m;
        // Change element attributes of mesh to label observational points of interest
        ChangeMeshAttributes(*mesh, m, v_obs_data_coords);                     
    }  

    g_obs_func.resize(size_obs_data);
    for (int i = 0; i < size_obs_data; i++)
    {    
        g_obs_func[i].resize(nLevels); 
        for(int ilevel(0); ilevel < nLevels; ++ilevel)
        {
            g_obs_func[i][ilevel] = parelag::make_unique<Vector>( solver_sequence[ilevel]->GetNumberOfDofs(dim) );
            *(g_obs_func[i][ilevel]) = 0.;
        }
    } 

    // Generate the vector of elements corresponding to associated points of interest
    fec = parelag::make_unique<L2_FECollection>(0, dim);
    fespace = parelag::make_unique<FiniteElementSpace>(mesh.get(), fec.get());
    if (m==0)
    {
        LinearForm fun_p(fespace.get());
        ConstantCoefficient one(1.0);
        fun_p.AddDomainIntegrator(new DomainLFIntegrator(one));
        fun_p.Update(fespace.get(), *(g_obs_func[0][0]), 0 );
        fun_p.Assemble();
    } 
    else 
    {
        Array<int> obs_data_attr( m +1 );
        ConstantCoefficient one(1.0);
        // Iterate over 'mesh attributes'
        for (int i = 2; i < m+2; i++)
        {
            obs_data_attr = 0;
            obs_data_attr[i-1] = 1;
            RestrictedCoefficient obs_data_coeff(one, obs_data_attr);
            LinearForm fun_p;
            fun_p.AddDomainIntegrator(new DomainLFIntegrator(obs_data_coeff));
            g_obs_func[i-2][0] = parelag::make_unique<Vector>(prior.SampleSize(0));
            fun_p.Update(fespace.get(), *(g_obs_func[i-2][0]), 0 );
            fun_p.Assemble();
        }
    }
    // Compute g_obs_func for levels 1:L-1
    for (int i = 0; i < size_obs_data; i++)
    { 
        for(int ilevel(0); ilevel < nLevels-1; ++ilevel)
        {
            g_obs_func[i][ilevel+1] =
                parelag::make_unique<Vector>( prior.SampleSize(ilevel+1) );
            solver_sequence[ilevel]->GetP(dim)->MultTranspose( *g_obs_func[i][ilevel],
                *g_obs_func[i][ilevel+1]);
            //solver_sequence[ilevel]->GetPi(dim)->GetProjectorMatrix().Mult( *g_obs_func[i][ilevel],
            //    *g_obs_func[i][ilevel+1]);
        }
    }

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                << "*  BayesianInverseProblem \n"
                << "*    Noise: " << noise << "\n"
                << "*    Number of observational data points: " << m << "\n"
                << "*    Epsilon for local average pressure: " << h << "\n"
                << "*    Observational data coordinates: \n"; 
        for (int i=0; i<m; i++)
        {
            std::cout << "*    ";
            for (int d = 0; d < dim; d++)
                std::cout << v_obs_data_coords[d+dim*i] << " ";
            std::cout << '\n';
        }
        std::cout << std::string(50,'*') << '\n';
    }
}
    
void BayesianInverseProblem::GenerateObservationalData()
{
    // Now determine if we compute reference_obs_data or if should read in data
    // Check a good input stream as well as correct size
    // Generate G_obs = G(u) + N(0, noise)
    bool generate_ref_obs_data = bayesian_list.Get(
        "Generate reference observational data", false);
    std::string ref_obs_data_filename = bayesian_list.Get(
        "Reference observational data filename", "reference_observational_data.dat"); 
    
    std::ifstream fid(ref_obs_data_filename.c_str());
    // Check if file exists
    if (!generate_ref_obs_data)
    {
        if (!fid)
        {
            if (myid==0) std::cout << "\nCan not open ref observational data file: " << ref_obs_data_filename
                << "\nWill generate reference observational data instead.\n" << std::endl;
            generate_ref_obs_data = true;
        }
    }
    // Load vector of reference observational data and check size
    if (!generate_ref_obs_data)
    {
        G_obs = parelag::make_unique<Vector>(size_obs_data);
        G_obs->Load(fid);
        if (G_obs->Size() != size_obs_data)
        {
            if (myid==0) std::cout << "\nReference observational data size != " << size_obs_data
                << "\nWill generate reference observational data instead.\n" << std::endl;
            generate_ref_obs_data = true;    
        }
    }
    fid.close();
    Vector xi;
    prior.Sample(0, xi);
    // Generate reference data if necessary
    if (generate_ref_obs_data)
    {
        if (myid==0) std::cout << "-- Generating reference observational data\n";
        Vector u, eta;
        prior.Eval(0, xi, u);
        G_obs = parelag::make_unique<Vector>(size_obs_data);
        
        this->ComputeG(0, u, *G_obs, c, q, false);

        eta.SetSize(G_obs->Size());
        NormalDistributionSampler noise_dist(0, noise);
        noise_dist(eta);
        *G_obs += eta;
    }
}

void BayesianInverseProblem::ComputeG(int ilevel, Vector & k_over_k_ref,
        Vector & G, double & C, double & Q, bool compute_Q)
{
    // Returns p-portion of soln in G with random input k_over_k_ref
    p = parelag::make_unique<Vector>(k_over_k_ref.Size());
    solver.SolveFwd_RtnPressure(ilevel, k_over_k_ref, *p, C, Q, compute_Q);
    
    G.SetSize(size_obs_data);
    for (int i = 0; i < size_obs_data; i++)
        G[i] = dot(*g_obs_func[i][ilevel], *p, mesh->GetComm())/sum(*g_obs_func[i][ilevel], mesh->GetComm()); 
    
    p.reset();
}

void BayesianInverseProblem::ComputeLikelihood(int ilevel, Vector & k_over_k_ref,
        double & likelihood, double & C)
{
    // Returns my_obs_points of G with random input k_over_k_ref
    this->ComputeG(ilevel, k_over_k_ref, Gl, C, q, false);

    Gl -= *G_obs;
    
    likelihood = std::exp((-1./(noise*2))*pow(Gl.Norml2(),2)); 
}

void BayesianInverseProblem::ComputeLikelihoodAndQ(int ilevel, Vector & k_over_k_ref,
        double & likelihood, double & C, double & Q)
{
    // Returns my_obs_points of G with random input k_over_k_ref
    this->ComputeG(ilevel, k_over_k_ref, Gl, C, Q, true);
    Gl -= *G_obs;
    
    likelihood = std::exp((-1./(noise*2))*pow(Gl.Norml2(),2)); 
}

void BayesianInverseProblem::ComputeR(int ilevel, Vector & k_over_k_ref,
        double & R, double & C)
{
    this->ComputeLikelihoodAndQ(ilevel, k_over_k_ref, R, C, q);
    R *= q;
}

void BayesianInverseProblem::SaveObsPointsOfInterestGLVis(
            const std::string prefix) const
{
    GridFunction color(fespace.get());
    color = 1.;
    Vector c, c_next;
    c.SetSize(mesh->GetNE());
    for (int i = 0; i < mesh->GetNE(); i++)
        c[i]=mesh->GetAttribute(i);
    color.MakeRef(fespace.get(), c, 0);
    // Level == 0
    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << "0"
        << "." << std::setw(6) << myid;
    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    color.Save(fid); 

    // Iterate over levels 1 to nLevels
    for (int i = 0; i < nLevels-1; ++i)
    {
        c_next.SetSize(solver_sequence[i+1]->GetNumberOfDofs(dim));
        solver_sequence[i]->GetPi(dim)->GetProjectorMatrix().Mult( c, c_next);
        
        prolongate_to_fine_grid(i+1, c_next, color); 
        std::ostringstream fid_name;
        fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << i+1
            << "." << std::setw(6) << myid;
        std::ofstream fid(fid_name.str().c_str());
        fid.precision(8);
        color.Save(fid);
        c = c_next;
    }

}

void BayesianInverseProblem::prolongate_to_fine_grid(int ilevel, const Vector & coeff,
        Vector & x) const
{
    Vector * feCoeff,  * help;

    feCoeff = new Vector;
    feCoeff->SetDataAndSize( coeff.GetData(), coeff.Size() );
    help = feCoeff;

    for(int lev(ilevel); lev > 0; --lev)
    {
        PARELAG_ASSERT(help->Size() == solver_sequence[lev-1]->ComputeTrueP(dim)->Width());
        feCoeff = new Vector(solver_sequence[lev-1]->ComputeTrueP(dim)->Height() );
        solver_sequence[lev-1]->ComputeTrueP(dim)->Mult(*help, *feCoeff);
        delete help;
        help = feCoeff;
    }

    PARELAG_ASSERT(x.Size()==feCoeff->Size());
    x = *feCoeff;

    delete feCoeff;
}

} /* namespace parelagmc */
