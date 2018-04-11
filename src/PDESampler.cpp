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
 
#include "PDESampler.hpp"

#include <cmath>
#include "Utilities.hpp"

namespace parelagmc 
{
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

PDESampler::PDESampler(
            const std::shared_ptr<mfem::ParMesh>& mesh_, 
            NormalDistributionSampler& dist_sampler_,
            ParameterList& master_list_):
        mesh(mesh_),
        nDim(mesh->Dimension()),
        uform(nDim-1),
        sform(nDim),
        dist_sampler(dist_sampler_),
        nLevels(0),
        fespace(nullptr),
        fespace_u(nullptr),
        master_list(master_list_),
        prob_list(master_list.Sublist("Problem parameters", true)),
        save_vtk(prob_list.Get("Save VTK", false)),                            
        lognormal(prob_list.Get("Lognormal", true)),                           
        verbose(prob_list.Get("Verbosity", false)),      
        entire_seq(prob_list.Get("Build entire sequence", false)),                      
        corlen(prob_list.Get("Correlation length", 0.1)),
        alpha(1./(corlen*corlen)),
        matern_coeff(ComputeScalingCoefficientForSPDE(corlen, nDim)),
        if_solver_hybridization(false),
        level_size(0),
        nnz(0),
        block_offsets(0,0),
        true_block_offsets(0,0),
        umap(0),
        pmap(0)
{

    MPI_Comm comm = mesh->GetComm();
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                << "*  PDESampler \n"
                << "*    Correlation length: " << corlen << "\n"
                << "*    Alpha: " << alpha << "\n"
                << "*    Matern Coefficient: " << matern_coeff << "\n";
        if (lognormal) std::cout << "*    Lognormal : true" << std::endl;
        std::cout << std::string(50,'*') << '\n';
    }

}

void PDESampler::BuildDeRhamSequence(
        std::vector< std::shared_ptr<AgglomeratedTopology> > & topology)
{
    nLevels = topology.size();

    int feorder(0);
    int upscalingOrder(0);
    sequence.resize(nLevels);
    {

        Timer timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Total");
 
        if(nDim == 2)
            sequence[0] = std::make_shared<DeRhamSequence2D_Hdiv_FE>(
                    topology[0], 
                    mesh.get(), 
                    feorder);
        else
            sequence[0] = std::make_shared<DeRhamSequence3D_FE>(
                    topology[0], 
                    mesh.get(), 
                    feorder);
        
        Array<Coefficient *> L2coeff;
        Array<VectorCoefficient *> Hdivcoeff;
        Array<VectorCoefficient *> Hcurlcoeff;
        Array<Coefficient *> H1coeff;
        fillVectorCoefficientArray(nDim, upscalingOrder, Hcurlcoeff);
        fillVectorCoefficientArray(nDim, upscalingOrder, Hdivcoeff);
        fillCoefficientArray(nDim, upscalingOrder, L2coeff);
        fillCoefficientArray(nDim, upscalingOrder+1, H1coeff);
        
        DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);

        std::vector<unique_ptr<MultiVector>>
            targets(sequence[0]->GetNumberOfForms());
        int jform(0);
        int jFormStart;
        if (entire_seq)
            jFormStart = 0;
        else
            jFormStart = nDim - 1;
     
        sequence[0]->SetjformStart(jFormStart);
        
        if (entire_seq)
        {
            targets[jform] =
                DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);
        }
        ++jform;

        if (entire_seq)
        {
            targets[jform] =
                DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
        }
        ++jform;

        if (nDim == 2)
        {
            targets[jform] =
                DRSequence_FE->InterpolateScalarTargets(jform,L2coeff);
            ++jform;
        }
        else
        {
            targets[jform] =
                DRSequence_FE->InterpolateVectorTargets(jform,Hdivcoeff);
            ++jform;
            targets[jform] =
                DRSequence_FE->InterpolateScalarTargets(jform,L2coeff);
            ++jform;
        }
        freeCoeffArray(L2coeff);
        freeCoeffArray(Hdivcoeff);
        freeCoeffArray(Hcurlcoeff);
        freeCoeffArray(H1coeff);
        
        // Set targets
        Array<MultiVector *> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();
        sequence[0]->SetTargets(targets_in);
    }

    for(int i(0); i < nLevels-1; ++i)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Sampler: DeRhamSequence Construction -- Level ")
            .append(std::to_string(i+1)));
        Timer tot_timer = TimeManager::GetTimer(
            "Sampler: DeRhamSequence Construction -- Total");
        sequence[i+1] = sequence[i]->Coarsen();
    }
}

void PDESampler::SetDeRhamSequence(
        std::vector< std::shared_ptr<DeRhamSequence> > & sequence_)
{
    sequence = sequence_;
}

void PDESampler::BuildHierarchy()
{
    // Check deRham sequence has been built and get fespaces
    {
        DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);
        fespace = DRSequence_FE->GetFeSpace(sform);
        fespace_u = DRSequence_FE->GetFeSpace(uform);
    }
    
    nLevels = sequence.size();

    Ps.resize(nLevels-1);
    for(int i(0); i < nLevels-1; ++i)
    {
        Ps[i] = sequence[i]->ComputeTrueP(sform);
    }

    block_offsets.SetSize( nLevels, 3 );
    true_block_offsets.SetSize( nLevels, 3 );

    umap.SetSize(nLevels);
    pmap.SetSize(nLevels);

    w_sqrt.resize(nLevels);
    A.resize(nLevels);
    level_size.SetSize(nLevels);
    invA.resize(nLevels);
    Vector alpha_vec;

    // Boundary conditions 
    // Size=2 since 2 forms; ess_attr[1].Size() = 0 because no BC's
    // for that form
    std::vector<Array<int>> ess_bdr(2);
    const int bdr_size = mesh->bdr_attributes.Size() ?
            mesh->bdr_attributes.Max() : 0;
    ess_bdr[0].SetSize(bdr_size);
    ess_bdr[0] = 1;
  
    nnz.SetSize(nLevels);

    for(int i(0); i < nLevels; ++i)
    {
        level_size[i] = sequence[i]->GetNumberOfTrueDofs(sform);

        umap[i] = &(sequence[i]->GetDofHandler(uform)->GetDofTrueDof());
        pmap[i] = &(sequence[i]->GetDofHandler(sform)->GetDofTrueDof());

        // The blocks, managed here
        unique_ptr<HypreParMatrix> pM;
        unique_ptr<HypreParMatrix> pB;
        unique_ptr<HypreParMatrix> pBt;
        unique_ptr<HypreParMatrix> pW;

        {
            auto M_s = sequence[i]->ComputeMassOperator(uform);
            auto W_s = sequence[i]->ComputeMassOperator(sform);
            auto D_s = sequence[i]->GetDerivativeOperator(uform);

            Array<int> ess_dof(sequence[i]->GetNumberOfDofs(uform));
            sequence[i]->GetDofHandler(uform)->MarkDofsOnSelectedBndr(
                    ess_bdr[0], ess_dof);
            for(int irow = 0; irow < ess_dof.Size(); ++irow)
                if(ess_dof[irow])
                    M_s->EliminateRowCol(irow);
            
            D_s->EliminateCols(ess_dof);

            auto B_s = ToUnique(Mult(*W_s, *D_s));
            auto Bt_s = ToUnique(Transpose(*B_s));
            // Get sqrt of digonal of W
            w_sqrt[i] = make_unique<Vector>(level_size[i]);
            W_s->GetDiag(*(w_sqrt[i]));
            {
                double * it = w_sqrt[i]->GetData();
                for(double *end = it+level_size[i]; it != end; ++it)
                    *it = sqrt(*it);
            }
            // Scale W_s by kappa^2
            alpha_vec.SetSize(W_s->Size());
            alpha_vec = -1.0*alpha;
            W_s->ScaleRows(alpha_vec);

            pM = Assemble(*umap[i], *M_s, *umap[i]);
            pB = Assemble(*pmap[i], *B_s, *umap[i]);
            pBt = Assemble(*umap[i], *Bt_s, *pmap[i]);
            pW = Assemble(*pmap[i], *W_s, *pmap[i]);
            
            nnz[i] = pM->NNZ() + pB->NNZ() + pBt->NNZ() + pW->NNZ();
        }

        // Fill block_offsets
        block_offsets(i,0) = 0;
        block_offsets(i,1) = sequence[i]->GetNumberOfDofs(uform);
        block_offsets(i,2) = block_offsets(i,1) + sequence[i]->GetNumberOfDofs(sform);


        true_block_offsets(i,0) = 0;
        true_block_offsets(i,1) = sequence[i]->GetNumberOfTrueDofs(uform);
        true_block_offsets(i,2) = true_block_offsets(i,1) +
                sequence[i]->GetNumberOfTrueDofs(sform);

        Array<int> true_offsets(true_block_offsets.GetRow(i),3);
        A[i] = std::make_shared<MfemBlockOperator>(true_offsets);
        A[i]->SetBlock(0,0,std::move(pM));
        A[i]->SetBlock(0,1,std::move(pBt));
        A[i]->SetBlock(1,0,std::move(pB));
        A[i]->SetBlock(1,1,std::move(pW));

        // Set solver and preconditioner based on parameter list specifications
        auto lib = SolverLibrary::CreateLibrary(
            master_list.Sublist("Preconditioner Library"));
        ParameterList& prob_list = master_list.Sublist("Sampler problem parameters",true);
        const std::string prec_type = prob_list.Get("Linear solver","MINRES-BJ-GS");
        if (prec_type.compare("Hybridization") == 0) if_solver_hybridization = true;
        
        auto solver_list = master_list.Sublist("Preconditioner Library")
                           .Sublist(prec_type);
        if (verbose)
            solver_list.Sublist("Solver Parameters").Set("Print final paragraph",true);

        const std::string new_prec_type = "new prec";
        lib->AddSolver(new_prec_type,std::move(solver_list));

        // Get the factory
        auto prec_factory = lib->GetSolverFactory(new_prec_type);
        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(sequence[i]);
        solver_state->SetBoundaryLabels(ess_bdr);
        solver_state->SetForms({uform,sform});
        if (if_solver_hybridization)
        {
            solver_state->SetExtraParameter("IsSameOrient",(i>0));
            solver_state->SetExtraParameter("L2MassWeight", alpha);
        }

        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sampler: Build Solver (").append(prec_type)
                .append(") -- Level ").append(std::to_string(i)));
            invA[i] = prec_factory->BuildSolver(A[i],*solver_state);
        }

        if (!myid && verbose && i==0)
            std::cout << '\n' << std::string(50,'*') << '\n'
                      << "*  SPDE Sampler "<< '\n'
                      << "*    Solver: " << prec_type << '\n'
                      << std::string(50,'*') << '\n' << std::endl;
    }

    // Create timer for solver
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(ilevel)));
    }
}

void PDESampler::Sample(const int level, Vector & xi)
{
    xi.SetSize( level_size[level] );
    dist_sampler(xi);
}

void PDESampler::Eval(const int level, const Vector & xi, Vector & s)
{
    const int size_s = level_size[level];
    const int size_u = sequence[level]->GetNumberOfDofs(uform);

    s.SetSize( size_s );
    s = 0.0;
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );

    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = -matern_coeff * xi(idof) * tmp(idof);
    }

    // Project rhs_s to correct level 
    while (xi_level < level)
    {
        auto tmp_s = make_unique<Vector>(
            level_size[xi_level+1] );
        Ps[xi_level]->MultTranspose(*rhs_s, *tmp_s);
        ++xi_level;
        rhs_s = std::move(tmp_s);
    }
    // Set rhs_u == 0
    auto rhs_u = make_unique<Vector>( size_u );
    *rhs_u = 0.;

    Array<int> true_offsets(true_block_offsets.GetRow(level), 3);
    Array<int> offsets(block_offsets.GetRow(level), 3);

    // Serial solution vector
    BlockVector sol(offsets);
    sol = 0.;
    {
        Timer timer = TimeManager::GetTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(level)));        
        if (if_solver_hybridization)
        {
            BlockVector rhs(offsets);
            rhs.GetBlock(0) = *rhs_u;
            rhs.GetBlock(1) = *rhs_s;
            invA[level]->Mult(rhs, sol);
        }
        else
        {
            BlockVector psol(true_offsets);
            psol = 0.;
            BlockVector prhs(true_offsets);
            umap[level]->Assemble(*rhs_u, prhs.GetBlock(0));
            pmap[level]->Assemble(*rhs_s, prhs.GetBlock(1));
            invA[level]->Mult(prhs, psol);
            pmap[level]->Distribute(psol.GetBlock(1), sol.GetBlock(1));
        }
    }
    s = sol.GetBlock(1);
    
    if(lognormal)
    {
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
    }
    
}

void PDESampler::Eval(const int level, const Vector & xi, 
    Vector & s, Vector & embed_s, bool use_init)
{
    const int size_s = level_size[level];
    const int size_u = sequence[level]->GetNumberOfDofs(uform);

    s.SetSize( size_s );
    s = 0.0;
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );

    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = -matern_coeff * xi(idof) * tmp(idof);
    }

    // Project rhs_s to correct level 
    while (xi_level < level)
    {
        auto tmp_s = make_unique<Vector>(
            level_size[xi_level+1] );
        Ps[xi_level]->MultTranspose(*rhs_s, *tmp_s);
        ++xi_level;
        rhs_s = std::move(tmp_s);
    }

    // Set rhs_u == 0
    auto rhs_u = make_unique<Vector>( size_u );
    *rhs_u = 0.;

    Array<int> true_offsets(true_block_offsets.GetRow(level), 3);
    Array<int> offsets(block_offsets.GetRow(level), 3);

    // Serial solution vector
    BlockVector sol(offsets);
    sol = 0.;

    if (if_solver_hybridization)
    {
        BlockVector rhs(offsets);
        rhs.GetBlock(0) = *rhs_u;
        rhs.GetBlock(1) = *rhs_s;
        // Create block sol
        if (use_init)
        {
            int init_level = level_size.Find( embed_s.Size() );
            Vector init_s;
            while(init_level > level)
            {
                init_s.SetSize(level_size[init_level-1]);
                Ps[init_level-1]->Mult(embed_s, init_s);
                --init_level;
                embed_s.SetSize(level_size[init_level]);
                embed_s = init_s;
            }
            sol.GetBlock(1) = embed_s;
            sol.GetBlock(0) = 0.;

            // "The HybridizationSolver cannot be used in iterative mode."
            //invA[level]->iterative_mode = true;
        }
        else
        {
            sol = 0.;
            invA[level]->iterative_mode = false;
        }
    
        // Solve
        {
            Timer timer = TimeManager::GetTimer(
                std::string("Sampler: Mult -- Level ")
                .append(std::to_string(level)));
            invA[level]->Mult(rhs, sol);
        }
    }
    else
    {
        BlockVector psol(true_offsets);
        psol = 0.;
        BlockVector prhs(true_offsets);
        umap[level]->Assemble(*rhs_u, prhs.GetBlock(0));
        pmap[level]->Assemble(*rhs_s, prhs.GetBlock(1));
        if (use_init)
        {
            int init_level = level_size.Find( embed_s.Size() );
            Vector init_s;
            while(init_level > level)
            {
                init_s.SetSize(level_size[init_level-1]);
                Ps[init_level-1]->Mult(embed_s, init_s);
                --init_level;
                embed_s.SetSize(level_size[init_level]);
                embed_s = init_s;
            }
            pmap[level]->Assemble(embed_s, psol.GetBlock(1));
            psol.GetBlock(0) = 0.;
            invA[level]->iterative_mode = true;
        }
        else
        {
            psol = 0.;
            invA[level]->iterative_mode = false;
        }
        {
            Timer timer = TimeManager::GetTimer(
                std::string("Sampler: Mult -- Level ")
                .append(std::to_string(level)));
            invA[level]->Mult(prhs, psol);
        }
        pmap[level]->Distribute(psol.GetBlock(1), sol.GetBlock(1));
    }

    s = sol.GetBlock(1);
    embed_s = sol.GetBlock(1);
    
    if(lognormal)
    {
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
    }
    
}

void PDESampler::Eval(const int level, const Vector & xi, Vector & s, Vector & u)
{
    const int size_s = level_size[level];
    const int size_u = sequence[level]->GetNumberOfDofs(uform);

    s.SetSize( size_s );
    u.SetSize( size_u );
    s = 0.;
    u = 0.;
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );

    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = -matern_coeff * xi(idof) * tmp(idof);
    }
    
    // Set rhs_u = 0
    auto rhs_u = make_unique<Vector>( size_u );
    *rhs_u = 0.;

    // Project rhs_s to correct level 
    while (xi_level < level)
    {
        auto tmp_s = make_unique<Vector>(
            level_size[xi_level+1] );
        Ps[xi_level]->MultTranspose(*rhs_s, *tmp_s);
        ++xi_level;
        rhs_s = std::move(tmp_s);
    }

    Array<int> true_offsets(true_block_offsets.GetRow(level), 3);
    Array<int> offsets(block_offsets.GetRow(level), 3);

    // Serial solution vector
    BlockVector sol(offsets);
    sol = 0.;
    {
        Timer timer = TimeManager::GetTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(level)));        
        if (if_solver_hybridization)
        {
            BlockVector rhs(offsets);
            rhs.GetBlock(0) = *rhs_u;
            rhs.GetBlock(1) = *rhs_s;
            invA[level]->Mult(rhs, sol);
        }
        else
        {
            BlockVector psol(true_offsets);
            psol = 0.;
            BlockVector prhs(true_offsets);
            umap[level]->Assemble(*rhs_u, prhs.GetBlock(0));
            pmap[level]->Assemble(*rhs_s, prhs.GetBlock(1));
            invA[level]->Mult(prhs, psol);
            pmap[level]->Distribute(psol.GetBlock(1), sol.GetBlock(1));
            umap[level]->Distribute(psol.GetBlock(0), sol.GetBlock(0));
        }
    }

    // Set sol.GetBlock(1) == s
    u = sol.GetBlock(0);
    s = sol.GetBlock(1);
    
    if(lognormal)
    {
        for(int idof = 0; idof < u.Size(); ++idof)
            u(idof) = exp( u(idof) );
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
    }
    
}
double PDESampler::ComputeL2Error(
    int level,    
    const Vector & coeff,
    double exact) const
{
    GridFunction x;
    prolongate_to_fine_grid(level, coeff, x);
    ConstantCoefficient exact_soln(exact);
    const double err = x.ComputeL2Error(exact_soln);
    return err * err;
}

double PDESampler::ComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    const double e1 = coeff.Max() - exact;
    const double e2 = exact - coeff.Min();
    return std::max(e1, e2);

}

void PDESampler::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     mesh->Print(mesh_ofs);
}

void PDESampler::SaveFieldGLVis(int level, 
        const Vector & coeff, 
        const std::string prefix) const
{
    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << level 
        << "." << std::setw(6) << myid;

    GridFunction x;
    prolongate_to_fine_grid(level, coeff, x);

    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);
    if (save_vtk)
    {
        // Printing to VTK file format
        std::ostringstream v_name;
        v_name << prefix << "_L" << std::setfill('0') << std::setw(2) <<
                    level << "." << std::setw(6) << myid << ".vtk";
        std::ofstream vid(v_name.str().c_str());
        vid.precision(8);
        mesh->PrintVTK(vid);
        x.SaveVTK(vid, "value", 0);
    }
}

void PDESampler::SaveFieldGLVis_u(int level, 
        const Vector & coeff, 
        const std::string prefix) const
{
    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << level 
        << "." << std::setw(6) << myid;

    GridFunction x;
    prolongate_to_fine_grid_u(level, coeff, x);

    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);
}

void PDESampler::SaveFieldGLVis_H1(int level, 
        const Vector & coeff, 
        const std::string prefix) const
{
    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << level 
        << "." << std::setw(6) << myid;

    GridFunction x;
    prolongate_to_fine_grid_H1(level, coeff, x);

    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);

    if (save_vtk)
    {

        // Printing to VTK file format
        std::ostringstream v_name;
        v_name << prefix << "_L" << std::setfill('0') << std::setw(2) <<
                    level << "." << std::setw(6) << myid << ".vtk";
        std::ofstream vid(v_name.str().c_str());
        vid.precision(8);
        mesh->PrintVTK(vid);
        x.SaveVTK(vid, "value", 0);
    }
}
void PDESampler::SaveFieldGLVis_H1Add(int level, 
        const Vector & coeff, 
        const std::string prefix,
        const Vector & mu) const
{
    FiniteElementCollection * fecH1 = new H1_FECollection(1,nDim);
    FiniteElementSpace * fespaceH1 = new FiniteElementSpace(mesh.get(), fecH1);

    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << level 
        << "." << std::setw(6) << myid;

    GridFunction x;
    prolongate_to_fine_grid(level, coeff, x);

    Vector node_values;
    x.GetNodalValues(node_values);
    for (int i = 0; i < node_values.Size(); i++)
        node_values[i] += mu[i]; 
    x.MakeRef(fespaceH1, node_values, 0);
    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);
    
    if (save_vtk)
    {
        // Printing to VTK file format
        std::ostringstream v_name;
        v_name << prefix << "_L" << std::setfill('0') << std::setw(2) <<
                    level << "." << std::setw(6) << myid << ".vtk";
        std::ofstream vid(v_name.str().c_str());
        vid.precision(8);
        mesh->PrintVTK(vid);
        x.SaveVTK(vid, "value", 0);
    }
    delete fecH1;
    delete fespaceH1;
}

void PDESampler::glvis_plot(int ilevel, 
        const Vector & coeff, 
        const std::string prefix,
        std::ostream & socket) const
{
    GridFunction x;
    prolongate_to_fine_grid(ilevel, coeff, x);

    socket << "parallel " << num_procs << " " << myid << "\n";
    socket.precision(8);
    socket << "solution\n" << *mesh << x << std::flush
           << "window_title '" << prefix << " Level " << ilevel << "'" << std::endl;
    MPI_Barrier(mesh->GetComm());
}

void PDESampler::prolongate_to_fine_grid(int ilevel, 
        const Vector & coeff, 
        GridFunction & x) const
{
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        auto help = make_unique<Vector>( Ps[lev-1]->Height() );
        Ps[lev-1]->Mult(*feCoeff, *help);
        feCoeff = std::move(help);
    }

    x.MakeRef(fespace, *feCoeff, 0);

    if(ilevel!=0)
    {
        x.MakeDataOwner();
        feCoeff->StealData();
    }
}

void PDESampler::prolongate_to_fine_grid_u(int ilevel, 
        const Vector & coeff, 
        GridFunction & x) const
{
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        auto help = make_unique<Vector>(sequence[lev-1]->ComputeTrueP(uform)->Height() );
        (sequence[lev-1]->ComputeTrueP(uform))->Mult(*feCoeff, *help);
        feCoeff = std::move(help);
    }

    x.MakeRef(fespace_u, *feCoeff, 0);

    if(ilevel!=0)
    {
        x.MakeDataOwner();
        feCoeff->StealData();
    }
}

void PDESampler::prolongate_to_fine_grid_H1(int ilevel, 
        const Vector & coeff, 
        GridFunction & x) const
{
    FiniteElementCollection * fecH1 = new H1_FECollection(1,nDim);
    FiniteElementSpace * fespaceH1 = new FiniteElementSpace(mesh.get(), fecH1);

    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        auto help = make_unique<Vector>(Ps[lev-1]->Height() );
        Ps[lev-1]->Mult(*feCoeff, *help);
        feCoeff = std::move(help);
    }

    x.MakeRef(fespaceH1, *feCoeff, 0);
    x.MakeOwner(fecH1);
    
    if(ilevel!=0)
    {
        x.MakeDataOwner();
        feCoeff->StealData();
    }
}
} /* namespace parelagmc */
