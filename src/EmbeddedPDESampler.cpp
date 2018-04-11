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
 
#include <cmath>

#include "EmbeddedPDESampler.hpp"
#include "Utilities.hpp"

namespace parelagmc 
{
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

EmbeddedPDESampler::EmbeddedPDESampler(
            const std::shared_ptr<mfem::ParMesh>& orig_mesh_,
            const std::shared_ptr<mfem::ParMesh>& embed_mesh_,
            NormalDistributionSampler& dist_sampler_,
            std::vector<Array<int>> & materialId_,
            ParameterList& master_list_):
        nDim(orig_mesh_->Dimension()),
        uform(nDim-1),
        sform(nDim),
        dist_sampler(dist_sampler_),
        nLevels(materialId_.size()),
        orig_mesh(orig_mesh_),
        embed_mesh(embed_mesh_),
        fespace(nullptr),
        fespace_u(nullptr),
        master_list(master_list_),
        prob_list(master_list.Sublist("Problem parameters", true)),
        save_vtk(prob_list.Get("Save VTK", false)),
        lognormal(prob_list.Get("Lognormal", true)),
        verbose(prob_list.Get("Verbosity", false)),
        corlen(prob_list.Get("Correlation length", 0.1)),
        alpha(1./(corlen*corlen)),
        matern_coeff(ComputeScalingCoefficientForSPDE(corlen, nDim)),
        if_solver_hybridization(false),
        block_offsets(0,0),
        true_block_offsets(0,0),
        level_size(0),
        orig_level_size(0),
        nnz(0),
        umap(0),
        pmap(0)
{
    MPI_Comm_rank(embed_mesh->GetComm(), &myid);
    MPI_Comm_size(embed_mesh->GetComm(), &num_procs);

    meshP_s.resize(nLevels);
    // Determine orig_level_size based on materialId
    orig_level_size.SetSize(nLevels);
    level_size.SetSize(nLevels);

    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        int orig_s_size = 0;
        const int * mat_id = materialId_[ilevel].GetData();
        for (int i = 0; i < materialId_[ilevel].Size(); i++)
        {
            if (mat_id[i] == 1)
                orig_s_size++;
        }
        level_size[ilevel] = materialId_[ilevel].Size();
        orig_level_size[ilevel] = orig_s_size;
        // Create prolongation matrices for mesh
        // Silliness: Want meshP[i]: orig_level_size[i] x level_size[i]
        // but there's trouble since we'd need sequence (of original) 
        // so now we have a block of zeros.
        meshP_s[ilevel] = make_unique<SparseMatrix>(level_size[ilevel], level_size[ilevel]); 
        int j = 0;
        for (int row = 0; row < level_size[ilevel]; row++)
        {
            if (mat_id[row] == 1)
            {
                meshP_s[ilevel]->Set(j,row,1.0);
                j++;
            }
        }
        meshP_s[ilevel]->Finalize();
    } 

    //matern_coeff = ComputeScalingCoefficientForSPDE(corlen, nDim);

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                << "*  EmbeddedPDESampler \n"
                << "*    Correlation length: " << corlen << "\n"
                << "*    Alpha: " << alpha << "\n"
                << "*    Matern Coefficient: " << matern_coeff << "\n";
        if (lognormal) std::cout << "*    Lognormal : true" << std::endl;
        std::cout << std::string(50,'*') << '\n';
    }

}

void EmbeddedPDESampler::BuildDeRhamSequence(
        std::vector< std::shared_ptr<AgglomeratedTopology> > & embed_topology)
{
    int feorder(0);
    int upscalingOrder(0);
    embed_sequence.resize(nLevels);

    {
        Timer timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Total");

        if(nDim == 2)
            embed_sequence[0] = std::make_shared<DeRhamSequence2D_Hdiv_FE>(
                    embed_topology[0], 
                    embed_mesh.get(), 
                    feorder);
        else
            embed_sequence[0] = std::make_shared<DeRhamSequence3D_FE>(
                    embed_topology[0], 
                    embed_mesh.get(), 
                    feorder);

        DeRhamSequenceFE * DRSequence_FE = embed_sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);

        Array<Coefficient *> L2coeff;
        Array<VectorCoefficient *> Hdivcoeff;
        Array<VectorCoefficient *> Hcurlcoeff;
        Array<Coefficient *> H1coeff;
        fillVectorCoefficientArray(nDim, upscalingOrder, Hcurlcoeff);
        fillVectorCoefficientArray(nDim, upscalingOrder, Hdivcoeff);
        fillCoefficientArray(nDim, upscalingOrder, L2coeff);
        fillCoefficientArray(nDim, upscalingOrder+1, H1coeff);
        
        std::vector<std::unique_ptr<MultiVector>>
            targets(embed_sequence[0]->GetNumberOfForms());
        int jform(0);

        targets[jform] =
            DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);
        ++jform;

        targets[jform] =
            DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
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
        embed_sequence[0]->SetTargets(targets_in);

    }

    for(int i(0); i < nLevels-1; ++i)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Sampler: DeRhamSequence Construction -- Level ")
            .append(std::to_string(i+1)));
        Timer tot_timer = TimeManager::GetTimer(
            "Sampler: DeRhamSequence Construction -- Total");
        embed_sequence[i+1] = embed_sequence[i]->Coarsen();
    }
}

void EmbeddedPDESampler::BuildHierarchy()
{
    // Check deRham sequence has been built and set fespace
    {
        DeRhamSequenceFE * DRSequence_FE = embed_sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);
        fespace = DRSequence_FE->GetFeSpace(sform);
        fespace_u = DRSequence_FE->GetFeSpace(uform);
    }

    Ps.resize(nLevels-1);
    for(int i(0); i < nLevels-1; ++i)
        Ps[i] = embed_sequence[i]->ComputeTrueP(sform);
    
    block_offsets.SetSize( nLevels, 3 );
    true_block_offsets.SetSize( nLevels, 3 );

    umap.SetSize(nLevels);
    pmap.SetSize(nLevels);

    w_sqrt.resize(nLevels);
    A.resize(nLevels);
    level_size.SetSize(nLevels);
    invA.resize(nLevels);
    Vector alpha_vec;
    meshP.resize(nLevels);
    
    // Boundary conditions 
    // Size=2 since 2 forms; ess_attr[1].Size() = 0 because no BC's
    // for that form
    std::vector<Array<int>> ess_bdr(2);
    const int bdr_size = embed_mesh->bdr_attributes.Size() ?
            embed_mesh->bdr_attributes.Max() : 0;
    ess_bdr[0].SetSize(bdr_size);
    ess_bdr[0] = 1;

    nnz.SetSize(nLevels);
    for(int i(0); i < nLevels; ++i)
    {
        level_size[i] = embed_sequence[i]->GetNumberOfTrueDofs(sform);

        umap[i] = &(embed_sequence[i]->GetDofHandler(uform)->GetDofTrueDof());
        pmap[i] = &(embed_sequence[i]->GetDofHandler(sform)->GetDofTrueDof());

        // The blocks, managed here
        unique_ptr<HypreParMatrix> pM;
        unique_ptr<HypreParMatrix> pB;
        unique_ptr<HypreParMatrix> pBt;
        unique_ptr<HypreParMatrix> pW;

        {
            auto M_s = embed_sequence[i]->ComputeMassOperator(uform);
            auto W_s = embed_sequence[i]->ComputeMassOperator(sform);
            auto D_s = embed_sequence[i]->GetDerivativeOperator(uform);

            Array<int> ess_dof(embed_sequence[i]->GetNumberOfDofs(uform));
            embed_sequence[i]->GetDofHandler(uform)->MarkDofsOnSelectedBndr(
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
        block_offsets(i,1) = embed_sequence[i]->GetNumberOfDofs(uform);
        block_offsets(i,2) = block_offsets(i,1) + embed_sequence[i]->GetNumberOfDofs(sform);


        true_block_offsets(i,0) = 0;
        true_block_offsets(i,1) = embed_sequence[i]->GetNumberOfTrueDofs(uform);
        true_block_offsets(i,2) = true_block_offsets(i,1) +
                embed_sequence[i]->GetNumberOfTrueDofs(sform);

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
        solver_state->SetDeRhamSequence(embed_sequence[i]);
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
                      << "*  Embedded SPDE Sampler "<< '\n'
                      << "*    Solver: " << prec_type << '\n'
                      << std::string(50,'*') << '\n' << std::endl;
        
        // Now that we have the maps, Assemble the mesh prolongators where
        // meshP_s is created in the constructor.    
        meshP[i] = Assemble(*pmap[i], *meshP_s[i], *pmap[i]);
        meshP_s[i].reset(); 
    }

    // Create timer for solver
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(ilevel)));
    }
}

void EmbeddedPDESampler::Sample(const int level, Vector & xi)
{
    xi.SetSize( level_size[level] );
    dist_sampler(xi);
}

void EmbeddedPDESampler::Eval(const int level, const Vector & xi, Vector & s)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();

    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    Vector embed_s( size_s );
    embed_s = 0.0;
    
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

    // Set sol.GetBlock(1) == s
    embed_s = sol.GetBlock(1);
    
    // Project to original mesh
    s.SetSize(level_size[level]);
    meshP[level]->Mult(embed_s, s);
    // Silliness to remove zeros from end of s after mult by meshP
    s.SetSize(orig_level_size[level]);
    if(lognormal)
    {
        for(int idof = 0; idof < s.Size(); ++idof)
            s(idof) = exp( s(idof) );
    }
}

void EmbeddedPDESampler::Eval(const int level, const Vector & xi, Vector & s,
    Vector & embed_s, bool use_init)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();

    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = -matern_coeff * xi(idof) * tmp(idof);
    }

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
    if (if_solver_hybridization)
    {
        // Create block rhs
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

    // Set sol.GetBlock(1) == s
    embed_s = sol.GetBlock(1);
    
    // Project to original mesh
    s.SetSize(size_s);
    meshP[level]->Mult(embed_s, s);
    // Silliness to remove zeros from end of s after mult by meshP
    s.SetSize(orig_level_size[level]);

    if(lognormal)
    {
        for(int idof = 0; idof < s.Size(); ++idof)
            s(idof) = exp( s(idof) );
    }
}

void EmbeddedPDESampler::EmbedEval(const int level, const Vector & xi, Vector & s)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();

    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    Vector embed_s( size_s );
    embed_s = 0.0;
    
    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = -matern_coeff * xi(idof) * tmp(idof);
    }

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
        }
    }

    // Set sol.GetBlock(1) == s
    s = sol.GetBlock(1);
    if(lognormal)
    {
        for(int idof = 0; idof < size_s; ++idof)
            s(idof) = exp( s(idof) );
    }
}

void EmbeddedPDESampler::EmbedEval(const int level, const Vector & xi, Vector & s, Vector & u)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();

    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    Vector embed_s( size_s );
    embed_s = 0.0;
    
    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = -matern_coeff * xi(idof) * tmp(idof);
    }

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

    s = sol.GetBlock(1);
    u = sol.GetBlock(0);
    if(lognormal)
    {
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
        for(int idof = 0; idof < u.Size(); ++idof)
            u(idof) = exp( u(idof) );
    }
}
double EmbeddedPDESampler::EmbedComputeL2Error(
    int level,    
    const Vector & coeff,
    double exact) const
{
    GridFunction x;
    embed_prolongate_to_fine_grid(level, coeff, x);
    ConstantCoefficient exact_soln(exact);
    const double err = x.ComputeL2Error(exact_soln);
    return err * err;
}

double EmbeddedPDESampler::ComputeL2Error(
    int level,
    const Vector & coeff,
    double exact) const
{
    GridFunction x;
    prolongate_to_fine_grid(level, coeff, x);
    ConstantCoefficient exact_soln(exact);
    const double err = x.ComputeL2Error(exact_soln);
    return err*err;
}

double EmbeddedPDESampler::EmbedComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    return std::max(coeff.Max() - exact,
            std::abs(exact - coeff.Min()));
    
}

double EmbeddedPDESampler::ComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    // Project to original mesh first
    Vector orig_coeff(level_size[level]);
    //feCoeff->SetSize(level_size[0]);
    meshP[level]->Mult(coeff, orig_coeff);
    orig_coeff.SetSize(orig_level_size[level]);
    const double e1 = orig_coeff.Max() - exact;
    const double e2 = exact - orig_coeff.Min();   
    return std::max(e1, e2); 

}

void EmbeddedPDESampler::EmbedSaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     embed_mesh->Print(mesh_ofs);
}

void EmbeddedPDESampler::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     orig_mesh->Print(mesh_ofs);
}

void EmbeddedPDESampler::EmbedSaveFieldGLVis(int level, 
        const Vector & coeff, 
        const std::string prefix) const
{
    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << level 
        << "." << std::setw(6) << myid;
    GridFunction x;
    embed_prolongate_to_fine_grid(level, coeff, x);

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
        embed_mesh->PrintVTK(vid);
        x.SaveVTK(vid, "value", 0);
    } 
}

void EmbeddedPDESampler::EmbedSaveFieldGLVis_u(int level, 
        const Vector & coeff, 
        const std::string prefix) const
{
    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << level 
        << "." << std::setw(6) << myid;
    GridFunction x;
    embed_prolongate_to_fine_grid_u(level, coeff, x);

    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);
}

void EmbeddedPDESampler::SaveFieldGLVis(int level,
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
        orig_mesh->PrintVTK(vid);
        x.SaveVTK(vid, "value", 0);
    } 
}

void EmbeddedPDESampler::Transfer(
        int level,
        const mfem::Vector & coeff,
        mfem::Vector & orig_coeff) const
{
    // Project coeff to "original" mesh on a particular level
    orig_coeff.SetSize(level_size[level]);
    meshP[level]->Mult(coeff, orig_coeff);    
    // Silliness to remove zeros from end after mult by meshP
    orig_coeff.SetSize(orig_level_size[level]);
}

void EmbeddedPDESampler::embed_glvis_plot(int ilevel, 
        const Vector & coeff, 
        const std::string prefix,
        std::ostream & socket) const
{
    GridFunction x;
    embed_prolongate_to_fine_grid(ilevel, coeff, x);
    socket << "parallel " << num_procs << " " << myid << "\n";
    socket.precision(8);

    socket << "solution\n" << *embed_mesh << x << std::flush
           << "window_title '" << prefix << " Level " << ilevel << "'" << std::endl;
    MPI_Barrier(embed_mesh->GetComm());
}

void EmbeddedPDESampler::glvis_plot(int ilevel,
        const Vector & coeff,
        const std::string prefix,
        std::ostream & socket) const
{
    GridFunction x;
    prolongate_to_fine_grid(ilevel, coeff, x);

    socket << "parallel " << num_procs << " " << myid << "\n";
    socket.precision(8);
    socket << "solution\n" << *orig_mesh << x << std::flush
           << "window_title '" << prefix << " Level " << ilevel << "'" << std::endl;
    MPI_Barrier(embed_mesh->GetComm());
}

void EmbeddedPDESampler::embed_prolongate_to_fine_grid(int ilevel, 
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

void EmbeddedPDESampler::embed_prolongate_to_fine_grid_u(int ilevel, 
        const Vector & coeff, 
        GridFunction & x) const
{
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        auto help = make_unique<Vector>(
            embed_sequence[lev-1]->ComputeTrueP(uform)->Height());
        (embed_sequence[lev-1]->ComputeTrueP(uform))->Mult(*feCoeff, *help);
        feCoeff = std::move(help);
    }

    x.MakeRef(fespace_u, *feCoeff, 0);

    if(ilevel!=0)
    {
        x.MakeDataOwner();
        feCoeff->StealData();
    }
}

void EmbeddedPDESampler::prolongate_to_fine_grid(int ilevel,
        const Vector & coeff,
        GridFunction & x) const
{
    // Prolongate to level 0 (fine)
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        auto help = make_unique<Vector>(Ps[lev-1]->Height() );
        Ps[lev-1]->Mult(*feCoeff, *help);
        feCoeff = std::move(help);
    }

    // Beware: wonkiness here!! now project to original mesh
    Vector * orig_feCoeff = new Vector(level_size[0]);
    feCoeff->SetSize(level_size[0]);
    meshP[0]->Mult(*feCoeff, *orig_feCoeff);
    feCoeff.reset(); 
    orig_feCoeff->SetSize(orig_level_size[0]);
    FiniteElementCollection * fec = new L2_FECollection(0, nDim);
    FiniteElementSpace * orig_fespace = new FiniteElementSpace(orig_mesh.get(),
            fec);
    
    x.MakeRef(orig_fespace, *orig_feCoeff, 0);
    
    {
        x.MakeDataOwner();
        x.MakeOwner(fec);
        orig_feCoeff->StealData();
    }
    
    delete orig_feCoeff;
}

} /* namespace parelagmc */
