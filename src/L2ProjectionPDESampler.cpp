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
 
#include "L2ProjectionPDESampler.hpp"
#include "transfer/ParMortarAssembler.hpp"
#include "Utilities.hpp"

namespace parelagmc 
{

using namespace mfem;
using namespace parelag;
using std::unique_ptr;

L2ProjectionPDESampler::L2ProjectionPDESampler(
            const std::shared_ptr<mfem::ParMesh>& orig_mesh_,                  
            const std::shared_ptr<mfem::ParMesh>& embed_mesh_,
            NormalDistributionSampler&  dist_sampler_,
            ParameterList& master_list_):
        orig_mesh(orig_mesh_),
        embed_mesh(embed_mesh_),
        dist_sampler(dist_sampler_),
        nDim(orig_mesh_->Dimension()),
        uform(nDim-1),                                                         
        sform(nDim),   
        nLevels(0),
        fespace(nullptr),
        orig_fespace(nullptr),
        master_list(master_list_),
        prob_list(master_list.Sublist("Problem parameters", true)),
        entire_seq(prob_list.Get("Build entire sequence", false)),
        save_vtk(prob_list.Get("Save VTK", false)),
        lognormal(prob_list.Get("Lognormal", true)),
        verbose(prob_list.Get("Verbosity", false)),
        corlen(prob_list.Get("Correlation length", 0.1)),
        alpha(1./(corlen*corlen)),
        matern_coeff(ComputeScalingCoefficientForSPDE(corlen, nDim)),
        if_solver_hybridization(false),
        compute_orig_sequence(true),
        level_size(0),
        orig_level_size(0),
        nnz(0),
        block_offsets(0,0),
        true_block_offsets(0,0),
        umap(0),
        pmap(0)
{
    MPI_Comm comm = orig_mesh->GetComm();
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                << "*  L2ProjectionPDESampler \n"
                << "*    Correlation length: " << corlen << "\n"
                << "*    Alpha: " << alpha << "\n"
                << "*    Matern Coefficient: " << matern_coeff << "\n";
        if (lognormal) std::cout << "*    Lognormal : true" << std::endl;;
        std::cout << std::string(50,'*') << '\n';
    }
}

void L2ProjectionPDESampler::SetDeRhamSequence(
        std::vector<std::shared_ptr<parelag::DeRhamSequence>> & orig_sequence_)
{
    orig_sequence = orig_sequence_;
    compute_orig_sequence = false;
}

void L2ProjectionPDESampler::BuildDeRhamSequence(
        std::vector< std::shared_ptr<AgglomeratedTopology> > & orig_topology,
        std::vector< std::shared_ptr<AgglomeratedTopology> > & topology)
{
    if (!myid && verbose) 
        std::cout << "-- L2ProjectionPDESampler::BuildDeRhamSequence" << std::endl;
    
    nLevels = topology.size();
    
    int feorder(0);
    int upscalingOrder(0);

    // Compute DeRhamSequence of enlarged mesh 
    sequence.resize(nLevels);
    {
        Timer timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Total");

        if(nDim == 2)
        {
            sequence[0] = std::make_shared<DeRhamSequence2D_Hdiv_FE>(
                    topology[0],
                    embed_mesh.get(), 
                    feorder);
        }
        else
        {
            sequence[0] = std::make_shared<DeRhamSequence3D_FE>(
                    topology[0], 
                    embed_mesh.get(), 
                    feorder);
        }
        
        DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence(); 
        PARELAG_ASSERT(DRSequence_FE);

        ConstantCoefficient coeffL2(1.);
        ConstantCoefficient coeffHdiv(1.);

        int jFormStart;
        if (entire_seq)
            jFormStart = 0;
        else
            jFormStart = nDim - 1;

        sequence[0]->SetjformStart(jFormStart);

        constexpr auto at_elem = AgglomeratedTopology::ELEMENT;

        DRSequence_FE->ReplaceMassIntegrator(
            at_elem, sform, make_unique<MassIntegrator>(coeffL2), false);
        DRSequence_FE->ReplaceMassIntegrator(
            at_elem, uform, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

        Array<Coefficient *> L2coeff;
        Array<VectorCoefficient *> Hdivcoeff;
        Array<VectorCoefficient *> Hcurlcoeff;
        Array<Coefficient *> H1coeff;
        fillVectorCoefficientArray(nDim, upscalingOrder, Hcurlcoeff);
        fillVectorCoefficientArray(nDim, upscalingOrder, Hdivcoeff);
        fillCoefficientArray(nDim, upscalingOrder, L2coeff);
        fillCoefficientArray(nDim, upscalingOrder+1, H1coeff);

        std::vector<unique_ptr<MultiVector>>
            targets(sequence[0]->GetNumberOfForms());
        int jform(0);

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

        // Set Targets 
        Array<MultiVector *> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();
        sequence[0]->SetTargets(targets_in);

    }
    
    if (!myid && verbose) 
        std::cout << "-- Coarsen the DeRham sequence of enlarged mesh" << std::endl;
    for(int i(0); i < nLevels-1; ++i)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Sampler: DeRhamSequence Construction -- Level ")
            .append(std::to_string(i+1)));
        Timer tot_timer = TimeManager::GetTimer(
            "Sampler: DeRhamSequence Construction -- Total");
        sequence[i+1] = sequence[i]->Coarsen();
    }
    
    // If necessary, compute DeRhamSequence of original (unstructured) mesh 
    if (compute_orig_sequence)
    {
        orig_sequence.resize(nLevels);
        Timer timer = TimeManager::GetTimer(
            "Sampler: DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::GetTimer(
            "Sampler: DeRhamSequence Construction -- Total");

        if(nDim == 2)
        {
            orig_sequence[0] = std::make_shared<DeRhamSequence2D_Hdiv_FE>(
                    orig_topology[0],
                    orig_mesh.get(),
                    feorder);
        }
        else
        {
            orig_sequence[0] = std::make_shared<DeRhamSequence3D_FE>(
                    orig_topology[0],
                    orig_mesh.get(),
                    feorder);
        }

        DeRhamSequenceFE * orig_DRSequence_FE = orig_sequence[0]->FemSequence();
        PARELAG_ASSERT(orig_DRSequence_FE);
       
        ConstantCoefficient coeffL2(1.);
        ConstantCoefficient coeffHdiv(1.);

        int jFormStart;
        if (entire_seq)
            jFormStart = 0;
        else
            jFormStart = nDim - 1;

        sequence[0]->SetjformStart(jFormStart);
        orig_sequence[0]->SetjformStart(jFormStart);

        constexpr auto at_elem = AgglomeratedTopology::ELEMENT;

        orig_DRSequence_FE->ReplaceMassIntegrator(
            at_elem, sform, make_unique<MassIntegrator>(coeffL2), false);
        orig_DRSequence_FE->ReplaceMassIntegrator(
            at_elem, uform, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

        Array<Coefficient *> L2coeff;
        Array<VectorCoefficient *> Hdivcoeff;
        Array<VectorCoefficient *> Hcurlcoeff;
        Array<Coefficient *> H1coeff;
        fillVectorCoefficientArray(nDim, upscalingOrder, Hcurlcoeff);
        fillVectorCoefficientArray(nDim, upscalingOrder, Hdivcoeff);
        fillCoefficientArray(nDim, upscalingOrder, L2coeff);
        fillCoefficientArray(nDim, upscalingOrder+1, H1coeff);

        std::vector<unique_ptr<MultiVector>>
            orig_targets(orig_sequence[0]->GetNumberOfForms());
        int jform(0);
        
        if (entire_seq)
        {
            orig_targets[jform] =
                orig_DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);
        }
        ++jform;
        if (entire_seq)
        {
            orig_targets[jform] =
                orig_DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
        }
        ++jform;

        if (nDim == 2)
        {
            orig_targets[jform] =
                orig_DRSequence_FE->InterpolateScalarTargets(jform,L2coeff);
            ++jform;
        }
        else
        {
            orig_targets[jform] =
                orig_DRSequence_FE->InterpolateVectorTargets(jform,Hdivcoeff);
            ++jform;
            orig_targets[jform] =
                orig_DRSequence_FE->InterpolateScalarTargets(jform,L2coeff);
            ++jform;
        }

        freeCoeffArray(L2coeff);
        freeCoeffArray(Hdivcoeff);
        freeCoeffArray(Hcurlcoeff);
        freeCoeffArray(H1coeff);

        // Set Targets 
        Array<MultiVector *> orig_targets_in(orig_targets.size());
        for (int ii = 0; ii < orig_targets_in.Size(); ++ii)
            orig_targets_in[ii] = orig_targets[ii].get();
        orig_sequence[0]->SetTargets(orig_targets_in);

        if (!myid && verbose) 
            std::cout << "-- Coarsen the DeRham sequence of original mesh" << std::endl;
        for(int i(0); i < nLevels-1; ++i)
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sampler: DeRhamSequence Construction -- Level ")
                .append(std::to_string(i+1)));
            Timer tot_timer = TimeManager::GetTimer(
                "Sampler: DeRhamSequence Construction -- Total");
            orig_sequence[i+1] = orig_sequence[i]->Coarsen();
        }
    }
} 

void L2ProjectionPDESampler::BuildHierarchy()
{
    // Check both deRham sequences have been built and set fespaces
    {
        DeRhamSequenceFE * DRSequence_FE = orig_sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);
        orig_fespace = DRSequence_FE->GetFeSpace(sform);
    }
    {
        DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);
        fespace = DRSequence_FE->GetFeSpace(sform);                            
    }

    block_offsets.SetSize( nLevels, 3 );
    true_block_offsets.SetSize( nLevels, 3 );
    
    umap.SetSize(nLevels);
    pmap.SetSize(nLevels);
    w_sqrt.resize(nLevels);
    orig_w_sqrt.resize(nLevels);

    A.resize(nLevels);
    invA.resize(nLevels);
    orig_level_size.SetSize(nLevels);
    level_size.SetSize(nLevels);
    nnz.SetSize(nLevels);

    Vector alpha_vec;
    
    // Boundary conditions 
    // Size=2 since 2 forms; ess_attr[1].Size() = 0 because no BC's
    // for that form
    std::vector<Array<int>> ess_bdr(2);
    const int bdr_size = embed_mesh->bdr_attributes.Size() ?
            embed_mesh->bdr_attributes.Max() : 0;
    ess_bdr[0].SetSize(bdr_size);
    ess_bdr[0] = 1;
    
    Ps.resize(nLevels-1);
    orig_Ps.resize(nLevels-1);

    for(int i(0); i < nLevels-1; ++i)
    {
        Ps[i] = sequence[i]->ComputeTrueP(sform);
        orig_Ps[i] = orig_sequence[i]->ComputeTrueP(sform);
    }

    for(int i(0); i < nLevels; ++i)
    {
        orig_level_size[i] = orig_sequence[i]->GetNumberOfDofs(sform);
        level_size[i] = sequence[i]->GetNumberOfDofs(sform);

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
            
            auto orig_W_s = orig_sequence[i]->ComputeMassOperator(sform);
             
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
            orig_w_sqrt[i] = make_unique<Vector>(orig_level_size[i]);
            orig_W_s->GetDiag(*(orig_w_sqrt[i]));
            {
                double * it = orig_w_sqrt[i]->GetData();
                for(double *end = it+orig_level_size[i]; it != end; ++it)
                    *it = sqrt(*it);
            }

            // Scale W_s by -kappa^2 
            alpha_vec.SetSize(W_s->Size());
            alpha_vec = -1.*alpha;

            W_s->ScaleRows(alpha_vec);

            pM = Assemble(*umap[i], *M_s, *umap[i]);
            pB = Assemble(*pmap[i], *B_s, *umap[i]);
            pBt = Assemble(*umap[i], *Bt_s, *pmap[i]);
            pW = Assemble(*pmap[i], *W_s, *pmap[i]);
            
            nnz[i] = pM->NNZ() + pB->NNZ() + pBt->NNZ() + pW->NNZ();
        }

        // Fill block_offsets
        block_offsets(i,0) = 0;
        true_block_offsets(i,0) = 0;
        block_offsets(i,1) = sequence[i]->GetNumberOfDofs(uform);
        block_offsets(i,2) = block_offsets(i,1) + sequence[i]->GetNumberOfDofs(sform);
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

        if (!myid && i==0)
        {
            std::cout << '\n' << std::string(50,'*') << '\n'
              << "*  L2 Projection SPDE Sampler "<< '\n'
              << "*    Solver: " << prec_type << '\n';
              std::cout << std::string(50,'*') << '\n' << std::endl;
        }
        
    } //nlevels 

    // L2 Prolongation Matrix G
    auto src_fe_coll  = std::make_shared<L2_FECollection>(0, embed_mesh->Dimension());
    auto src_fe       = std::make_shared<ParFiniteElementSpace>(embed_mesh.get(), src_fe_coll.get());

    auto dest_fe_coll = std::make_shared<L2_FECollection>(0, orig_mesh->Dimension());
    auto dest_fe      = std::make_shared<ParFiniteElementSpace>(orig_mesh.get(), dest_fe_coll.get());
    if (!myid && verbose) std::cout << "-- Build L2 Projector" << std::endl; 
    ParMortarAssembler assembler(embed_mesh->GetComm(), src_fe, dest_fe);
    assembler.AddMortarIntegrator(std::make_shared<L2MortarIntegrator>());
    Gt.resize(nLevels);
    {
        Timer genG = TimeManager::AddTimer("L2 Projector: Assemble -- Level 0");
        bool generateG = assembler.Assemble(Gt[0]);
        PARELAG_TEST_FOR_EXCEPTION(!generateG,
                std::runtime_error,
                "L2ProjectionPDESampler::ParMortarAssembler::Assemble "
                "No intersection no transfer!");
    }

    for (int ilevel = 0; ilevel < nLevels-1; ilevel++)
    {
        Timer genG = TimeManager::AddTimer(
            std::string("L2 Projector: Assemble -- Level ")
            .append(std::to_string(ilevel+1)));
        Gt[ilevel+1].reset( mfem::RAP(orig_Ps[ilevel].get(), 
                Gt[ilevel].get(), Ps[ilevel].get()));
    }

    // Create timer for L2-Proj mult, solver
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("L2 Projector: Apply -- Level ")
            .append(std::to_string(ilevel)));
        Timer mult_timer = TimeManager::AddTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(ilevel)));
    } 
}

void L2ProjectionPDESampler::Sample(const int level, Vector & xi)
{
    xi.SetSize( level_size[level] );
    dist_sampler(xi);
}

void L2ProjectionPDESampler::Eval(const int level, const Vector & xi, Vector & s)
{
    const int size_u = sequence[level]->GetNumberOfDofs(uform);

    int xi_level = level_size.Find( xi.Size() );

    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    // Compute orig_rhs_s = -g W^{1/2} xi
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
    mfem::BlockVector sol(offsets);
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

    // Project the solution to "original" mesh
    // G^T*s_bar
    s.SetSize(orig_level_size[level]);
    
    {
        Timer timer = TimeManager::GetTimer(
            std::string("L2 Projector: Apply -- Level ")
            .append(std::to_string(level)));
        Gt[level]->Mult(sol.GetBlock(1), s);
    }

    // Compute W^{-1}*G^T*s_bar
    {
        Vector & tmp(*(orig_w_sqrt[level]));
        for (int i = 0; i < s.Size(); i++)
            s(i) /= tmp(i)*tmp(i);
    }

    if(lognormal)
    {
        for(int idof = 0; idof < s.Size(); ++idof)
            s(idof) = exp( s(idof) );
    }
    
}

void L2ProjectionPDESampler::Eval(const int level, const Vector & xi,
        Vector & s, Vector & embed_s, bool use_init)
{
    const int size_u = sequence[level]->GetNumberOfDofs(uform);
    int xi_level = level_size.Find( xi.Size() );

    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    // Compute orig_rhs_s = -g W^{1/2} xi
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
    
    mfem::BlockVector sol(offsets);
    
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
    
    // Project the solution to "original" mesh
    // G^T*s_bar
    embed_s.SetSize(level_size[level]);
    s.SetSize(orig_level_size[level]);
    embed_s = sol.GetBlock(1);
    
    {
        Timer timer = TimeManager::GetTimer(
            std::string("L2 Projector: Apply -- Level ")
            .append(std::to_string(level)));
        Gt[level]->Mult(sol.GetBlock(1), s);
    }
    // Compute W^{-1}*G^T*s_bar
    {

        Vector & tmp(*(orig_w_sqrt[level]));
        for (int i = 0; i < s.Size(); i++)
            s(i) /= tmp(i)*tmp(i);
    }

    if(lognormal)
    {
        for(int idof = 0; idof < s.Size(); ++idof)
            s(idof) = exp( s(idof) );
    }
}

void L2ProjectionPDESampler::EmbedEval(const int level, const Vector & xi, Vector & s)
{
    const int size_u = sequence[level]->GetNumberOfDofs(uform);

    int xi_level = level_size.Find( xi.Size() );

    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    // Compute orig_rhs_s = -g W^{1/2} xi
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
    mfem::BlockVector sol(offsets);
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
        for(int idof = 0; idof < s.Size(); ++idof)
            s(idof) = exp( s(idof) );
    }
    
}
double L2ProjectionPDESampler::EmbedComputeL2Error(
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

double L2ProjectionPDESampler::ComputeL2Error(
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

double L2ProjectionPDESampler::EmbedComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    return std::max(coeff.Max() - exact,
            std::abs(exact - coeff.Min()));
    
}

double L2ProjectionPDESampler::ComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    const double e1 = coeff.Max() - exact;
    const double e2 = exact - coeff.Min();   
    return std::max(e1, e2); 

}

void L2ProjectionPDESampler::EmbedSaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     embed_mesh->Print(mesh_ofs);
    if (save_vtk)
    {
        // Printing to VTK file format
        mesh_name << ".vtk";
        std::ofstream vid(mesh_name.str().c_str());
        vid.precision(8);
        embed_mesh->PrintVTK(vid);
    }
}

void L2ProjectionPDESampler::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     orig_mesh->Print(mesh_ofs);
    if (save_vtk)
    {
        // Printing to VTK file format
        mesh_name << ".vtk";
        std::ofstream vid(mesh_name.str().c_str());
        vid.precision(8);
        orig_mesh->PrintVTK(vid);
    }
}

void L2ProjectionPDESampler::EmbedSaveFieldGLVis(int level, 
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

void L2ProjectionPDESampler::SaveFieldGLVis(int level,
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

void L2ProjectionPDESampler::SaveFieldGLVis_H1(int level,
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
        orig_mesh->PrintVTK(vid);
        x.SaveVTK(vid, "value", 0);
    }
}

void L2ProjectionPDESampler::SaveFieldGLVis_H1Add(int level,
        const Vector & coeff,
        const std::string prefix,
        const Vector & mu) const
{
    FiniteElementCollection * fecH1 = new H1_FECollection(1,nDim);
    FiniteElementSpace * fespaceH1 = new FiniteElementSpace(orig_mesh.get(), fecH1);

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
        orig_mesh->PrintVTK(vid);
        x.SaveVTK(vid, "value", 0);
    }
}

void L2ProjectionPDESampler::Transfer(
        int level,
        const mfem::Vector & coeff,
        mfem::Vector & orig_coeff) const
{
    // Project coeff to "original" mesh on a particular level
    orig_coeff.SetSize(orig_level_size[level]);
    Gt[level]->Mult(coeff, orig_coeff);

    // Compute W^{-1}*G^T*s_bar 
    {
        Vector & tmp(*(orig_w_sqrt[level]));
        for (int i = 0; i < orig_coeff.Size(); i++)
            orig_coeff(i) /= tmp(i)*tmp(i);
    }
}

void L2ProjectionPDESampler::embed_glvis_plot(int ilevel, 
        const Vector & coeff, 
        const std::string prefix,
        std::ostream & socket) const
{
    GridFunction x;
    embed_prolongate_to_fine_grid(ilevel, coeff, x);
    socket << "parallel " << num_procs << " " << myid << "\n";
    socket.precision(8);

    socket << "solution\n" << *embed_mesh<< x << std::flush
           << "window_title '" << prefix << " Level " << ilevel << "'" << std::endl;
    MPI_Barrier(embed_mesh->GetComm());
}

void L2ProjectionPDESampler::glvis_plot(int ilevel,
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
    MPI_Barrier(orig_mesh->GetComm());
}

void L2ProjectionPDESampler::embed_prolongate_to_fine_grid(int ilevel, 
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

void L2ProjectionPDESampler::prolongate_to_fine_grid(int ilevel,
        const Vector & coeff,
        GridFunction & x) const
{
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        auto help = make_unique<Vector>(orig_Ps[lev-1]->Height() );
        orig_Ps[lev-1]->Mult(*feCoeff, *help);
        feCoeff = std::move(help);
    }

    x.MakeRef(orig_fespace, *feCoeff, 0);

    if(ilevel!=0)
    {
        x.MakeDataOwner();
        feCoeff->StealData();
    }
}

void L2ProjectionPDESampler::prolongate_to_fine_grid_H1(int ilevel,
        const Vector & coeff,
        GridFunction & x) const
{
    FiniteElementCollection * fecH1 = new H1_FECollection(1,nDim);
    FiniteElementSpace * fespaceH1 = new FiniteElementSpace(orig_mesh.get(), fecH1);

    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        auto help = make_unique<Vector>(orig_Ps[lev-1]->Height() );
        orig_Ps[lev-1]->Mult(*feCoeff, *help);
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
