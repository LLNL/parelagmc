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

#include "L2ProjectionPDESampler_Legacy.hpp"
#include "transfer/ParMortarAssembler.hpp"
#include "Utilities.hpp"

namespace parelagmc 
{
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

L2ProjectionPDESampler_Legacy::L2ProjectionPDESampler_Legacy(
            const std::shared_ptr<mfem::ParMesh>& orig_mesh_,                  
            const std::shared_ptr<mfem::ParMesh>& embed_mesh_,
            NormalDistributionSampler& dist_sampler_,
            ParameterList& master_list_):
        orig_mesh(orig_mesh_),
        embed_mesh(embed_mesh_),
        fespace(nullptr),
        orig_fespace(nullptr),
        nDim(orig_mesh->Dimension()),
        uform(nDim-1),                                                         
        sform(nDim),
        dist_sampler(dist_sampler_),
        nLevels(0),
        prob_list(master_list_.Sublist("Problem parameters", true)),
        save_vtk(prob_list.Get("Save VTK", false)),                            
        lognormal(prob_list.Get("Lognormal", true)),                           
        verbose(prob_list.Get("Verbosity", false)),                            
        corlen(prob_list.Get("Correlation length", 0.1)),                      
        alpha(1./(corlen*corlen)),                                             
        matern_coeff(ComputeScalingCoefficientForSPDE(corlen, nDim)),
        iter(-1),
        compute_orig_sequence(true),
        level_size(0),
        orig_level_size(0),
        nnz(0)
{
    MPI_Comm comm = orig_mesh->GetComm();
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    
    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                << "*  L2ProjectionPDESampler_Legacy \n"
                << "*    Correlation length: " << corlen << "\n"
                << "*    Alpha: " << alpha << "\n"
                << "*    Matern Coefficient: " << matern_coeff << "\n";
        if (lognormal) std::cout << "*    Lognormal : true" << std::endl;
        std::cout << std::string(50,'*') << '\n';
    }
}

void L2ProjectionPDESampler_Legacy::SetDeRhamSequence(
        std::vector<std::shared_ptr<parelag::DeRhamSequence>> & orig_sequence_)
{
    orig_sequence = orig_sequence_;
    compute_orig_sequence = false;
}

void L2ProjectionPDESampler_Legacy::BuildDeRhamSequence(
        std::vector< std::shared_ptr<AgglomeratedTopology> > & orig_topology,
        std::vector< std::shared_ptr<AgglomeratedTopology> > & topology)
{
    if (!myid && verbose) 
        std::cout << "-- L2ProjectionPDESampler_Legacy::BuildDeRhamSequence" << std::endl;

    nLevels = topology.size();


    int feorder(0);
    int orderupscaling(0);
    sequence.resize(nLevels);
    {
        Timer timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Total");

        if (nDim == 2)
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

        std::vector<std::unique_ptr<MultiVector>>
            targets(sequence[0]->GetNumberOfForms());

        Array<Coefficient *> scalarCoeffL2;
        Array<VectorCoefficient *> vectCoeffHdiv;

        fillVectorCoefficientArray(nDim, orderupscaling, vectCoeffHdiv);
        fillCoefficientArray(nDim, orderupscaling, scalarCoeffL2);

        for (int jform = 0; jform<sequence[0]->GetNumberOfForms(); ++jform)
            targets[jform] = nullptr;

        targets[uform] = DRSequence_FE->InterpolateVectorTargets(uform, 
                vectCoeffHdiv);
        targets[sform] = DRSequence_FE->InterpolateScalarTargets(sform, 
                scalarCoeffL2);

        Array<MultiVector *> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();
        sequence[0]->SetTargets(targets_in);
        
        freeCoeffArray(vectCoeffHdiv);
        freeCoeffArray(scalarCoeffL2);

    }

    // Coarsen the DeRham sequence of enlarged mesh
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
        {
            Timer timer = TimeManager::GetTimer(
                "Sampler: DeRhamSequence Construction -- Level 0");
            Timer tot_timer = TimeManager::GetTimer(
                "Sampler: DeRhamSequence Construction -- Total");

            if (nDim == 2)
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
            
            Array<Coefficient *> scalarCoeffL2;
            Array<VectorCoefficient *> vectCoeffHdiv;

            fillVectorCoefficientArray(nDim, orderupscaling, vectCoeffHdiv);
            fillCoefficientArray(nDim, orderupscaling, scalarCoeffL2);

            std::vector<std::unique_ptr<MultiVector>>
                orig_targets(orig_sequence[0]->GetNumberOfForms());

            for (int jform = 0; jform<orig_sequence[0]->GetNumberOfForms(); ++jform)
                orig_targets[jform] = nullptr;

            orig_targets[uform] = orig_DRSequence_FE->InterpolateVectorTargets(uform, 
                    vectCoeffHdiv);
            orig_targets[sform] = orig_DRSequence_FE->InterpolateScalarTargets(sform, 
                    scalarCoeffL2);

            Array<MultiVector *> orig_targets_in(orig_targets.size());
            for (int ii = 0; ii < orig_targets_in.Size(); ++ii)
                orig_targets_in[ii] = orig_targets[ii].get();
            orig_sequence[0]->SetTargets(orig_targets_in);
        
            freeCoeffArray(vectCoeffHdiv);
            freeCoeffArray(scalarCoeffL2);

        }

        // Coarsen the DeRham sequence of original mesh
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
   
void L2ProjectionPDESampler_Legacy::BuildHierarchy()
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

    Array<int> ess_bdr(embed_mesh->bdr_attributes.Size() ? 
            embed_mesh->bdr_attributes.Max() : 0);
    ess_bdr = 1;
    
    Ps.resize(nLevels-1);
    orig_Ps.resize(nLevels-1);
    for(int i(0); i < nLevels-1; ++i)
    {
        Ps[i] = sequence[i]->ComputeTrueP(sform);
        orig_Ps[i] = orig_sequence[i]->ComputeTrueP(sform);
    }

    W.resize(nLevels);
    w_sqrt.resize(nLevels);
    orig_w_sqrt.resize(nLevels);

    D.resize(nLevels);
    A.resize(nLevels);
    precA.resize(nLevels);
    invA.resize(nLevels);
    nnz.SetSize(nLevels);
    orig_level_size.SetSize(nLevels);
    level_size.SetSize(nLevels);

    for(int i(0); i < nLevels; ++i)
    {
        orig_level_size[i] = orig_sequence[i]->GetNumberOfDofs(sform);
        level_size[i] = sequence[i]->GetNumberOfDofs(sform);
        auto M_s = sequence[i]->ComputeMassOperator(uform);
        auto W_s = sequence[i]->ComputeMassOperator(sform);
        auto orig_W_s = orig_sequence[i]->ComputeMassOperator(sform);
        
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

        auto D_s = sequence[i]->GetDerivativeOperator(uform);
        unique_ptr<SparseMatrix> DtWD{RAP(*D_s, *W_s, *D_s)};
        unique_ptr<SparseMatrix> A_s{Add(1., *M_s, 1./alpha, *DtWD)};
        DtWD.reset();
        M_s.reset();

        SharingMap & umap = sequence[i]->GetDofHandler(uform)->GetDofTrueDof();
        SharingMap & pmap = sequence[i]->GetDofHandler(sform)->GetDofTrueDof();
        
        Array<int> ess_dof(sequence[i]->GetNumberOfDofs(uform));
        sequence[i]->GetDofHandler(uform)->MarkDofsOnSelectedBndr(
                ess_bdr, ess_dof);

        for(int irow = 0; irow < ess_dof.Size(); ++irow)
            if(ess_dof[irow])
                A_s->EliminateRowCol(irow);

        D_s->EliminateCols(ess_dof);

        A[i] = Assemble(umap, *A_s, umap);
        W[i] = Assemble(pmap, *W_s, pmap);
        D[i] = Assemble(pmap, *D_s, umap);

        nnz[i] = 0;
        W_s.reset();
        A_s.reset();

        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sampler: Build Legacy Prec -- Level ")
                .append(std::to_string(i)));
            
            if(nDim==3)
                precA[i] = make_unique<HypreExtension::HypreADS>(
                        *A[i],
                        sequence[i].get(),
                        adsData);
            else
                precA[i] = make_unique<HypreExtension::HypreAMS>(
                        *A[i],
                        sequence[i].get(),
                        amsData);
        }
        invA[i] = make_unique<CGSolver>(A[i]->GetComm());
        invA[i]->SetPrintLevel( 0 );
        invA[i]->SetMaxIter( 1000 );
        invA[i]->SetRelTol( 1e-6 );
        invA[i]->SetAbsTol( 1e-12 );
        invA[i]->SetOperator(*A[i]);
        invA[i]->SetPreconditioner(*precA[i]);
        
    } //nlevels 

    // L2 Prolongation Matrix G
    auto src_fe_coll  = std::make_shared<L2_FECollection>(0, embed_mesh->Dimension());
    auto src_fe       = std::make_shared<ParFiniteElementSpace>(embed_mesh.get(), src_fe_coll.get());

    auto dest_fe_coll = std::make_shared<L2_FECollection>(0, orig_mesh->Dimension());
    auto dest_fe      = std::make_shared<ParFiniteElementSpace>(orig_mesh.get(), dest_fe_coll.get());

    ParMortarAssembler assembler(embed_mesh->GetComm(), src_fe, dest_fe);
    assembler.AddMortarIntegrator(std::make_shared<L2MortarIntegrator>());
    Gt.resize(nLevels);
    {
        Timer genG = TimeManager::AddTimer("L2 Projector Assemble -- Level 0");
        bool generateG = assembler.Assemble(Gt[0]);
        PARELAG_TEST_FOR_EXCEPTION(!generateG,
                std::runtime_error,
                "L2ProjectionPDESampler_Legacy::ParMortarAssembler::Assemble "
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

void L2ProjectionPDESampler_Legacy::Sample(const int level, Vector & xi)
{
    xi.SetSize( level_size[level] );
    dist_sampler(xi);
}

void L2ProjectionPDESampler_Legacy::Eval(const int level, const Vector & xi, Vector & s)
{
    const int size_u = A[level]->Height();
    int xi_level = level_size.Find( xi.Size() );
    Vector embed_s( level_size[level] );
    embed_s = 0.0;
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    // Compute orig_rhs_s = -g W^{1/2} xi
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = xi(idof) * tmp(idof);
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

    Vector u(size_u);
    u = 0.0;
    D[level]->MultTranspose(*rhs_s, *rhs_u);
    *rhs_u *= -matern_coeff/alpha;
    {
        Timer timer = TimeManager::GetTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(level)));
        invA[level]->Mult(*rhs_u, u);
    }

    iter = invA[level]->GetNumIterations();
    
    if (verbose)
    {
        std::stringstream msg;
        if(invA[level]->GetConverged())
            msg << "Level " << level << " Converged in " << invA[level]->GetNumIterations() 
                << " iters with a residual norm " << invA[level]->GetFinalNorm() << "\n";
        else
            msg << "Level " << level <<  " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
        if(myid == 0)
          std::cout << msg.str();
    }
    PARELAG_ASSERT(invA[level]->GetConverged());
    
    Vector div_u( level_size[level] );
    D[level]->Mult(u, div_u);
    {
        Vector & tmp(*(w_sqrt[level]));
        double alphainv = 1./alpha;
        double matern_alphainv = matern_coeff*alphainv;
        for(int idof = 0; idof < level_size[level]; ++idof)
            embed_s(idof) = alphainv*div_u(idof) +
                    matern_alphainv * (*rhs_s)(idof) / (tmp(idof)*tmp(idof));
    }
    
    // Project the solution to "original" mesh
    // G^T*s_bar 
    s.SetSize(orig_level_size[level]);
    {
        Timer timer = TimeManager::GetTimer(
            std::string("L2 Projector: Apply -- Level ")
            .append(std::to_string(level)));
        Gt[level]->Mult(embed_s, s);
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

void L2ProjectionPDESampler_Legacy::Eval(const int level, const Vector & xi, Vector & s,
        Vector & embed_s, bool use_init)
{
    const int size_u = A[level]->Height();
    int xi_level = level_size.Find( xi.Size() );
    
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = xi(idof) * tmp(idof);
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

    Vector u(size_u);
    u = 0.0;
    D[level]->MultTranspose(*rhs_s, *rhs_u);
    *rhs_u *= -matern_coeff/alpha;
    {
        Timer timer = TimeManager::GetTimer(
                std::string("Sampler: Mult -- Level ")
                .append(std::to_string(level)));
        invA[level]->Mult(*rhs_u, u);
    }
    iter = invA[level]->GetNumIterations();
    
    if (verbose)
    {
        std::stringstream msg;
        if(invA[level]->GetConverged())
            msg << "Level " << level << " Converged in " << invA[level]->GetNumIterations() 
                << " iters with a residual norm " << invA[level]->GetFinalNorm() << "\n";
        else
            msg << "Level " << level <<  " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
        if(myid == 0)
          std::cout << msg.str();
    }
    PARELAG_ASSERT(invA[level]->GetConverged());
    
    embed_s.SetSize(level_size[level]); 
    Vector div_u( level_size[level] );
    D[level]->Mult(u, div_u);
    {
        Vector & tmp(*(w_sqrt[level]));
        double alphainv = 1./alpha;
        double matern_alphainv = matern_coeff*alphainv;
        for(int idof = 0; idof < level_size[level]; ++idof)
            embed_s(idof) = alphainv*div_u(idof) +
                    matern_alphainv * (*rhs_s)(idof) / (tmp(idof)*tmp(idof));
    }
    
    // Project the solution to "original" mesh
    // G^T*s_bar 
    s.SetSize(orig_level_size[level]);
    {
        Timer timer = TimeManager::GetTimer(
            std::string("L2 Projector: Apply -- Level ")
            .append(std::to_string(level)));
        Gt[level]->Mult(embed_s, s);
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

void L2ProjectionPDESampler_Legacy::EmbedEval(const int level, const Vector & xi, Vector & s)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();
    s.SetSize( size_s );
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );

    // Compute rhs_s = -g W^{1/2} xi 
    auto rhs_s = make_unique<Vector>( level_size[xi_level] );
    {
        Vector & tmp(*(w_sqrt[xi_level]));
        for(int idof = 0; idof < level_size[xi_level]; ++idof)
            (*rhs_s)[idof] = xi(idof) * tmp(idof);
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

    Vector u(size_u);
    u = 0.0;
    D[level]->MultTranspose(*rhs_s, *rhs_u);
    *rhs_u *= -matern_coeff/alpha;
    
    {
        Timer timer = TimeManager::GetTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(level)));
        invA[level]->Mult(*rhs_u, u);
    }

    if (verbose)
    {
        std::stringstream msg;
        if(invA[level]->GetConverged())
            msg << "Level " << level << " Converged in " << invA[level]->GetNumIterations() << 
                    " iters with a residual norm " << invA[level]->GetFinalNorm() << "\n";
        else
            msg << "Level " << level <<  " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
        if(myid == 0)
            std::cout << msg.str();
    }
    
    PARELAG_ASSERT(invA[level]->GetConverged());

    Vector div_u( size_s );
    D[level]->Mult(u, div_u);
    {
        Vector & tmp(*(w_sqrt[level]));
        double alphainv = 1./alpha;
        double matern_alphainv = matern_coeff*alphainv;
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = alphainv*div_u(idof) + 
                    matern_alphainv * (*rhs_s)(idof) / (tmp(idof)*tmp(idof));
    }

    if(lognormal)
    {
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
    }
}

double L2ProjectionPDESampler_Legacy::EmbedComputeL2Error(
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

double L2ProjectionPDESampler_Legacy::ComputeL2Error(
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

double L2ProjectionPDESampler_Legacy::EmbedComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    return std::max(coeff.Max() - exact,
            std::abs(exact - coeff.Min()));
    
}

double L2ProjectionPDESampler_Legacy::ComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    const double e1 = coeff.Max() - exact;
    const double e2 = exact - coeff.Min();   
    return std::max(e1, e2); 

}

void L2ProjectionPDESampler_Legacy::EmbedSaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     embed_mesh->Print(mesh_ofs);
}

void L2ProjectionPDESampler_Legacy::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     orig_mesh->Print(mesh_ofs);
}

void L2ProjectionPDESampler_Legacy::EmbedSaveFieldGLVis(int level, 
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

void L2ProjectionPDESampler_Legacy::SaveFieldGLVis(int level,
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

void L2ProjectionPDESampler_Legacy::Transfer(
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

void L2ProjectionPDESampler_Legacy::embed_glvis_plot(int ilevel, 
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

void L2ProjectionPDESampler_Legacy::glvis_plot(int ilevel,
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

void L2ProjectionPDESampler_Legacy::embed_prolongate_to_fine_grid(int ilevel, 
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

void L2ProjectionPDESampler_Legacy::prolongate_to_fine_grid(int ilevel,
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

} /* namespace parelagmc */
