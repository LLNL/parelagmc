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

#include "EmbeddedPDESampler_Legacy.hpp"
#include "Utilities.hpp"

namespace parelagmc 
{
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

EmbeddedPDESampler_Legacy::EmbeddedPDESampler_Legacy( 
            const std::shared_ptr<mfem::ParMesh>& orig_mesh_,                  
            const std::shared_ptr<mfem::ParMesh>& embed_mesh_,
            NormalDistributionSampler& dist_sampler_,
            std::vector<Array<int>> & materialId_,
            ParameterList& master_list_):
        orig_mesh(orig_mesh_),
        embed_mesh(embed_mesh_),
        nDim(orig_mesh->Dimension()),
        uform(nDim-1),                                                         
        sform(nDim),                                                           
        dist_sampler(dist_sampler_),
        nLevels(materialId_.size()),
        fespace(nullptr),
        fespace_u(nullptr),
        iter(-1),
        prob_list(master_list_.Sublist("Problem parameters", true)),            
        save_vtk(prob_list.Get("Save VTK", false)),                            
        lognormal(prob_list.Get("Lognormal", true)),                           
        verbose(prob_list.Get("Verbosity", false)),                            
        corlen(prob_list.Get("Correlation length", 0.1)),                      
        alpha(1./(corlen*corlen)), 
        matern_coeff(ComputeScalingCoefficientForSPDE(corlen, nDim)),
        level_size(0),
        orig_level_size(0),
        nnz(0)
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
        // but I coundn't make that work so now we have a block of zeros.
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

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                << "*  EmbeddedPDESampler_Legacy \n"
                << "*    Correlation length: " << corlen << "\n"
                << "*    Alpha: " << alpha << "\n"
                << "*    Matern Coefficient: " << matern_coeff << "\n";
        if (lognormal) std::cout << "*    Lognormal : true" << std::endl;
        std::cout << std::string(50,'*') << '\n';
    }

}

void EmbeddedPDESampler_Legacy::BuildDeRhamSequence(
        std::vector< std::shared_ptr<AgglomeratedTopology> > & embed_topology)
{
    int feorder(0);
    int orderupscaling(0);
    embed_sequence.resize(nLevels);
    nnz.SetSize(nLevels);
    {
        Timer timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "Sampler: DeRhamSequence Construction -- Total");

        if(nDim == 2)
        {
            embed_sequence[0] = std::make_shared<DeRhamSequence2D_Hdiv_FE>(
                    embed_topology[0], 
                    embed_mesh.get(), 
                    feorder);
        }
        else
        {
            embed_sequence[0] = std::make_shared<DeRhamSequence3D_FE>(
                    embed_topology[0], 
                    embed_mesh.get(), 
                    feorder);
        }
        DeRhamSequenceFE * DRSequence_FE = embed_sequence[0]->FemSequence(); 
        PARELAG_ASSERT(DRSequence_FE);

        std::vector<std::unique_ptr<MultiVector>>
            targets(embed_sequence[0]->GetNumberOfForms());

        Array<Coefficient *> scalarCoeffL2;
        Array<VectorCoefficient *> vectCoeffHdiv;

        fillVectorCoefficientArray(nDim, orderupscaling, vectCoeffHdiv);
        fillCoefficientArray(nDim, orderupscaling, scalarCoeffL2);

        for (int jform = 0; jform<embed_sequence[0]->GetNumberOfForms(); ++jform)
            targets[jform] = nullptr;

        targets[uform] = DRSequence_FE->InterpolateVectorTargets(uform, 
                vectCoeffHdiv);
        targets[sform] = DRSequence_FE->InterpolateScalarTargets(sform, 
                scalarCoeffL2);

        freeCoeffArray(vectCoeffHdiv);
        freeCoeffArray(scalarCoeffL2);

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

void EmbeddedPDESampler_Legacy::BuildHierarchy()
{
    // Check deRham sequence has been built and get fespaces
    {
        DeRhamSequenceFE * DRSequence_FE = embed_sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);
        fespace = DRSequence_FE->GetFeSpace(sform);
        fespace_u = DRSequence_FE->GetFeSpace(uform);
    }

    Array<int> ess_bdr(embed_mesh->bdr_attributes.Size() ? 
            embed_mesh->bdr_attributes.Max() : 0);
    ess_bdr = 1;
    Ps.resize(nLevels-1);
    for(int i(0); i < nLevels-1; ++i)
        Ps[i] = embed_sequence[i]->ComputeTrueP(sform);
    
    W.resize(nLevels);
    w_sqrt.resize(nLevels);
    D.resize(nLevels);
    A.resize(nLevels);
    precA.resize(nLevels);
    invA.resize(nLevels);
    nnz.SetSize(nLevels);
    meshP.resize(nLevels);

    for(int i(0); i < nLevels; ++i)
    {
        SparseMatrix * M_s = embed_sequence[i]->ComputeMassOperator(uform).release();
        SparseMatrix * W_s = embed_sequence[i]->ComputeMassOperator(sform).release();
        SparseMatrix * D_s = embed_sequence[i]->GetDerivativeOperator(uform);
        unique_ptr<SparseMatrix> DtWD{RAP(*D_s, *W_s, *D_s)};
        unique_ptr<SparseMatrix> A_s{Add(1., *M_s, 1./alpha, *DtWD)};
        DtWD.reset();
        delete M_s;

        // Get sqrt of digonal of W
        w_sqrt[i] = make_unique<Vector>(level_size[i]);
        W_s->GetDiag(*(w_sqrt[i]));

        {
            double * it = w_sqrt[i]->GetData();
            for(double *end = it+level_size[i]; it != end; ++it)
                *it = sqrt(*it);
        }
        SharingMap & umap = embed_sequence[i]->GetDofHandler(uform)->GetDofTrueDof();
        SharingMap & pmap = embed_sequence[i]->GetDofHandler(sform)->GetDofTrueDof();

        Array<int> ess_dof(embed_sequence[i]->GetNumberOfDofs(uform));
        embed_sequence[i]->GetDofHandler(uform)->MarkDofsOnSelectedBndr(
                ess_bdr, ess_dof);

        for(int irow = 0; irow < ess_dof.Size(); ++irow)
            if(ess_dof[irow])
                A_s->EliminateRowCol(irow);

        D_s->EliminateCols(ess_dof);

        A[i] = Assemble(umap, *A_s, umap);
        W[i] = Assemble(pmap, *W_s, pmap);
        D[i] = Assemble(pmap, *D_s, umap);
        //FIXME: This is nnz of Schur complement 
        nnz[i] = A[i]->NNZ();
        delete W_s;
        //delete D_s;
        A_s.reset();
        
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sampler: Build Solver -- Level ")
                .append(std::to_string(i)));

            if(nDim==3)
                precA[i] = make_unique<HypreExtension::HypreADS>(
                        *A[i], 
                        embed_sequence[i].get(), 
                        adsData);
            else
                precA[i] = make_unique<HypreExtension::HypreAMS>(
                        *A[i], 
                        embed_sequence[i].get(), 
                        amsData);
        }
        invA[i] = make_unique<CGSolver>(A[i]->GetComm());
        invA[i]->SetPrintLevel( 0 );
        invA[i]->SetMaxIter( 1000 );
        invA[i]->SetRelTol( 1e-6 );
        invA[i]->SetAbsTol( 1e-12 );
        invA[i]->SetOperator(*A[i]);
        invA[i]->SetPreconditioner(*precA[i]);
        
        // Now that we have the maps, Assemble the mesh prolongators where
        // meshP_s is created in the constructor.    
        meshP[i] = Assemble(pmap, *meshP_s[i], pmap);
        meshP_s[i].reset(); 
    } //nlevels 

    // Create timer for solver
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(ilevel)));
    }
}

void EmbeddedPDESampler_Legacy::Sample(const int level, Vector & xi)
{
    xi.SetSize( level_size[level] );
    dist_sampler(xi);
}

void EmbeddedPDESampler_Legacy::Eval(const int level, const Vector & xi, Vector & s)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();
    
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );
    
    Vector embed_s( size_s);
    embed_s = 0.0;
   
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
            msg << "Level " << level << " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
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
            embed_s(idof) = alphainv*div_u(idof) +
                    matern_alphainv * (*rhs_s)(idof) / (tmp(idof)*tmp(idof));
    }
    
    // Project to original mesh
    s.SetSize(level_size[level]);
    meshP[level]->Mult(embed_s, s);
    // Silliness to remove zeros from end of s after mult by meshP
    s.SetSize(orig_level_size[level]);
    if(lognormal)
    {
        for(int idof = 0; idof < orig_level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
    }
}

void EmbeddedPDESampler_Legacy::Eval(const int level, const Vector & xi, Vector & s,
        Vector & embed_s, bool use_init)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();
    
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    PARELAG_ASSERT( level_size[xi_level] > 0 );
    
    embed_s.SetSize( size_s);
    embed_s = 0.0;
   
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
            msg << "Level " << level << " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
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
            embed_s(idof) = alphainv*div_u(idof) +
                    matern_alphainv * (*rhs_s)(idof) / (tmp(idof)*tmp(idof));
    }
    
    // Project to original mesh
    s.SetSize(level_size[level]);
    meshP[level]->Mult(embed_s, s);
    // Silliness to remove zeros from end of s after mult by meshP
    s.SetSize(orig_level_size[level]);
    if(lognormal)
    {
        for(int idof = 0; idof < orig_level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
    }
}
void EmbeddedPDESampler_Legacy::EmbedEval(const int level, const Vector & xi, Vector & s, Vector & u)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();
    s.SetSize( size_s );
    s = 0.;
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

    u.SetSize( size_u );
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
            msg << "Level " << level << " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
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
        for(int idof = 0; idof < u.Size(); ++idof)
            u(idof) = exp( u(idof) );
    }
}
void EmbeddedPDESampler_Legacy::EmbedEval(const int level, const Vector & xi, Vector & s)
{
    int size_s = level_size[level];
    int size_u = A[level]->Height();
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
    iter = invA[level]->GetNumIterations();

    if (verbose)
    {
        std::stringstream msg;
        if(invA[level]->GetConverged())
            msg << "Level " << level << " Converged in " << invA[level]->GetNumIterations() 
                << " iters with a residual norm " << invA[level]->GetFinalNorm() << "\n";
        else
            msg << "Level " << level << " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";

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

double EmbeddedPDESampler_Legacy::EmbedComputeL2Error(
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

double EmbeddedPDESampler_Legacy::ComputeL2Error(
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

double EmbeddedPDESampler_Legacy::EmbedComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    return std::max(coeff.Max() - exact,
            std::abs(exact - coeff.Min()));
    
}

double EmbeddedPDESampler_Legacy::ComputeMaxError(
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

void EmbeddedPDESampler_Legacy::EmbedSaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     embed_mesh->Print(mesh_ofs);
}

void EmbeddedPDESampler_Legacy::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     orig_mesh->Print(mesh_ofs);
}

void EmbeddedPDESampler_Legacy::EmbedSaveFieldGLVis(int level, 
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

void EmbeddedPDESampler_Legacy::EmbedSaveFieldGLVis_u(int level, 
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

void EmbeddedPDESampler_Legacy::SaveFieldGLVis(int level,
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

void EmbeddedPDESampler_Legacy::Transfer(
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

void EmbeddedPDESampler_Legacy::embed_glvis_plot(int ilevel, 
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

void EmbeddedPDESampler_Legacy::glvis_plot(int ilevel,
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

void EmbeddedPDESampler_Legacy::embed_prolongate_to_fine_grid(int ilevel, 
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

void EmbeddedPDESampler_Legacy::embed_prolongate_to_fine_grid_u(int ilevel, 
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

void EmbeddedPDESampler_Legacy::prolongate_to_fine_grid(int ilevel,
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
