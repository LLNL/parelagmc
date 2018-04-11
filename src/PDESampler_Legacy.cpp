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
 
#include "PDESampler_Legacy.hpp"

#include <cmath>
#include "Utilities.hpp"

namespace parelagmc 
{

using namespace mfem;
using namespace parelag;
using std::unique_ptr;

PDESampler_Legacy::PDESampler_Legacy(
            const std::shared_ptr<mfem::ParMesh>& mesh_, 
            NormalDistributionSampler & dist_sampler_,
            ParameterList& master_list_):
        mesh(mesh_),
        nDim(mesh->Dimension()),
        uform(nDim-1),
        sform(nDim), 
        dist_sampler(dist_sampler_),
        nLevels(0),
        fespace(nullptr),
        fespace_u(nullptr),
        prob_list(master_list_.Sublist("Problem parameters", true)),            
        save_vtk(prob_list.Get("Save VTK", false)),                            
        lognormal(prob_list.Get("Lognormal", true)),                           
        verbose(prob_list.Get("Verbosity", false)),                            
        corlen(prob_list.Get("Correlation length", 0.1)),                      
        alpha(1./(corlen*corlen)),                                             
        matern_coeff(ComputeScalingCoefficientForSPDE(corlen, nDim)),
        iter(-1),
        level_size(0),
        nnz(0)
{
    MPI_Comm comm = mesh->GetComm();
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);

    if(!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                << "*  PDESampler_Legacy \n"
                << "*    Correlation length: " << corlen << "\n"
                << "*    Alpha: " << alpha << "\n"
                << "*    Matern Coefficient: " << matern_coeff << "\n";
        if (lognormal) std::cout << "*    Lognormal : true" << std::endl;
        std::cout << std::string(50,'*') << '\n';
    }

}

void PDESampler_Legacy::BuildDeRhamSequence(
        std::vector< std::shared_ptr<AgglomeratedTopology> > & topology)
{
    nLevels = topology.size();


    int feorder(0);
    int orderupscaling(0);
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

        freeCoeffArray(vectCoeffHdiv);
        freeCoeffArray(scalarCoeffL2);

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

void PDESampler_Legacy::SetDeRhamSequence(
        std::vector< std::shared_ptr<DeRhamSequence> > & sequence_)
{
    sequence = sequence_;
}

void PDESampler_Legacy::BuildHierarchy()
{
    // Check deRham sequence has been built and set fespaces
    {
        DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);
        fespace = DRSequence_FE->GetFeSpace(sform);
        fespace_u = DRSequence_FE->GetFeSpace(uform);
    }
    
    nLevels = sequence.size();

    W.resize(nLevels);
    w_sqrt.resize(nLevels);
    D.resize(nLevels);
    A.resize(nLevels);
    precA.resize(nLevels);
    invA.resize(nLevels);
    level_size.SetSize(nLevels);
    nnz.SetSize(nLevels);

    Ps.resize(nLevels-1);
    for(int i(0); i < nLevels-1; ++i)
    {
        Ps[i] = sequence[i]->ComputeTrueP(sform);
    }
 
    Array<int> ess_bdr(mesh->bdr_attributes.Size() ? 
            mesh->bdr_attributes.Max() : 0);
    ess_bdr = 1;

    for(int i(0); i < nLevels; ++i)
    {
        level_size[i] = sequence[i]->GetNumberOfTrueDofs(sform);
        SparseMatrix * M_s = sequence[i]->ComputeMassOperator(uform).release();
        SparseMatrix * W_s = sequence[i]->ComputeMassOperator(sform).release();
        SparseMatrix * D_s = sequence[i]->GetDerivativeOperator(uform);
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

        nnz[i] = A[i]->NNZ();

        delete W_s;
        //delete D_s;
        A_s.reset();
        
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Sampler: Build Solver -- Level ")
                .append(std::to_string(i)));
            if(nDim==3) //form == 2
            {
                adsData.dataAMG.theta = 0.6;
                adsData.dataAMS.dataAlpha.theta = 0.6;
                precA[i] = make_unique<HypreExtension::HypreADS>(
                        *A[i], 
                        sequence[i].get(), 
                        adsData);
            }
            else //form == 1
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
    }

    // Create timer for solver
    for (int ilevel = 0; ilevel < nLevels; ilevel++)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Sampler: Mult -- Level ")
            .append(std::to_string(ilevel)));
    }
}

void PDESampler_Legacy::Sample(const int level, Vector & xi)
{
    xi.SetSize( level_size[level] );
    dist_sampler(xi);
}

void PDESampler_Legacy::Eval(const int level, const Vector & xi, Vector & s)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();
    s.SetSize( size_s );
    s = 0.0;
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    
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
            msg << "Level " << level << " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
      if(myid == 0)
          std::cout << msg.str();
    }
    
    PARELAG_ASSERT(invA[level]->GetConverged());

    Vector div_u( size_s );
    div_u = 0.;
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

void PDESampler_Legacy::Eval(const int level, const Vector & xi, Vector & s,
        Vector & embed_s, bool use_init)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();
    s.SetSize( size_s );
    s = 0.0;
    int xi_level = level_size.Find( xi.Size() );
    PARELAG_ASSERT( xi_level <= level );
    
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
            msg << "Level " << level << " Not Converged. Residual norm " << invA[level]->GetFinalNorm() << "\n";
      if(myid == 0)
          std::cout << msg.str();
    }
    
    PARELAG_ASSERT(invA[level]->GetConverged());

    Vector div_u( size_s );
    div_u = 0.;
    D[level]->Mult(u, div_u);
    {
        Vector & tmp(*(w_sqrt[level]));
        double alphainv = 1./alpha;
        double matern_alphainv = matern_coeff*alphainv;
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = alphainv*div_u(idof) + 
                    matern_alphainv * (*rhs_s)(idof) / (tmp(idof)*tmp(idof));
    }

    embed_s = s;  
    
    if(lognormal)
    {
        for(int idof = 0; idof < level_size[level]; ++idof)
            s(idof) = exp( s(idof) );
    }
    
}

void PDESampler_Legacy::Eval(const int level, const Vector & xi, Vector & s, Vector & u)
{
    const int size_s = level_size[level];
    const int size_u = A[level]->Height();
    
    s.SetSize( size_s );
    s = 0.;
    u.SetSize( size_u );
    u = 0.;

    int xi_level = level_size.Find( xi.Size() );
   
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
    // Set rhs_u == 0
    auto rhs_u = make_unique<Vector>( size_u );
    *rhs_u = 0.;
    
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
    div_u = 0.;
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

double PDESampler_Legacy::ComputeL2Error(
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

double PDESampler_Legacy::ComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    const double e1 = coeff.Max() - exact;
    const double e2 = exact - coeff.Min();
    return std::max(e1, e2);

}

void PDESampler_Legacy::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     mesh->Print(mesh_ofs);
}

void PDESampler_Legacy::SaveFieldGLVis(int level, 
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

void PDESampler_Legacy::SaveFieldGLVis_u(int level, 
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

void PDESampler_Legacy::SaveFieldGLVis_H1(int level, 
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
void PDESampler_Legacy::SaveFieldGLVis_H1Add(int level, 
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

void PDESampler_Legacy::glvis_plot(int ilevel, 
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

void PDESampler_Legacy::prolongate_to_fine_grid(int ilevel, 
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

void PDESampler_Legacy::prolongate_to_fine_grid_u(int ilevel, 
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

void PDESampler_Legacy::prolongate_to_fine_grid_H1(int ilevel, 
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
