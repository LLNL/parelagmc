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
 
#include <memory>
#include <utilities/MPIDataTypes.hpp>

#include "DarcySolver_Legacy.hpp"
#include "Utilities.hpp"
#include "MeshUtilities.hpp"

namespace parelagmc 
{
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

DarcySolver_Legacy::DarcySolver_Legacy( 
            const std::shared_ptr<mfem::ParMesh>& mesh_,
            ParameterList& master_list_):
        prefix_u("velocity_legacy"),
        prefix_p("pressure_legacy"),
        mesh(mesh_),
        uspace(nullptr),
        pspace(nullptr),
        nLevels(0),
        offsets(0,0),
        true_offsets(0,0),
        nDimensions(mesh->Dimension()),
        uform(nDimensions-1),
        pform(nDimensions),
        iter(-1),
        prob_list(master_list_.Sublist("Problem parameters", true)),
        verbose(prob_list.Get("Verbosity", false)),
        saveGLVis(prob_list.Get("Visualize", false)),                          
        feorder(prob_list.Get("Finite element order", 0)),                     
        upscalingOrder(prob_list.Get("Upscaling order", 0))
{
    MPI_Comm_rank(mesh->GetComm(), &myid);
    MPI_Comm_size(mesh->GetComm(), &num_procs);
}

void DarcySolver_Legacy::BuildHierachySpaces(
        std::vector< std::shared_ptr<AgglomeratedTopology > > & topology, 
        unique_ptr<BilinearFormIntegrator> massIntegrator)
{
    nLevels = topology.size();  
    sequence.resize(nLevels);
    nnz.SetSize(nLevels);
    {
        Timer timer = TimeManager::AddTimer(
            "Darcy: DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "Darcy: DeRhamSequence Construction -- Total");

        if(nDimensions == 2)
            sequence[0] = std::make_shared<DeRhamSequence2D_Hdiv_FE>(topology[0], 
                    mesh.get(), 
                    feorder);
        else
            sequence[0] = std::make_shared<DeRhamSequence3D_FE>(topology[0], 
                    mesh.get(), 
                    feorder);
        
        DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);

        DRSequence_FE->ReplaceMassIntegrator(AgglomeratedTopology::ELEMENT, 
                uform, 
                std::move(massIntegrator));
        ConstantCoefficient coeffL2(1.); 
        DRSequence_FE->ReplaceMassIntegrator(
                AgglomeratedTopology::ELEMENT,
                pform, make_unique<MassIntegrator>(coeffL2), false);

        Array<Coefficient *> scalarCoeffL2;
        Array<VectorCoefficient *> vectCoeffHdiv;

        fillVectorCoefficientArray(nDimensions, upscalingOrder, vectCoeffHdiv);
        fillCoefficientArray(nDimensions, upscalingOrder, scalarCoeffL2);

        std::vector<unique_ptr<MultiVector>>
            targets(sequence[0]->GetNumberOfForms());
        
        targets[uform] = DRSequence_FE->InterpolateVectorTargets(uform, 
                vectCoeffHdiv);
        targets[pform] = DRSequence_FE->InterpolateScalarTargets(pform, 
                scalarCoeffL2);

        freeCoeffArray(vectCoeffHdiv);
        freeCoeffArray(scalarCoeffL2);
        mfem::Array<MultiVector*> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();

        sequence[0]->SetTargets(targets_in);
    
        uspace = DRSequence_FE->GetFeSpace(uform);
        pspace = DRSequence_FE->GetFeSpace(pform);
    }

    for(int i(0); i < nLevels-1; ++i)
    {
        Timer timer = TimeManager::AddTimer(
            std::string("Darcy: DeRhamSequence Construction -- Level ")
            .append(std::to_string(i+1)));
        Timer tot_timer = TimeManager::GetTimer(
            "Darcy: DeRhamSequence Construction -- Total");
        sequence[i+1] = sequence[i]->Coarsen();
    }

    // Define the BlockStructure of the problem, i.e. define the array of
    // offsets for each variable. The last component of the Array is the sum
    // of the dimensions of each block.
    offsets.SetSize( nLevels, 3 );
    for(int i(0); i < nLevels; ++i)
    {
        offsets(i,0) = 0;
        offsets(i,1) = sequence[i]->GetNumberOfDofs(uform);
        offsets(i,2) = offsets(i,1) + sequence[i]->GetNumberOfDofs(pform);
    }
    true_offsets.SetSize( nLevels, 3 );
    for(int i(0); i < nLevels; ++i)
    {
        true_offsets(i,0) = 0;
        true_offsets(i,1) = sequence[i]->GetNumberOfTrueDofs(uform);
        true_offsets(i,2) = true_offsets(i,1) + 
                sequence[i]->GetNumberOfTrueDofs(pform);
    }

    // Get the maps for u and p
    // Compute the (local) B, Bt 
    umap.SetSize( nLevels );
    pmap.SetSize( nLevels );

    for(int i(0); i < nLevels; ++i)
    {
        umap[i] = &(sequence[i]->GetDofHandler(uform)->GetDofTrueDof());
        pmap[i] = &(sequence[i]->GetDofHandler(pform)->GetDofTrueDof());
    }

    P.resize( nLevels-1 );                                                     
    Pi.resize( nLevels-1 );

    for(int i(0); i < nLevels-1; ++i)
    {
        Array<int> f_offsets(offsets.GetRow(i),3);
        Array<int> c_offsets(offsets.GetRow(i+1),3);

        P[i] = make_unique<BlockMatrix>(f_offsets,c_offsets);
        P[i]->SetBlock( 0,0, (sequence[i]->GetP(uform)) );
        P[i]->SetBlock( 1,1, (sequence[i]->GetP(pform)) );

        Pi[i] = make_unique<BlockMatrix>(c_offsets, f_offsets);
        Pi[i]->SetBlock(0,0, const_cast<SparseMatrix *>(
                    &sequence[i]->GetPi(uform)->GetProjectorMatrix()));
        Pi[i]->SetBlock(1,1, const_cast<SparseMatrix *>(
                    &sequence[i]->GetPi(pform)->GetProjectorMatrix()));
    }

    // Create timer for Mult (i.e. linear solve of Darcy)
    for (int i = 0; i < nLevels; i++)
    {
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Darcy: Mult -- Level ")
                .append(std::to_string(i)));
        }
        {
            Timer timer = TimeManager::AddTimer(
                std::string("Darcy: Build Solver -- Level ")
                .append(std::to_string(i)));
        }
    }
}

void DarcySolver_Legacy::BuildVolumeObservationFunctional_P(
        LinearFormIntegrator * observationFunctional_p)
{
    g_obs_func.resize(nLevels);
    p_g_obs_func.resize( nLevels );
    for(int i(0); i < nLevels; ++i)
    {
        g_obs_func[i] = make_unique<Vector>(
                sequence[i]->GetNumberOfDofs(pform) + sequence[i]->GetNumberOfDofs(uform));
        *(g_obs_func[i]) = 0.;
    }
    LinearForm fun_p;
    fun_p.AddDomainIntegrator(observationFunctional_p);
    fun_p.Update(pspace, *(g_obs_func[0]), sequence[0]->GetNumberOfDofs(uform) );
    fun_p.Assemble();
    for(int i(0); i < nLevels-1; ++i)
        P[i]->MultTranspose( *(g_obs_func[i]), *(g_obs_func[i+1]));
    for(int i(0); i < nLevels; ++i)
        p_g_obs_func[i] = parAssemble(i, *(g_obs_func[i]));
}

void DarcySolver_Legacy::BuildVolumeObservationFunctional(
        LinearFormIntegrator * observationFunctional_u, 
        LinearFormIntegrator * observationFunctional_p)
{
    obs_func.resize(nLevels);
    p_obs_func.resize( nLevels );
    for(int i(0); i < nLevels; ++i)
    {
        obs_func[i] = make_unique<Vector>(sequence[i]->GetNumberOfDofs(uform)
                +sequence[i]->GetNumberOfDofs(pform) );
        *(obs_func[i]) = 0.;
    }

    LinearForm fun_u;
    fun_u.AddDomainIntegrator(observationFunctional_u);
    fun_u.Update(uspace,  *(obs_func[0]), 0 );
    fun_u.Assemble();

    LinearForm fun_p;
    fun_p.AddDomainIntegrator(observationFunctional_p);
    fun_p.Update(pspace, *(obs_func[0]), sequence[0]->GetNumberOfDofs(uform) );
    fun_p.Assemble();

    for(int i(0); i < nLevels-1; ++i)
        P[i]->MultTranspose( *(obs_func[i]), *(obs_func[i+1]));
    
    for(int i(0); i < nLevels; ++i)
        p_obs_func[i] = parAssemble(i, *(obs_func[i]));
}

void DarcySolver_Legacy::BuildBdrObservationFunctional(
        LinearFormIntegrator * observationFunctional_u)
{
    obs_func.resize( nLevels );
    p_obs_func.resize( nLevels );
    for(int i(0); i < nLevels; ++i)
    {
        obs_func[i] = make_unique<Vector>( sequence[i]->GetNumberOfDofs(uform)
                +sequence[i]->GetNumberOfDofs(pform) );
        *(obs_func[i]) = 0.;
    }

    LinearForm fun_u;
    fun_u.AddBoundaryIntegrator(observationFunctional_u);
    fun_u.Update(uspace,  *(obs_func[0]), 0 );
    fun_u.Assemble();

    for(int i(0); i < nLevels-1; ++i)
        P[i]->MultTranspose( *(obs_func[i]), *(obs_func[i+1]));

    for(int i(0); i < nLevels; ++i)
        p_obs_func[i] = parAssemble(i, *(obs_func[i]));

}

void DarcySolver_Legacy::BuildPWObservationFunctional_p(
       std::vector<double> & v_obs_data_coords, const double eps)
{
    ChangeMeshAttributes(*mesh, 1, v_obs_data_coords, eps );

    if (saveGLVis) this->SaveMeshGLVis("pw_p_mesh");

    obs_func.resize( nLevels );
    p_obs_func.resize( nLevels );

    for(int i(0); i < nLevels; ++i)
    {
        obs_func[i] = make_unique<Vector>( sequence[i]->GetNumberOfDofs(uform)
                +sequence[i]->GetNumberOfDofs(pform) );
        *(obs_func[i]) = 0.;
    }

    {
        auto  fec = parelag::make_unique<L2_FECollection>(0, nDimensions);
        auto fespace = parelag::make_unique<FiniteElementSpace>(mesh.get(), fec.get());
        Array<int> obs_data_attr(2);
        obs_data_attr[0] = 0;
        obs_data_attr[1] = 1;
        ConstantCoefficient one(1.0);
        RestrictedCoefficient obs_data_coeff(one, obs_data_attr);
        LinearForm fun_p;
        fun_p.AddDomainIntegrator(new DomainLFIntegrator(obs_data_coeff));
        fun_p.Update(fespace.get(), *(obs_func[0]), sequence[0]->GetNumberOfDofs(uform));
        fun_p.Assemble();
    }

    for(int i(0); i < nLevels-1; ++i)
        P[i]->MultTranspose( *(obs_func[i]), *(obs_func[i+1]));

    for(int i(0); i < nLevels; ++i)
        p_obs_func[i] = parAssemble(i, *(obs_func[i]));

}

void DarcySolver_Legacy::SetEssBdrConditions(Array<int> & ess_bc_, 
        VectorCoefficient & u_bdr )
{
    ess_data.resize( nLevels );
    for(int i(0); i < nLevels; ++i)
        ess_data[i] = make_unique<Vector>( sequence[i]->GetNumberOfDofs(uform)
                +sequence[i]->GetNumberOfDofs(pform) );

    *(ess_data[0]) = 0.0;
    ess_bc_.Copy(ess_bc);
    GridFunction u;
    u.MakeRef(uspace, *(ess_data[0]), 0 );
    u.ProjectBdrCoefficientNormal(u_bdr, ess_bc);

    for(int i(0); i < nLevels-1; ++i)
        Pi[i]->Mult( *(ess_data[i]), *(ess_data[i+1]));
}

void DarcySolver_Legacy::BuildForcingTerms(VectorCoefficient & fcoef, 
        Coefficient & pcoef_bdr, 
        Coefficient & qcoef)
{
    rhs.resize( nLevels );
    for(int i(0); i < nLevels; ++i)
    {
        rhs[i] = make_unique<Vector>( sequence[i]->GetNumberOfDofs(uform)+
                sequence[i]->GetNumberOfDofs(pform) );
        *(rhs[i]) = 0.0;
    }

    LinearForm rhs_u;
    rhs_u.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoef));
    rhs_u.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(pcoef_bdr));
    rhs_u.Update(uspace, *(rhs[0]), 0);
    rhs_u.Assemble();

    LinearForm rhs_p;
    rhs_p.AddDomainIntegrator(new DomainLFIntegrator(qcoef));
    rhs_p.Update(pspace,  *(rhs[0]), sequence[0]->GetNumberOfDofs(uform) );
    rhs_p.Assemble();

    for(int i(0); i < nLevels-1; ++i)
        P[i]->MultTranspose( *(rhs[i]), *(rhs[i+1]));

}

void DarcySolver_Legacy::SolveFwd(int ilevel, Vector & k_over_k_ref, 
        double & Q, double & C)
{
    BlockMatrix * A = assemble(ilevel, k_over_k_ref);
    Vector rhs_bc(A->NumRows());
    rhs_bc = *(rhs[ilevel]);

    Array<int> ess_bc_dofs( sequence[ilevel]->GetNumberOfDofs(uform)+
            sequence[ilevel]->GetNumberOfDofs(pform) );
    ess_bc_dofs = 0;
    Array<int> ess_bc_udofs(ess_bc_dofs.GetData(), 
            sequence[ilevel]->GetNumberOfDofs(uform) );
    sequence[ilevel]->GetDofHandler(uform)->
            MarkDofsOnSelectedBndr(ess_bc, ess_bc_udofs);

    A->EliminateRowCol(ess_bc_dofs, *(ess_data[ilevel]), rhs_bc);

    unique_ptr<Vector> p_rhs = parAssemble(ilevel, rhs_bc);
    unique_ptr<Vector> p_sol = make_unique<Vector>(p_rhs->Size());

    solve(ilevel, *A, *p_rhs, *p_sol);
    Q = dot( *(p_obs_func[ilevel]), *p_sol, umap[ilevel]->GetComm());

    C = umap[ilevel]->GetTrueGlobalSize() + pmap[ilevel]->GetTrueGlobalSize();

    if(saveGLVis)
    {
        auto coeff = parDistribute(ilevel, *p_sol);
        SaveFieldGLVis(ilevel, *coeff, prefix_u,prefix_p);
    }

    delete A;
}

void DarcySolver_Legacy::SolveFwd_RtnPressure(int ilevel, Vector & k_over_k_ref, 
        Vector & P, double & C, double & Q, bool compute_Q)
{
    BlockMatrix * A = assemble(ilevel, k_over_k_ref);
    Vector rhs_bc(A->NumRows());
    rhs_bc = *(rhs[ilevel]);

    Array<int> ess_bc_dofs( sequence[ilevel]->GetNumberOfDofs(uform)+
            sequence[ilevel]->GetNumberOfDofs(pform) );
    ess_bc_dofs = 0;
    Array<int> ess_bc_udofs(ess_bc_dofs.GetData(), 
            sequence[ilevel]->GetNumberOfDofs(uform) );
    sequence[ilevel]->GetDofHandler(uform)->
            MarkDofsOnSelectedBndr(ess_bc, ess_bc_udofs);

    A->EliminateRowCol(ess_bc_dofs, *(ess_data[ilevel]), rhs_bc);

    unique_ptr<Vector> p_rhs = parAssemble(ilevel, rhs_bc);
    unique_ptr<Vector> p_sol = make_unique<Vector>(p_rhs->Size());

    solve(ilevel, *A, *p_rhs, *p_sol);
   
    if (compute_Q) Q = dot( *(p_obs_func[ilevel]), *p_sol, umap[ilevel]->GetComm()); 
    
    // Want p-portion of p_sol so creating a block vector (silly)
    Array<int> r_offsets(offsets.GetRow(ilevel), 3);
    Array<int> trueBlockOffsets(true_offsets.GetRow(ilevel), 3);

    BlockVector t_sol(p_sol->GetData(), trueBlockOffsets);
    Vector * out = new Vector(r_offsets.Last());
    BlockVector b_out(out->GetData(), r_offsets);
    umap[ilevel]->Distribute(t_sol.GetBlock(0), b_out.GetBlock(0));
    pmap[ilevel]->Distribute(t_sol.GetBlock(1), b_out.GetBlock(1));

    P = b_out.GetBlock(1); 

    C = umap[ilevel]->GetTrueGlobalSize() + pmap[ilevel]->GetTrueGlobalSize();

    delete out;
    delete A;
}

BlockMatrix* DarcySolver_Legacy::assemble(int ilevel, Vector & k_over_k_ref)
{
    auto M = sequence[ilevel]->ComputeMassOperator(uform, k_over_k_ref);
    
    auto W = sequence[ilevel]->ComputeMassOperator(pform);
    auto D = sequence[ilevel]->GetDerivativeOperator(uform);

    auto B = Mult(*W,*D);
    auto BT = Transpose(*B);

    Array<int> my_offsets(offsets.GetRow(ilevel), 3);
    BlockMatrix * out = new BlockMatrix(my_offsets);
    out->owns_blocks = true;
    out->SetBlock(0,0, M.release());
    out->SetBlock(1,0, B);
    out->SetBlock(0,1, BT);

    return out;
}

void DarcySolver_Legacy::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     mesh->Print(mesh_ofs);
}

void DarcySolver_Legacy::SaveFieldGLVis(int level, 
        const Vector & coeff, 
        const std::string prefix_u, 
        const std::string prefix_p) const
{
    Array<int> d_offsets(const_cast<int*>( offsets.GetRow(0) ), 3);
    BlockVector x(d_offsets);
    prolongate_to_fine_grid(level, coeff, x);

    GridFunction gf;
    {
        gf.MakeRef(uspace, x.GetBlock(0), 0);
        std::ostringstream fid_name;
        fid_name << prefix_u << "_L" << std::setfill('0') << std::setw(2) << 
                level << "." << std::setw(6) << myid;
        std::ofstream fid(fid_name.str().c_str());
        fid.precision(8);
        gf.Save(fid);
    }
    {
        gf.MakeRef(pspace, x.GetBlock(1), 0);
        std::ostringstream fid_name;
        fid_name << prefix_p << "_L" << std::setfill('0') << std::setw(2) << 
                level << "." << std::setw(6) << myid;
        std::ofstream fid(fid_name.str().c_str());
        fid.precision(8);
        gf.Save(fid);
    }
}

void DarcySolver_Legacy::solve(int ilevel, BlockMatrix & A, 
        const Vector & rhs_bc, Vector & sol)
{
    std::unique_ptr<HypreParMatrix> p_M = Assemble( *umap[ilevel], 
                                    A.GetBlock(0,0), 
                                    *umap[ilevel]);
    std::unique_ptr<HypreParMatrix> p_Bt = Assemble( *umap[ilevel], 
                                    A.GetBlock(0,1), 
                                    *pmap[ilevel]);
    std::unique_ptr<HypreParMatrix> p_B = Assemble( *pmap[ilevel], 
                                    A.GetBlock(1,0), 
                                    *umap[ilevel]);

    nnz[ilevel] = p_M->NNZ() + p_Bt->NNZ() + p_B->NNZ();

    Vector M_diag(p_M->Height());
    p_M->GetDiag(M_diag);

    // Setup Schur complements
    auto tmp = Assemble( *umap[ilevel], 
                         A.GetBlock(0,1), 
                         *pmap[ilevel]);
    tmp->InvScaleRows(M_diag);
    std::unique_ptr<HypreParMatrix>  S = ToUnique(ParMult(p_B.get(), tmp.get()));

    Array<int> trueBlockOffsets(true_offsets.GetRow(ilevel), 3);
    BlockOperator p_A(trueBlockOffsets);
    // operator will NOT delete p_M, p_Bt, p_B
    p_A.owns_blocks = false;
    p_A.SetBlock(0,0, p_M.get() );
    p_A.SetBlock(0,1, p_Bt.get() );
    p_A.SetBlock(1,0, p_B.get() );

    BlockDiagonalPreconditioner prec(trueBlockOffsets);
    {
        Timer timer = TimeManager::AddTimer(
                std::string("Darcy: Build Solver -- Level ")
                .append(std::to_string(ilevel)));
        // prec will delete 
        prec.owns_blocks = true;
        prec.SetDiagonalBlock(0, new HypreSmoother(*p_M));
        
        HypreExtension::HypreBoomerAMG * amg_prec = new HypreExtension::HypreBoomerAMG(*S);
        //AMGdata.print_level = 1;
        //amg_prec->SetParameters(AMGdata);
        prec.SetDiagonalBlock(1, amg_prec);
    }
    MINRESSolver solver(umap[ilevel]->GetComm());
    solver.SetPrintLevel(-1);
    solver.SetAbsTol(1e-12);
    solver.SetRelTol(1e-6);
    solver.SetMaxIter(2000);
    solver.SetOperator(p_A);
    solver.SetPreconditioner(prec);
    sol = 0.0;
 
    if (verbose) 
    {
        double local_norm = rhs_bc.Norml2() * rhs_bc.Norml2();
        double global_norm;
        MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                       MPI_SUM,0,mesh->GetComm());

        if (!myid)
                std::cout <<  "-- Initial residual norm: " << std::sqrt(global_norm)
                          << std::endl;
    }
 
    {
        Timer timer = TimeManager::AddTimer(
                std::string("Darcy: Mult -- Level ")
                .append(std::to_string(ilevel)));
        solver.Mult(rhs_bc, sol);
    }

    if (verbose)
    {
        mfem::Vector tmp(p_A.Height());
        p_A.Mult(sol,tmp);
        tmp -= rhs_bc;
        double local_norm = tmp.Norml2() * tmp.Norml2();
        double global_norm;
        MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                   MPI_SUM,0,mesh->GetComm());

        if (!myid)
            std::cout << "-- Final residual norm: " << std::sqrt(global_norm)
                      << std::endl;
    }

    if (!solver.GetConverged())
    {
        std::cout << "DarcySolver_Legacy did not converge! Residual norm " << solver.GetFinalNorm() << "\n";
    }
    iter = solver.GetNumIterations();
    PARELAG_ASSERT(solver.GetConverged());

    tmp.reset();
    S.reset();
}

unique_ptr<Vector> DarcySolver_Legacy::parAssemble(int ilevel, Vector & rhs)
{
    Array<int> r_offsets(offsets.GetRow(ilevel), 3);
    BlockVector rrhs(rhs.GetData(), r_offsets);
    Array<int> trueBlockOffsets(true_offsets.GetRow(ilevel), 3);
    unique_ptr<Vector> out = make_unique<Vector>(trueBlockOffsets.Last());
    BlockVector prhs(out->GetData(), trueBlockOffsets);
    umap[ilevel]->Assemble(rrhs.GetBlock(0), prhs.GetBlock(0));
    pmap[ilevel]->Assemble(rrhs.GetBlock(1), prhs.GetBlock(1));

    return out;
}

unique_ptr<Vector> DarcySolver_Legacy::parDistribute(int ilevel, Vector & tsol)
{
    Array<int> r_offsets(offsets.GetRow(ilevel), 3);
    Array<int> trueBlockOffsets(true_offsets.GetRow(ilevel), 3);

    BlockVector t_sol(tsol.GetData(), trueBlockOffsets);
    unique_ptr<Vector> out = make_unique<Vector>(r_offsets.Last());
    BlockVector b_out(out->GetData(), r_offsets);
    umap[ilevel]->Distribute(t_sol.GetBlock(0), b_out.GetBlock(0));
    pmap[ilevel]->Distribute(t_sol.GetBlock(1), b_out.GetBlock(1));

    return out;
}

void DarcySolver_Legacy::prolongate_to_fine_grid(int ilevel, const Vector & coeff, 
        Vector & x) const
{
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    for(int lev(ilevel); lev > 0; --lev)
    {
        PARELAG_ASSERT(feCoeff->Size() == P[lev-1]->Width());
        auto help = make_unique<Vector>( P[lev-1]->Height() );
        P[lev-1]->Mult(*feCoeff, *help);
        feCoeff = std::move(help);
    }

    PARELAG_ASSERT(x.Size()==feCoeff->Size());
    x = *feCoeff;
}

} /* namespace parelagmc */
