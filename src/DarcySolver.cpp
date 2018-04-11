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
 
#include <utilities/MPIDataTypes.hpp>

#include "DarcySolver.hpp"
#include "Utilities.hpp"
#include "MeshUtilities.hpp"

namespace parelagmc 
{
using namespace mfem;
using namespace parelag;
using std::unique_ptr;

DarcySolver::DarcySolver( 
            const std::shared_ptr<mfem::ParMesh>& mesh_,
            ParameterList& master_list_):
        prefix_u("velocity"),
        prefix_p("pressure"),
        mesh(mesh_),
        uspace(nullptr),
        pspace(nullptr),
        nLevels(0),
        offsets(0,0),
        true_offsets(0,0),
        nDimensions(mesh->Dimension()),
        uform(nDimensions-1),
        pform(nDimensions),
        master_list(master_list_),
        prob_list(master_list.Sublist("Problem parameters", true)),
        entire_seq(prob_list.Get("Build entire sequence", false)),
        verbose(prob_list.Get("Verbosity", false)),
        saveGLVis(prob_list.Get("Visualize", false)),
        feorder(prob_list.Get("Finite element order", 0)),
        upscalingOrder(prob_list.Get("Upscaling order", 0)),
        prec_type(master_list.Sublist("Physical problem parameters",true)
            .Get("Linear solver","MINRES-BJ-GS"))
{
    MPI_Comm_rank(mesh->GetComm(), &myid);
    MPI_Comm_size(mesh->GetComm(), &num_procs);

    if (!myid)
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                  << "*  DarcySolver \n"
                  << "*    Solver: " << prec_type << '\n';
                  std::cout << std::string(50,'*') << '\n' << std::endl;
    }
}

void DarcySolver::BuildHierachySpaces(
        std::vector< std::shared_ptr<AgglomeratedTopology > > & topology, 
        unique_ptr<BilinearFormIntegrator> massIntegrator)
{
    nLevels = topology.size();  
    nnz.SetSize(nLevels);
    nnz = 0;
    sequence.resize(nLevels);
    
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
                std::move(massIntegrator), true);

        ConstantCoefficient coeffL2(1.);
        DRSequence_FE->ReplaceMassIntegrator(
                AgglomeratedTopology::ELEMENT, 
                pform, make_unique<MassIntegrator>(coeffL2), false);
    
        mfem::Array<mfem::Coefficient *> L2coeff;
        mfem::Array<mfem::VectorCoefficient *> Hdivcoeff;
        mfem::Array<mfem::VectorCoefficient *> Hcurlcoeff;
        mfem::Array<mfem::Coefficient *> H1coeff;
        fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
        fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
        fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);
        fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);

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
        if (nDimensions == 2)
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
        mfem::Array<MultiVector*> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();
        
        int jFormStart;
        if (entire_seq)
            jFormStart = 0;
        else
            jFormStart = nDimensions - 1;
        sequence[0]->SetjformStart(jFormStart); 
        
        sequence[0]->SetTargets(targets_in);
    
        uspace = DRSequence_FE->GetFeSpace(uform);
        pspace = DRSequence_FE->GetFeSpace(pform);
    }
    
    if (!myid) std::cout << "-- Darcy: Coarsen de Rham sequence" << std::endl; 
    
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

    // GetNumberOfDofs: number of local dofs (true + shared)
    offsets.SetSize( nLevels, 3 );
    for(int i(0); i < nLevels; ++i)
    {
        offsets(i,0) = 0;
        offsets(i,1) = sequence[i]->GetNumberOfDofs(uform);
        offsets(i,2) = offsets(i,1) + sequence[i]->GetNumberOfDofs(pform);
    }
    // GetNumberTrueDofs: number of local true dofs
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
    B.resize( nLevels );
    Bt.resize( nLevels );
    for(int i(0); i < nLevels; ++i)
    {
        umap[i] = &(sequence[i]->GetDofHandler(uform)->GetDofTrueDof());
        pmap[i] = &(sequence[i]->GetDofHandler(pform)->GetDofTrueDof());
    
        auto W = sequence[i]->ComputeMassOperator(pform);
        auto D = sequence[i]->GetDerivativeOperator(uform);
        
        B[i] = ToUnique(Mult(*W,*D));
        Bt[i] = ToUnique(Transpose(*B[i]));
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
                std::string("Darcy: Build Solver (").append(prec_type)
                .append(") -- Level ").append(std::to_string(i)));
        }
    }
}

void DarcySolver::BuildVolumeObservationFunctional_P(
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

void DarcySolver::BuildVolumeObservationFunctional(
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

void DarcySolver::BuildBdrObservationFunctional(
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

void DarcySolver::BuildPWObservationFunctional_p( 
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

void DarcySolver::SetEssBdrConditions(Array<int> & ess_bc_, 
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

    // Make BCs for parelag's solver_state
    // Size=2 since 2 forms; ess_attr[1].Size() = 0 because no BC's
    // for that form
    ess_attr.resize(2);
    ess_attr[0].resize(ess_bc.Size());
    for (int i = 0; i < ess_bc.Size(); i++)
        ess_attr[0][i] = ess_bc[i];
}

void DarcySolver::BuildForcingTerms(VectorCoefficient & fcoef, 
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

void DarcySolver::SolveFwd(int ilevel, Vector & k_over_k_ref, 
        double & Q, double & C)
{
    // Generate block A and form rhs_bc
    std::shared_ptr<MfemBlockOperator> A = assemble(ilevel, k_over_k_ref, rhs_bc);
   
    // solve will correct the size of p_sol depending on solver
    std::unique_ptr<Vector> p_sol = make_unique<Vector>(rhs_bc.Size());
    
    solve(ilevel, A, k_over_k_ref, rhs_bc, *p_sol);
    
    Q = dot( *(p_obs_func[ilevel]), *p_sol, umap[ilevel]->GetComm()); 

    C = umap[ilevel]->GetTrueGlobalSize() + pmap[ilevel]->GetTrueGlobalSize();

    if(saveGLVis)
    {
        auto coeff = parDistribute(ilevel, *p_sol);
        SaveFieldGLVis(ilevel, *coeff, prefix_u, prefix_p);
    }
    
}

void DarcySolver::SolveFwd_RtnPressure(int ilevel, Vector & k_over_k_ref, 
        Vector & P, double & C, double & Q, bool compute_Q)
{
    std::shared_ptr<MfemBlockOperator> A = assemble(ilevel, k_over_k_ref, rhs_bc);

    // solve will correct the size depending on solver
    std::unique_ptr<Vector> p_sol = make_unique<Vector>(rhs_bc.Size());
   
    solve(ilevel, A, k_over_k_ref, rhs_bc, *p_sol);

    if (compute_Q) Q = dot( *(p_obs_func[ilevel]), *p_sol, umap[ilevel]->GetComm());

    if(saveGLVis)
    {
        auto coeff = parDistribute(ilevel, *p_sol);
        SaveFieldGLVis(ilevel, *coeff, prefix_u,prefix_p);
    }
    
    // Want p-portion of p_sol so creating a block vector (silly)
    Array<int> r_offsets(offsets.GetRow(ilevel), 3);
    Array<int> trueBlockOffsets(true_offsets.GetRow(ilevel), 3);

    BlockVector t_sol(p_sol->GetData(), trueBlockOffsets);
    unique_ptr<Vector> out = make_unique<Vector>(r_offsets.Last());
    BlockVector b_out(out->GetData(), r_offsets);
    umap[ilevel]->Distribute(t_sol.GetBlock(0), b_out.GetBlock(0));
    pmap[ilevel]->Distribute(t_sol.GetBlock(1), b_out.GetBlock(1));

    P = b_out.GetBlock(1);
    
    C = pmap[ilevel]->GetTrueGlobalSize() + umap[ilevel]->GetTrueGlobalSize();
}

std::shared_ptr<MfemBlockOperator> DarcySolver::assemble(int ilevel, Vector & k_over_k_ref,
    Vector & rhs_bc)
{
    Array<int> my_offsets(offsets.GetRow(ilevel), 3);
    Array<int> my_true_offsets(true_offsets.GetRow(ilevel), 3);
    
    // Hdiv mass matrix
    auto M = sequence[ilevel]->ComputeMassOperator(uform, k_over_k_ref);
    
    BlockMatrix * mA = new BlockMatrix(my_offsets);
    mA->owns_blocks = 0;
    mA->SetBlock(0,0,M.get());
    mA->SetBlock(0,1,Bt[ilevel].get());
    mA->SetBlock(1,0,B[ilevel].get());

    Array<int> ess_bc_dofs( sequence[ilevel]->GetNumberOfDofs(uform)+
            sequence[ilevel]->GetNumberOfDofs(pform) );
    ess_bc_dofs = 0;
    Array<int> ess_bc_udofs(ess_bc_dofs.GetData(),
            sequence[ilevel]->GetNumberOfDofs(uform) );
    sequence[ilevel]->GetDofHandler(uform)->
            MarkDofsOnSelectedBndr(ess_bc, ess_bc_udofs);

    rhs_bc.SetSize(mA->NumRows());
    rhs_bc = *(rhs[ilevel]);
    
    mA->EliminateRowCol(ess_bc_dofs, *(ess_data[ilevel]), rhs_bc);
   
    std::unique_ptr<HypreParMatrix> p_M = Assemble( *umap[ilevel],
                                    mA->GetBlock(0,0),
                                    *umap[ilevel]);
    std::unique_ptr<HypreParMatrix> p_Bt = Assemble( *umap[ilevel],
                                    mA->GetBlock(0,1),
                                    *pmap[ilevel]);
    std::unique_ptr<HypreParMatrix> p_B = Assemble( *pmap[ilevel],
                                    mA->GetBlock(1,0),
                                    *umap[ilevel]);
    // nnz calculation
    nnz[ilevel] = p_M->NNZ() + p_Bt->NNZ() + p_B->NNZ();
 
    auto A = std::make_shared<MfemBlockOperator>(my_true_offsets);
    
    A->SetBlock(0,0,std::move(p_M)); 
    A->SetBlock(1,0,std::move(p_B)); 
    A->SetBlock(0,1,std::move(p_Bt)); 

    delete mA;
    return A;
}

void DarcySolver::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     mesh->Print(mesh_ofs);
}

void DarcySolver::SaveFieldGLVis(int level, 
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

void DarcySolver::solve(int ilevel, const std::shared_ptr<MfemBlockOperator> p_A,
        const Vector & k_over_k_ref, const Vector & rhs_bc, Vector & p_sol)
{
    // Create the solver/preconditioner

    // Get the factory
    auto lib = SolverLibrary::CreateLibrary(
        master_list.Sublist("Preconditioner Library"));

    auto solver_list = master_list.Sublist("Preconditioner Library")
                       .Sublist(prec_type);
    if (verbose)
        solver_list.Sublist("Solver Parameters").Set("Print final paragraph",true);

    const std::string new_prec_type = "new prec";
    lib->AddSolver(new_prec_type,std::move(solver_list));

    // Get the factory
    auto prec_factory = lib->GetSolverFactory(new_prec_type);
    auto solver_state = prec_factory->GetDefaultState();
    solver_state->SetDeRhamSequence(sequence[ilevel]);
    solver_state->SetBoundaryLabels(ess_attr);
    solver_state->SetForms({uform,pform});

    if (prec_type.compare("Hybridization")==0)
    {
        solver_state->SetExtraParameter("IsSameOrient",(ilevel>0));
        auto shared_k_over_k = std::make_shared<Vector>(k_over_k_ref.GetData(), k_over_k_ref.Size());
        solver_state->SetVector("elemMatrixScaling", shared_k_over_k);
    }
    
    unique_ptr<mfem::Solver> solver;

    // Build the solver/preconditioner
    {
        Timer timer = TimeManager::AddTimer(
                std::string("Darcy: Build Solver (").append(prec_type)
                .append(") -- Level ").append(std::to_string(ilevel)));
        solver = prec_factory->BuildSolver(p_A,*solver_state);
    }
  
    if (verbose)
    {
        double local_norm = rhs_bc.Norml2() * rhs_bc.Norml2();
        double global_norm;
        MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                       MPI_SUM,0,mesh->GetComm());

        if (!myid)
                std::cout <<  "-- Initial residual norm: " << std::sqrt(global_norm);
    }
 
    std::unique_ptr<Vector> p_rhs_bc = parAssemble(ilevel, rhs_bc);
    {
        Timer timer = TimeManager::AddTimer(
                std::string("Darcy: Mult -- Level ")
                .append(std::to_string(ilevel)));
        if (prec_type.compare("Hybridization")==0)
        {
            Array<int> my_offsets(offsets.GetRow(ilevel), 3);
            mfem::BlockVector sol(my_offsets);
            sol = 0.;
            solver->Mult(rhs_bc, sol);
            p_sol = *parAssemble(ilevel, sol);
        }
        else
        {
            p_sol.SetSize(p_rhs_bc->Size()); 
            p_sol = 0.;
            solver->Mult(*p_rhs_bc, p_sol);
        }
    }

    if (verbose)
    {
        mfem::Vector tmp(p_A->Height());
        p_A->Mult(p_sol,tmp);
        tmp -= *p_rhs_bc; 
        double local_norm = tmp.Norml2() * tmp.Norml2();
        double global_norm;
        MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                   MPI_SUM,0,mesh->GetComm());

        if (!myid)
            std::cout << "-- Final residual norm: " << std::sqrt(global_norm)
                      << std::endl;
    }
}

unique_ptr<Vector> DarcySolver::parAssemble(int ilevel, const Vector & rhs) 
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

unique_ptr<Vector> DarcySolver::parDistribute(int ilevel, const Vector & tsol) 
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

void DarcySolver::prolongate_to_fine_grid(int ilevel, const Vector & coeff, 
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
