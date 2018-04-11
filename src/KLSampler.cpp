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
 
#include "KLSampler.hpp"
#include "Utilities.hpp"

namespace parelagmc
{

using namespace mfem;
using namespace parelag;

KLSampler::KLSampler(
            const std::shared_ptr<ParMesh>& mesh_, 
            NormalDistributionSampler& dist_sampler_, 
            CovarianceFunction& covariance_, 
            ParameterList& master_list_):
        mesh(mesh_),
        dist_sampler(dist_sampler_),
        covariance(covariance_),
        nDim(mesh->Dimension()),
        prob_list(master_list_.Sublist("Problem parameters", true)),    
        lognormal(prob_list.Get("Lognormal", true)),
        nLevels(0),
        fec(nullptr),
        fespace(nullptr)
{
    MPI_Comm comm = mesh->GetComm();
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);

    // Create the fespace                                                      
    fec = parelag::make_unique<L2_FECollection>(0, nDim);
    fespace = parelag::make_unique<FiniteElementSpace>(mesh.get(), fec.get());

}

void KLSampler::SetDeRhamSequence( 
        std::vector< std::shared_ptr<DeRhamSequence> > & sequence_)
{
    sequence = sequence_;
}

void KLSampler::BuildDeRhamSequence( 
        std::vector< std::shared_ptr<AgglomeratedTopology> > & topology)
{
    nLevels = topology.size();
    sequence.resize( nLevels );

    int feorder = 0;
    int upscalingOrder = 0;
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

    sequence[0]->SetjformStart(nDim);

    mfem::ConstantCoefficient coeffH1(1.);
    mfem::ConstantCoefficient coeffDer(1.);

    std::vector<std::unique_ptr<MultiVector>>
        targets(sequence[0]->GetNumberOfForms());

//    auto const AT_elem = AgglomeratedTopology::ELEMENT;
    //mfem::Array<MultiVector *> targets(sequence[0]->GetNumberOfForms());
    
    /*
    DRSequence_FE->ReplaceMassIntegrator(
        AT_elem,
        0,
        make_unique<MassIntegrator>(coeffH1),
        false);
    DRSequence_FE->ReplaceMassIntegrator(
        AT_elem,
        1,
        make_unique<VectorFEMassIntegrator>(coeffDer),
        true);
*/
    Array<Coefficient *> L2coeff;
    Array<VectorCoefficient *> Hdivcoeff;
    Array<VectorCoefficient *> Hcurlcoeff;
    Array<Coefficient *> H1coeff;
    fillVectorCoefficientArray(nDim, upscalingOrder, Hcurlcoeff);
    fillVectorCoefficientArray(nDim, upscalingOrder, Hdivcoeff);
    fillCoefficientArray(nDim, upscalingOrder, L2coeff);
    fillCoefficientArray(nDim, upscalingOrder+1, H1coeff);

    for (int jform = 0; jform<sequence[0]->GetNumberOfForms(); ++jform)
        targets[jform] = nullptr;

    int jform(0);

    targets[jform] =
        DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);
    ++jform;

    if(nDim == 3)
    {
        targets[jform] =
            DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
        ++jform;
    }

    targets[jform] =
        DRSequence_FE->InterpolateVectorTargets(jform, Hdivcoeff);
    ++jform;
    
    targets[jform] =
        DRSequence_FE->InterpolateScalarTargets(jform, L2coeff);
    ++jform;

    freeCoeffArray(L2coeff);
    freeCoeffArray(Hdivcoeff);
    freeCoeffArray(Hcurlcoeff);
    freeCoeffArray(H1coeff);

    Array<MultiVector *> targets_in(targets.size());
    for (int ii = 0; ii < targets_in.Size(); ++ii)
        targets_in[ii] = targets[ii].get();
    sequence[0]->SetTargets(targets_in);

    for(int i(0); i < nLevels-1; ++i)
        sequence[i+1] = sequence[i]->Coarsen();
}
 
void KLSampler::BuildHierarchy()
{
    // Check that sequence exists
    PARELAG_ASSERT(sequence[0].get());
    nLevels = sequence.size();
    P.resize(nLevels-1);
    for(int i(0); i < nLevels-1; ++i)
        P[i] = sequence[i]->ComputeTrueP(nDim);

    totnmodes = covariance.NumberOfModes();
    level_size.SetSize(nLevels);
    level_size[0] = sequence[0]->GetNumberOfDofs(nDim);

    evect.SetSize( nLevels );
    {
        parelag::Timer timer = parelag::TimeManager::AddTimer(
            "KLSampler -- SolveEigenvalue");
        covariance.SolveEigenvalue();
    }
    eval = &covariance.Eigenvalues();
    evect[0] = &covariance.Eigenvectors();
    
    BilinearForm mass(fespace.get());
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();
    nnz.SetSize(nLevels);
    nnz[0] = 0;
    for(int i = 1; i < nLevels; ++i)
    {
        nnz[i] = 0;
        level_size[i] = sequence[i]->GetNumberOfDofs(nDim);
        evect[i] = new DenseMatrix(level_size[i], totnmodes );
        sequence[i-1]->GetPi(nDim)->ComputeProjector();
        Mult(sequence[i-1]->GetPi(nDim)->GetProjectorMatrix(), 
                *(evect[i-1]), *(evect[i]) );
        /*
        // Unit scaling of ews
        for (int j = 0; j < totnmodes; j++)
        {
            Vector v;
            evect[i]->GetColumn(j, v);
            double s = 1./mass.InnerProduct(v,v);
            v *= sqrt(s);
        }
        */
    }
}

void KLSampler::Sample(int level, Vector & xi)
{
    xi.SetSize(level_size[level]);
    dist_sampler(xi);
}

void KLSampler::Eval(int level, const Vector & xi, Vector & s)
{
    
    int ndofs( level_size[level]);//evect[level]->Height() );
    s.SetSize( ndofs );
    s = 0.;

    double coeff(0);
    Vector v;
//    if (totnmodes > level_size[level])
//        totnmodes = level_size[level];
    totnmodes = eval->Size(); //level_size[level];
    for(int i(0); i < totnmodes; ++i)
    {
        coeff = sqrt((*eval)(i)) * xi(i);
        evect[level]->GetColumnReference(i, v);
        s.Add(coeff, v);
    }

    if(lognormal)
    {
        for(int i(0); i < ndofs; ++i)
            s(i) = exp(s(i));
    }
}

double KLSampler::ComputeL2Error(
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

double KLSampler::ComputeMaxError(
    int level,
    const Vector & coeff,
    double exact) const
{
    const double e1 = coeff.Max() - exact;
    const double e2 = exact - coeff.Min();
    return std::max(e1, e2);
}

void KLSampler::SaveMeshGLVis(const std::string prefix) const
{
     std::ostringstream mesh_name;
     mesh_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

     std::ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     mesh->Print(mesh_ofs);

}

void KLSampler::SaveFieldGLVis(int level, const Vector & coeff, 
        const std::string prefix) const
{
    std::ostringstream fid_name;
    fid_name << prefix << "_L" << std::setfill('0') << std::setw(2) << 
            level << "." << std::setw(6) << myid; 
        
    GridFunction x;
    prolongate_to_fine_grid(level, coeff, x);
 
    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);

    // Printing to VTK file format
    std::ostringstream v_name;
    v_name << prefix << "_L" << std::setfill('0') << std::setw(2) << 
            level << ".vtk";

    std::ofstream vid(v_name.str().c_str());
    vid.precision(8);
    mesh->PrintVTK(vid);
    x.SaveVTK(vid, "value", 0);

}

void KLSampler::prolongate_to_fine_grid(int ilevel, 
        const Vector & coeff, 
        GridFunction & x) const 
{
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );        
    for(int lev(ilevel); lev > 0; --lev)                                       
    {                                                                          
        auto help = make_unique<Vector>( P[lev-1]->Height() );                
        P[lev-1]->Mult(*feCoeff, *help);                                     
        feCoeff = std::move(help);                                             
    }                                                                          
                                                                               
    x.MakeRef(fespace.get(), *feCoeff, 0);                                           
                                                                               
    if(ilevel!=0)                                                              
    {                                                                          
        x.MakeDataOwner();                                                     
        feCoeff->StealData();                                                  
    }          
}

void KLSampler::glvis_plot(int ilevel, 
        Vector & coeff, 
        std::string prefix, 
        std::ostream & socket)
{
    GridFunction x;
    prolongate_to_fine_grid(ilevel, coeff, x);
    socket << "parallel " << num_procs << " " << myid << "\n";
    socket.precision(8);
    socket << "solution\n" << *mesh << x << std::flush
           << "window_title '" << prefix << " Level " << ilevel << "'" << std::endl;;
    MPI_Barrier(mesh->GetComm());
}

void KLSampler::SaveVTK(Mesh * mesh, GridFunction & coeff, std::string prefix) const
{
    std::ostringstream name;
    name << prefix << ".vtk";
    std::ofstream ofs(name.str().c_str());
    ofs.precision(8);
    mesh->PrintVTK(ofs);
    coeff.SaveVTK(ofs, "KLE" ,0);
}

KLSampler::~KLSampler()
{
    for(int i(1); i < evect.Size(); ++i)
        delete evect[i];
}

} /* namespace parelagmc */
