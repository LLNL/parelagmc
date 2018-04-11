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
 
#include "MaternCovariance.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Utilities.hpp"

namespace parelagmc
{
using namespace mfem;
using namespace parelag;

double xfun(Vector &v)
{
    return v(0);
}
double yfun(Vector &v)
{
    return v(1);
}
double zfun(Vector &v)
{
    return v(2);
}

MaternCovariance::MaternCovariance(
            const std::shared_ptr<mfem::ParMesh>& mesh_, 
            parelag::ParameterList& master_list_):
        lobpcg(false),
        mesh(mesh_),
        nDim(mesh->Dimension()),
        prob_list(master_list_.Sublist("Problem parameters", true)),            
        corlen(prob_list.Get("Correlation length", 0.1)),
        kappa(1./corlen),
        nu(2. - static_cast<double>(nDim)/2.),
        scale(1./(std::tgamma(nu)*std::pow(2.,nu-1.))),
        point_coords_exist(false),
        cov_matrix_exist(false),
        fec(nullptr),
        fespace(nullptr)
{
    MPI_Comm comm = mesh->GetComm();
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);

    // Create the fespace
    fec = parelag::make_unique<L2_FECollection>(0, nDim);
    fespace = parelag::make_unique<ParFiniteElementSpace>(mesh.get(), fec.get());
    ndofs = fespace->GetNDofs();

    std::vector<int> nmodes = prob_list.Get("Number of modes", std::vector<int>(10, 10));
    totnmodes = nmodes[0];

    PointCoordinates.SetSize(ndofs,nDim);

    ConstantCoefficient one(1.0);
    massL2 = parelag::make_unique<ParBilinearForm>(fespace.get());
    massL2->AddDomainIntegrator(new MassIntegrator(one));
    massL2->Assemble();
    massL2->Finalize();

    if (totnmodes > ndofs)
        totnmodes = ndofs;

}

void MaternCovariance::ShowMe(std::ostream & os)
{
    os << "Matern Covariance -- KL Expansion: \n";
    os << "Number of modes: " << totnmodes << "\n";
    os << "Correlation length: " << corlen << "\n";

    double sum_eigs(0);
    for(double * it = evals.GetData(); it != evals.GetData()+evals.Size(); ++it)
        sum_eigs += *it;
    
    BilinearForm mass(fespace.get());
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();
    
    {
        ConstantCoefficient one(1.);
        GridFunction tmp(fespace.get());
        tmp.ProjectCoefficient(one);
        double measD = mass.InnerProduct(tmp, tmp);
        sum_eigs /= measD;
    }

    os << "Fraction of variability captured by the Truncated KL expansion: "
            << sum_eigs << "\n";
    os << "Saving ews/evs to Matern_Eigenvalues.dat, Matern_Eigenvector.dat" << "\n";
    std::ofstream EVecout ("Matern_Eigenvector.dat");
    evect.PrintMatlab(EVecout);

    std::ofstream Evalout ("Matern_Eigenvalues.dat");
    evals.Print(Evalout,1);

}

void MaternCovariance::GenerateCovarianceMatrix()
{
    // make sure PointCoordinates has been computed
    if (!point_coords_exist)
        GeneratePointCoordinates();

    Matern.SetSize(ndofs,ndofs);
    Vector x(nDim), y(nDim);

    for (int i=0; i < ndofs; ++i)
    {
        Matern(i,i)=1.;
        for (int l=0; l < nDim; ++l)
            x(l)=PointCoordinates(i,l);

        for (int j=0; j < i; ++j)
        {
            for (int l=0; l < nDim; ++l)
                y(l)=PointCoordinates(j,l);

            Matern(i,j)=Matern(j,i)=Compute(x,y);
        }
    }
    massL2->SpMat().GetDiag(massL2diag);

    Matern.LeftScaling(massL2diag);
    Matern.RightScaling(massL2diag);
    cov_matrix_exist = true;
}

void MaternCovariance::GenerateCovarianceMatrix(SparseMatrix &P, int version)
{
    switch (version)
    {
        case 1: generateCovarianceMatrix_v1(P);
        break;
        case 2: generateCovarianceMatrix_v2(P);
        break;
        default: mfem_error("Not valid version");
    }
}


void MaternCovariance::generateCovarianceMatrix_v1(SparseMatrix &P)
{
    SparseMatrix *WP = Mult(massL2->SpMat(),P);
    SparseMatrix * Q = Transpose(*WP);
    SparseMatrix * Wc=Mult(*Q,P);
    Wc->GetDiag(massL2diag);

    int * i_Q = Q->GetI();
    int * j_Q = Q->GetJ();
    double * val_Q = Q->GetData();

    int n = Q->Size();
    Matern.SetSize(n);

    Array<int> i_index;
    Array<int> j_index;

    for (int i=0; i <n; ++i)
    {
        i_index.MakeRef(j_Q+i_Q[i],i_Q[i+1]-i_Q[i]);

        Vector Q_i(val_Q+i_Q[i],i_Q[i+1]-i_Q[i]);
        Matern(i,i)=ComputeCoarseCovarianceMatrixEntry(i_index,i_index,Q_i,Q_i);

        for (int j=0; j<i; ++j)
        {

            j_index.MakeRef(j_Q+i_Q[j],i_Q[j+1]-i_Q[j]);
            Vector Q_j(val_Q+i_Q[j],i_Q[j+1]-i_Q[j]);
            Matern(i,j)=Matern(j,i)=ComputeCoarseCovarianceMatrixEntry(i_index,j_index,Q_i,Q_j);

        }
    }

    delete Wc;
    delete Q;
    delete WP;
}

void MaternCovariance::generateCovarianceMatrix_v2(SparseMatrix &P)
{
    SparseMatrix *Pt = Transpose(P);
    SparseMatrix *Pi = Mult(*Pt,massL2->SpMat());
    SparseMatrix * Wc=Mult(*Pi,P);

    Wc->GetDiag(massL2diag);
    for (int i=0; i < massL2diag.Size(); ++i)
        Pi->ScaleRow(i,1./massL2diag(i));

    DenseMatrix cPointCoordinates(massL2diag.Size(),nDim);
    Mult(*Pi,PointCoordinates,cPointCoordinates);

    Matern.SetSize(cPointCoordinates.Height(),cPointCoordinates.Height());
    Vector x(nDim), y(nDim);
    for (int i=0; i < cPointCoordinates.Height(); ++i)
    {
        Matern(i,i)=1.;
        for (int l=0; l < nDim; ++l)
            x(l)=cPointCoordinates(i,l);

        for (int j=0; j < i; ++j)
        {
            for (int l=0; l < nDim; ++l)
                y(l)=cPointCoordinates(j,l);

            Matern(i,j)=Matern(j,i)=Compute(x,y);
        }
    }

    Matern.LeftScaling(massL2diag);
    Matern.RightScaling(massL2diag);

    delete Wc;
    delete Pi;
    delete Pt;
}


void MaternCovariance::GenerateCovarianceMatrix(
        Array<int> &row_index,
        Array<int> &col_index,
        DenseMatrix &G)
{
    Vector x(nDim), y(nDim);

    for (int i=0; i < row_index.Size(); ++i)
    {
        for (int l=0; l < nDim; ++l)
            x(l)=PointCoordinates(row_index[i],l);

        for (int j=0; j < col_index.Size(); ++j)
        {
            for (int l=0; l < nDim; ++l)
                y(l)=PointCoordinates(col_index[j],l);

            G(i,j)=Compute(x,y);
        }
    }
}

double MaternCovariance::ComputeCoarseCovarianceMatrixEntry(
        Array<int> &row_index,
        Array<int> &col_index,
        Vector &weights_row, 
        Vector &weights_col)
{
    Vector tmp(row_index.Size());
    tmp=0.0;
    Vector x(nDim), y(nDim);

    for (int i=0; i < row_index.Size(); ++i)
    {
        for (int l=0; l < nDim; ++l)
            x(l)=PointCoordinates(row_index[i],l);

        for (int j=0; j < col_index.Size(); ++j)
        {
            for (int l=0; l < nDim; ++l)
                y(l)=PointCoordinates(col_index[j],l);

            tmp(i)+=Compute(x,y)*weights_col(j);
        }
    }
    return tmp*weights_row;
}


void MaternCovariance::GeneratePointCoordinates()
{
    GridFunction k1;
    k1.SetSpace(fespace.get());
    FunctionCoefficient x(xfun);
    FunctionCoefficient y(yfun);
    FunctionCoefficient z(zfun);

    k1.ProjectCoefficient(x);
    for (int j=0; j < ndofs; ++j)
    {
        PointCoordinates(j,0)=k1(j);
    }

    k1.ProjectCoefficient(y);
    for (int j=0; j < ndofs; ++j)
    {
        PointCoordinates(j,1)=k1(j);
    }
    if (nDim ==3){
        k1.ProjectCoefficient(z);
        for (int j=0; j < ndofs; ++j)
        {
            PointCoordinates(j,2)=k1(j);
        }
    }
}

void MaternCovariance::SolveEigenvalue()
{
    // check Matern has been filled
    if (!cov_matrix_exist)
        GenerateCovarianceMatrix();
    evect.SetSize(Matern.Size(), totnmodes);
    evals.SetSize(totnmodes);
    if (lobpcg)
        solveEigenvalueLOBPCG();
    else 
        solveEigenvalueSymEigensolver();
}    

void MaternCovariance::solveEigenvalueSymEigensolver()
{
    SymEigensolver EigSolve;

    EigSolve.SetOverwrite(true);
    //EigSolve.ComputeDiagonalAll(Matern, massL2diag, evals, evect);
    std::vector<double> eigenvalues;
    eigenvalues.resize(evals.Size());
    EigSolve.ComputeDiagonalFixedNumber(Matern, massL2diag, eigenvalues, evect, 
        Matern.Size() - totnmodes +1, Matern.Size());
    ParBilinearForm mass(fespace.get());
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();

    // FIXME: This is silly!
         
    for (int i = 0; i < totnmodes; i++)
    {
        Vector v;
        evect.GetColumn(i, v);
        double s = 1./mass.InnerProduct(v,v);
        v *= sqrt(s);
        // FIXME: This is silly!
        evals(i) = eigenvalues[i];
    }

    std::cout << "SymEigenSolver" << "\n";
    std::cout << "Sum of eigenvalues: " << evals.Sum() << "\n";
    std::cout << "Total Volume: " << massL2diag.Sum() << "\n";
}

void MaternCovariance::solveEigenvalueLOBPCG()
{
    SparseMatrix Sp_Matern( Matern.Height(), Matern.Width());
    for (int i = 0; i < Matern.Height(); i++)
    {
        for (int j = 0; j < Matern.Width(); j ++)
        {
            Sp_Matern.Set(i,j, -1.*Matern(i,j));
        }
    }
    Sp_Matern.Finalize();
    SharingMap * map = new SharingMap(mesh->GetComm());
    map->SetUp(*mesh, 0);
    std::unique_ptr<HypreParMatrix>  A = Assemble(*map, Sp_Matern, *map); 
    std::unique_ptr<HypreParMatrix>  M = Assemble(*map, massL2->SpMat(), *map);
    
    HypreLOBPCG * lobpcg = new HypreLOBPCG(mesh->GetComm());
    HypreSolver *    amg = new HypreBoomerAMG(*A);

    lobpcg->SetNumModes(totnmodes);
    lobpcg->SetPreconditioner(*amg);
    lobpcg->SetMaxIter(100);
    lobpcg->SetTol(1e-6);
    lobpcg->SetPrecondUsageMode(1);
    lobpcg->SetPrintLevel(0);
    lobpcg->SetMassMatrix(*M);
    lobpcg->SetOperator(*A);

    // Obtain the eigenvalues and eigenvectors
    lobpcg->Solve();
    Array<double> eigenvalues;
    lobpcg->GetEigenvalues(eigenvalues);
    for (int i = 0; i < totnmodes; i++)
    {
        evals[i] = -1.*eigenvalues[i];
        HypreParVector x = lobpcg->GetEigenvector(i);
        ParGridFunction g(fespace.get());
        g = lobpcg->GetEigenvector(i);
        Vector val;
        g.GetNodalValues(val);
        for (int j = 0; j < val.Size(); j++)
        {
            evect(j,i) = -1.*val[j];
        }
    }

    ParBilinearForm mass(fespace.get());
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();

    for (int i = 0; i < totnmodes; i++)
    {
        Vector v;
        evect.GetColumn(i, v);
        double s = 1./mass.InnerProduct(v,v);
        v *= sqrt(s);
    }

    std::cout << "LOBPCG" << "\n";
    std::cout <<  "Sum of eigenvalues: " << evals.Sum() << "\n";
    std::cout << "Total Volume: " << massL2diag.Sum() << "\n";

}

double MaternCovariance::ComputeScalingCoefficient() const
{
    const double gnu=std::tgamma(nu);
    const double gnudim=std::tgamma(nu + .5*static_cast<double>(nDim));
    const double c= std::pow(16.*std::atan(1.0),.5*static_cast<double>(nDim));
    const double k= std::pow(kappa,2.*nu);

    return sqrt(c*gnudim*k/gnu);
}

double MaternCovariance::Compute(const Vector &x, const Vector &y)
{
    double normdiff=0.0;
    for (int l=0; l < x.Size(); ++l)
    {
        normdiff += std::pow(x(l)-y(l),2);
    }
    normdiff = kappa*sqrt(normdiff);
    if (normdiff < 1e-10)
        return 1.;
    else
    {
        if (nu == 0.5)
            return std::exp(-1.*normdiff); 
        else // nu == 1
            return scale*std::pow(std::sqrt(2.*nu)*normdiff,nu)*bessk1(std::sqrt(2.*nu)*normdiff);
    }
}

double MaternKernel::func(Vector &x)
{
    return K->Compute(x,y);
}

Vector MaternKernel::y(0);
MaternCovariance *MaternKernel::K(nullptr);

} /* namespace parelagmc */

