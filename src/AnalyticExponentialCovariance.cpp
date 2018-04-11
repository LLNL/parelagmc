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
 
#include <fstream>
#include <iostream>

#include "AnalyticExponentialCovariance.hpp"

namespace parelagmc 
{
using namespace mfem;

AnalyticExponentialCovariance::AnalyticExponentialCovariance(
            const std::shared_ptr<mfem::ParMesh>& mesh_,
            parelag::ParameterList& master_list_):
        mesh(mesh_),
        ndim(mesh->Dimension()),
        prob_list(master_list_.Sublist("Problem parameters", true)),
        nmodes(prob_list.Get("Number of modes", std::vector<int>(10, 10))),
        size(mesh->GetNE()),
        var(prob_list.Get("Variance", 1.0)),
        domain_lengths(prob_list.Get("Domain lengths", std::vector<double>(1., 1.))),
        totnmodes(1),
        fec(nullptr),
        fespace(nullptr),
        measD(1.)
{
    // Create the fespace 
    fec = parelag::make_unique<L2_FECollection>(0, ndim);
    fespace = parelag::make_unique<FiniteElementSpace>(mesh.get(), fec.get());

    correlation_lengths.SetSize(ndim);
    correlation_lengths = prob_list.Get("Correlation length", 0.1);
    
    for(int idim(0); idim < ndim; ++idim)
        totnmodes *= nmodes[idim];

    // TODO: Correct this instead of throwing?
    PARELAG_ASSERT(totnmodes <= size);
    //if (totnmodes > size)
    //    totnmodes = size;
}

void AnalyticExponentialCovariance::SolveEigenvalue() 
{
    eval.SetSize(totnmodes);
    evect.SetSize(size, totnmodes);

    computeEigs(eval, evect);
    std::cout << "Sum of eigenvalues: " << eval.Sum() << std::endl;
}

void AnalyticExponentialCovariance::ShowMe(std::ostream & os)
{
    os << "KL Expansion: \n";
    os << "Nmodes: ";
    for(int i(0); i <ndim; ++i)
        os << std::setw(5) << nmodes[i];
    os << "  (" << totnmodes << ")\n";

    os << "Correlation lengths: ";
    for(int i(0); i <ndim; ++i)
        os << std::setw(5) << correlation_lengths(i);
    os << "\n";

    os << "Domain length: ";
    for(int i(0); i <ndim; ++i)
        os << std::setw(5) << domain_lengths[i];
    os << "\n";

    double sum_eigs(0);
    for(double * it = eval.GetData(); it != eval.GetData()+eval.Size(); ++it)
        sum_eigs += *it;
    sum_eigs /= measD;

    os << "Fraction of variability captured by the Truncated KL expansion: " 
            << sum_eigs << "\n";
    os << "Saving ews/evs to Analytic_Eigenvalues.dat, Analytic_Eigenvector.dat" << "\n";
    std::ofstream EVecout ("Analytic_Eigenvector.dat");
    evect.PrintMatlab(EVecout);

    std::ofstream Evalout ("Analytic_Eigenvalues.dat");
    eval.Print(Evalout,1);
}

void AnalyticExponentialCovariance::CheckOrthogonalityEigenvectors(
        std::ostream & os)
{
    Vector s( evect.Width() );
    evect.SingularValues( s );

    os << "Minimum singular vector in the eigenvalue matrix is " << 
            s(s.Size() - 1) << "\n";
}

void AnalyticExponentialCovariance::SaveVTK(Mesh * mesh, 
        GridFunction & coeff, 
        const std::string prefix) const 
{
    std::ostringstream name;
    name << prefix << ".vtk";
    std::ofstream ofs(name.str().c_str());
    ofs.precision(8);
    mesh->PrintVTK(ofs);
    coeff.SaveVTK(ofs, "KLE" ,0);

}

void AnalyticExponentialCovariance::computeEigs(Vector & eval, 
        DenseMatrix & evect)
{
    BilinearForm mass(fespace.get());
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();

    {
        ConstantCoefficient one(1.);
        GridFunction tmp(fespace.get());
        tmp.ProjectCoefficient(one);
        measD = mass.InnerProduct(tmp, tmp);
    }

    if(ndim==1)
    {
        const int coor = 0;
        double scaled_corr_len = correlation_lengths(coor)/domain_lengths[coor];
        Vector omega(totnmodes);
        computeOmega(totnmodes, scaled_corr_len, omega);
        computeEigenvalues1d(omega, scaled_corr_len, 
                domain_lengths[coor], eval);
        computeEigenvectors1d(coor, omega, scaled_corr_len, 
                domain_lengths[coor], mass.SpMat(), evect);
    }
    else
    {    
        int sum_of_modes = 0;
        for(int idim=0; idim < ndim; ++idim)
            sum_of_modes += nmodes[idim];

        double * omegas  = new double[sum_of_modes];
        double * eval1ds    = new double[sum_of_modes];
        double * evect1ds = new double[sum_of_modes*size];

        double * it_omega   = omegas;
        double * it_eval1d  = eval1ds;
        double * it_evect1d = evect1ds;

        Vector omega1d;
        Vector eval1d;
        DenseMatrix evect1d;

        for(int coor=0; coor < ndim; ++coor)
        {
            omega1d.SetDataAndSize(it_omega, nmodes[coor]);
            eval1d.SetDataAndSize(it_eval1d, nmodes[coor]);
            evect1d.UseExternalData(it_evect1d, size, nmodes[coor]);

            double scaled_corr_len = 
                    correlation_lengths(coor)/domain_lengths[coor];
            computeOmega(nmodes[coor], scaled_corr_len, omega1d);
            computeEigenvalues1d(omega1d, scaled_corr_len, 
                    domain_lengths[coor], eval1d);
            computeEigenvectors1d(coor, omega1d, scaled_corr_len, 
                    domain_lengths[coor], mass.SpMat(), evect1d);
            it_omega  += nmodes[coor];
            it_eval1d += nmodes[coor];
            evect1d.ClearExternalData();
            it_evect1d += size*nmodes[coor];
        }

        if(ndim == 2)
        {
            double * it_eval = eval.GetData();
            double * it_evect = evect.Data();

            for (int i=0; i<nmodes[0]; i++)
                for (int j=0; j<nmodes[1]; j++)
                {
                    *(it_eval++) = eval1ds[i]*eval1ds[j+nmodes[0]];
                    Vector ev(it_evect, size);
                    for(int is(0); is < size; ++is)
                        *(it_evect++) = evect1ds[i*size+is] * evect1ds[(j+nmodes[0])*size+is];
                    double s2 = 1. / mass.InnerProduct(ev,ev);
                    ev *= sqrt(s2);
                }
        }
        else if (ndim == 3)
        {
            double * it_eval = eval.GetData();
            double * it_evect = evect.Data();

            for (int i=0; i<nmodes[0]; i++)
                for (int j=0; j<nmodes[1]; j++)
                    for (int k=0; k<nmodes[2]; k++)
                {
                    *(it_eval++) = eval1ds[i]*eval1ds[j+nmodes[0]]
                            *eval1ds[k+nmodes[0]+nmodes[1]];
                    Vector ev(it_evect, size);
                    for(int is(0); is < size; ++is)
                        *(it_evect++) = evect1ds[i*size+is] * 
                                evect1ds[(j+nmodes[0])*size+is] * 
                                evect1ds[(k+nmodes[0]+nmodes[1])*size+is];
                    double s2 = 1. / mass.InnerProduct(ev,ev);
                    ev *= sqrt(s2);
                }
        }
        delete[] omegas;
        delete[] eval1ds;
        delete[] evect1ds;
    }
}
// Computing the (real) solutions of the trancendental equation 
// tan(omega) = 2*lambda*omega/(lambda*lambda*omega*omega-1) where lambda is the correlation length 
void AnalyticExponentialCovariance::computeOmega(int nmodes, 
        double scaled_corr_length, 
        Vector & omega)
{
    omega.SetSize(nmodes);
    const int maxit = 1000;
    const double tol = 0.00001;

    double asyx = 1.0 / scaled_corr_length;
    double * xlvec = new double[nmodes+2];

    int ctr = 0;

    xlvec[0] = M_PI/2.0;
    if (asyx < M_PI/2.0)
    {
        xlvec[0] = asyx;
        xlvec[1] = M_PI/2.0;
        ctr++;
    }

    while ( ctr<(nmodes+1) )
    {
            ctr++;
            xlvec[ctr] = xlvec[ctr-1] + M_PI;
            if ((asyx<xlvec[ctr])&&(asyx>xlvec[ctr-1])){
                xlvec[ctr] = asyx;
                ctr++;
                xlvec[ctr] = xlvec[ctr-2] + M_PI;
            }
    }
    double xl, xr, xm, fl, fr, fm;
    for ( int j=0; j<nmodes; j++ )
    {
        xl = 1.001*xlvec[j];
        xr = 0.999*xlvec[j+1];
        xm = (xl+xr)/2.0;
        fl = tan(xl)-(2.0*scaled_corr_length*xl)/
                (scaled_corr_length*scaled_corr_length*xl*xl-1.0);
        fr = tan(xr)-(2.0*scaled_corr_length*xr)/
                (scaled_corr_length*scaled_corr_length*xr*xr-1.0);
        fm = tan(xm)-(2.0*scaled_corr_length*xm)/
                (scaled_corr_length*scaled_corr_length*xm*xm-1.0);

        int it=0;
        while ((fabs(fm)>tol)&&(it<maxit)){
            xm = (xl+xr)/2.0;
            fm = tan(xm)-(2.0*scaled_corr_length*xm)/
                    (scaled_corr_length*scaled_corr_length*xm*xm-1.0);
            if ((fl*fm)<0)
                xr=xm;
            else
                xl=xm;
            fl = tan(xl)-(2.0*scaled_corr_length*xl)/
                    (scaled_corr_length*scaled_corr_length*xl*xl-1.0);
            fr = tan(xr)-(2.0*scaled_corr_length*xr)/
                    (scaled_corr_length*scaled_corr_length*xr*xr-1.0);
            it++;
        }

        omega(j) = xm;
    }

    delete[] xlvec;
}

void AnalyticExponentialCovariance::computeEigenvalues1d(
        const Vector & omega, 
        double scaled_corr_length, 
        double lx, 
        Vector & eval1d)
{
    int nmodes = omega.Size();
    eval1d.SetSize(nmodes);
    for(int i(0); i < nmodes; ++i)
        eval1d(i) = 2.*lx*scaled_corr_length / 
                (scaled_corr_length*scaled_corr_length*omega(i)*omega(i) + 1.);
}

void AnalyticExponentialCovariance::computeEigenvectors1d(
        int coord, 
        const Vector & omega, 
        double scaled_corr_length, 
        double lx, 
        const SparseMatrix & mass, 
        DenseMatrix & evect1d)
{
    int nmodes = omega.Size();
    int nvals  = fespace->GetNDofs();

    evect1d.SetSize(nvals, nmodes);

    AnalyticExponentialEvect1dCoefficient c(coord, scaled_corr_length, lx);
    GridFunction gf;
    Vector col_view;
    double Aj;

    for(int j(0); j < nmodes; ++j)
    {
        evect1d.GetColumnReference(j,col_view);
        gf.MakeRef(fespace.get(), col_view, 0);
        c.SetOmega( omega(j) );
        gf.ProjectCoefficient(c);
        //Scaling so that norm(evect1d_j)=1
        Aj = 1./sqrt( mass.InnerProduct(gf,gf) );
        gf *= Aj;
    }

}

//----------------------------------------------------------------------------//
AnalyticExponentialEvect1dCoefficient::AnalyticExponentialEvect1dCoefficient(
        int coord_, 
        double lambda_, 
        double lx_, 
        double omega_n_):
    coord(coord_),
    lambda(lambda_),
    omega_n(omega_n_),
    lx(lx_)
{

}

void AnalyticExponentialEvect1dCoefficient::SetOmega(double omega_n_)
{
    omega_n = omega_n_;
}

double AnalyticExponentialEvect1dCoefficient::Eval(ElementTransformation &T, 
        const IntegrationPoint &ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    double xc = x[coord] * omega_n / lx;

    return ( sin(xc) + lambda*omega_n*cos(xc) )/lx;
}

} /* namespace parelagmc */
