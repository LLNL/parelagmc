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
 
#include "Utilities.hpp"

namespace parelagmc
{

using namespace mfem;
using namespace parelag;

void BuildTopologyGeometric(
    std::shared_ptr<ParMesh> pmesh,
    Array<int> & level_nElements,
    std::vector< std::shared_ptr< AgglomeratedTopology > > & topology)
{
    const int nLevels = topology.size();
    const int nDimensions = pmesh->Dimension();
    const auto AT_elem = AgglomeratedTopology::ELEMENT;
    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    topology[0] = make_unique<AgglomeratedTopology>( pmesh, nDimensions );
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(AT_elem));
        partitioner.Partition(
                topology[ilevel]->GetNumberLocalEntities(AT_elem),
                level_nElements[ilevel+1],
                partitioning);
        // CoarsenLocalPartitioning(Array<int>& partitioning, bool check_topology, bool preserve_material_interfaces
        topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
    }
}

void EmbeddedBuildTopologyGeometric(
    std::shared_ptr<mfem::ParMesh> pmesh,
    std::shared_ptr<mfem::ParMesh> pembedmesh,
    mfem::Array<int> & level_nElements,
    mfem::Array<int> & embed_level_nElements,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & topology,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & embed_topology,
    std::vector<Array<int> > & material_id)
{
    const int nLevels = topology.size();
    const int nDimensions = pmesh->Dimension();
    const auto AT_elem = AgglomeratedTopology::ELEMENT;
    // Using LogicalPartitioner for structured coarsening

    MFEMRefinedMeshPartitioner partitioner(nDimensions);

    std::vector<Array<MFEMMaterialId>> info(nLevels);
    info[0].SetSize(pembedmesh->GetNE());

    LogicalPartitioner lpartitioner;
    std::vector<std::unique_ptr<CoarsenMFEMMaterialId>>
            coarseningOp(nLevels-1);

    embed_topology[0] = make_unique<AgglomeratedTopology>( pembedmesh, nDimensions );

    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> lpartitioning(
                embed_topology[ilevel]->GetNumberLocalEntities(AT_elem));
        coarseningOp[ilevel] = make_unique<CoarsenMFEMMaterialId>( partitioner,
                                        *(embed_topology[ilevel]),
                                        embed_level_nElements[ilevel+1],
                                        info[ilevel]);
        // Setup logical info on fine grid
        if(ilevel == 0)
            coarseningOp[0]->FillFinestMFEMMaterialId(*pembedmesh,info[0]);
        // Generate the metis partitioning
        lpartitioner.Partition<MFEMMaterialId,CoarsenMFEMMaterialId>(
            *(embed_topology[ilevel]->LocalElementElementTable()),
            info[ilevel],
            *(coarseningOp[ilevel]),
            lpartitioning);
        // Build coarser topology based on partitioning
        embed_topology[ilevel+1] = embed_topology[ilevel]->
            CoarsenLocalPartitioning(lpartitioning, 0, 0);
        // Setup logical info for next level
        lpartitioner.ComputeCoarseLogical<MFEMMaterialId,CoarsenMFEMMaterialId>(
            *(coarseningOp[ilevel]),
            embed_topology[ilevel]->AEntityEntity(AT_elem),
            info[ilevel],
            info[ilevel+1]);
    }

    Array<int> orig_num_partitions;
    orig_num_partitions.SetSize(nLevels);
    // FIXME!! a better way!!
    for (int i = 0; i < nLevels; i++)
    {
        int count = 0;
        material_id[i].SetSize(info[i].Size());
        for (int j = 0; j < info[i].Size(); j++)
        {
            (material_id[i])[j] = (info[i])[j].materialId;
            if ((info[i])[j].materialId == 1)
                count++;
        }
        orig_num_partitions[i] = count;
    }
    // original mesh
    topology[0] = make_unique<AgglomeratedTopology>( pmesh, nDimensions );
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(AT_elem));
        partitioner.Partition(
                topology[ilevel]->GetNumberLocalEntities(AT_elem),
                orig_num_partitions[ilevel+1],
                partitioning);
        topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);

    }

}

void BuildTopologyAlgebraic(
    std::shared_ptr<ParMesh> pmesh,
    const int coarsening_factor,
    std::vector< std::shared_ptr< AgglomeratedTopology > > & topology)
{
    const auto AT_elem = AgglomeratedTopology::ELEMENT;
    const int nLevels = topology.size();
    MetisGraphPartitioner partitioner;
    // Specifies BISECTION
    partitioner.setFlags(MetisGraphPartitioner::KWAY);
    // Fix the seed
    partitioner.setOption(METIS_OPTION_SEED, 0);
    // Ask metis to provide contiguous partitions
    partitioner.setOption(METIS_OPTION_CONTIG,1);
    partitioner.setOption(METIS_OPTION_MINCONN,1);
    partitioner.setUnbalanceToll(1.05);
    topology[0] = std::make_shared<AgglomeratedTopology>( pmesh, pmesh->Dimension() );
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(AT_elem));
        int num_partitions = partitioning.Size()/coarsening_factor;
        if (num_partitions == 0) num_partitions = 1;
        partitioner.doPartition( *(topology[ilevel]->LocalElementElementTable()),
                topology[ilevel]->Weight(AT_elem),
                num_partitions,
                partitioning);
        topology[ilevel+1] = topology[ilevel]->   // int checkTopology, int preserveMaterialInterfaces
                CoarsenLocalPartitioning(partitioning, 0, 0);
    }
}

void EmbeddedBuildTopologyAlgebraic(
    std::shared_ptr<mfem::ParMesh> pmesh,
    std::shared_ptr<mfem::ParMesh> pembedmesh,
    const int coarsening_factor,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & topology,
    std::vector< std::shared_ptr< parelag::AgglomeratedTopology > > & embed_topology,
    std::vector<Array<int> > & material_id)
{
    const int nLevels = topology.size();
    const int nDimensions = pmesh->Dimension();
    const auto AT_elem = AgglomeratedTopology::ELEMENT;

    std::vector<Array<MetisMaterialId>> info(nLevels);
    info[0].SetSize(pembedmesh->GetNE());
    MetisGraphPartitioner mpartitioner;
    mpartitioner.setFlags(MetisGraphPartitioner::KWAY ); // BISECTION
    mpartitioner.setOption(METIS_OPTION_SEED, 0);         // Fix the seed
    mpartitioner.setOption(METIS_OPTION_CONTIG,1);        // Ask metis to provide contiguous partitions
    mpartitioner.setOption(METIS_OPTION_MINCONN,1);       //
    mpartitioner.setUnbalanceToll(1.05); //

    LogicalPartitioner lpartitioner;
    std::vector<std::unique_ptr<CoarsenMetisMaterialId>>
            coarseningOp(nLevels-1);

    embed_topology[0] = std::make_shared<AgglomeratedTopology>( pembedmesh, nDimensions );

    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> lpartitioning(
                embed_topology[ilevel]->GetNumberLocalEntities(AT_elem));
        int lnum_partitions = lpartitioning.Size()/coarsening_factor;
        if(lnum_partitions == 0) lnum_partitions = 1;
        coarseningOp[ilevel] = make_unique<CoarsenMetisMaterialId>( mpartitioner,
                                        *(embed_topology[ilevel]),
                                        lnum_partitions,
                                        info[ilevel]);
        // Setup logical info on fine grid
        if(ilevel == 0)
            coarseningOp[0]->FillFinestMetisMaterialId(*pembedmesh,info[0]);
        // Generate the metis partitioning
        lpartitioner.Partition<MetisMaterialId,CoarsenMetisMaterialId>(
            *(embed_topology[ilevel]->LocalElementElementTable()),
            info[ilevel],
            *(coarseningOp[ilevel]),
            lpartitioning);
        // Build coarser topology based on partitioning
        embed_topology[ilevel+1] = embed_topology[ilevel]->
            CoarsenLocalPartitioning(lpartitioning, 0, 0);
        // Setup logical info for next level
        lpartitioner.ComputeCoarseLogical<MetisMaterialId,CoarsenMetisMaterialId>(
            *(coarseningOp[ilevel]),
            embed_topology[ilevel]->AEntityEntity(AT_elem),
            info[ilevel],
            info[ilevel+1]);
    }

    Array<int> orig_num_partitions;
    orig_num_partitions.SetSize(nLevels);
    // FIXME!! Gotta be a better way!!
    for (int i = 0; i < nLevels; i++)
    {
        int count = 0;
        material_id[i].SetSize(info[i].Size());
        for (int j = 0; j < info[i].Size(); j++)
        {
            (material_id[i])[j] = (info[i])[j].materialId;
            if ((info[i])[j].materialId == 1)
                count++;
        }
        orig_num_partitions[i] = count;
    }
    // Use regular Metis partitioner for original mesh
    MetisGraphPartitioner partitioner;
    // Specifies BISECTION
    partitioner.setFlags(MetisGraphPartitioner::KWAY );
    // Fix the seed
    partitioner.setOption(METIS_OPTION_SEED, 0);
    // Ask metis to provide contiguous partitions
    partitioner.setOption(METIS_OPTION_CONTIG,1);
    partitioner.setOption(METIS_OPTION_MINCONN,1);
    partitioner.setUnbalanceToll(1.05);
    topology[0] = std::make_shared<AgglomeratedTopology>( pmesh, nDimensions );
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(AT_elem));
        int num_partitions = partitioning.Size()/coarsening_factor;
        if(num_partitions == 0) num_partitions = 1;
        partitioner.doPartition( *(topology[ilevel]->LocalElementElementTable()),
                topology[ilevel]->Weight(AT_elem),
                orig_num_partitions[ilevel+1],
                partitioning);
        topology[ilevel+1] = topology[ilevel]->   // int checkTopology, int preserveMaterialInterfaces
                CoarsenLocalPartitioning(partitioning, 0, 0);
    }

}

// y = x^out
double expWRegression(const Vector & y, 
        const Vector & x, 
        int skip_n_last)
{
    int n = y.Size()-1-skip_n_last;

    if(n < 1)
        return 0.0;

    Vector logdy(n);
    Vector logdx(n);
    Vector weights(n);

    for(int i(0); i < n; ++i)
        logdy(i) = log( fabs(y(i)/y(i+1)) );

    for(int i(0); i < n; ++i)
        logdx(i) = log( x(i)/x(i+1) );

    for(int i(0); i < n; ++i)
        weights(i) = pow(.5, i);

    DWeightedInnerProduct dot(weights);

    return dot(logdy, logdx) / dot(logdx, logdx);

}

void GeneratePointCoordinates(Mesh *mesh, 
        DenseMatrix &PointCoordinates)
{
    FiniteElementCollection *fecL2 = new L2_FECollection(0,mesh->Dimension());
    FiniteElementSpace *fespaceL2 = new FiniteElementSpace(mesh,fecL2);

    PointCoordinates.SetSize(fespaceL2->GetNDofs(), mesh->Dimension());

    GridFunction k1;
    k1.SetSpace(fespaceL2);
    FunctionCoefficient x(xfun<0>);
    FunctionCoefficient y(xfun<1>);
    FunctionCoefficient z(xfun<2>);

    k1.ProjectCoefficient(x);
    for (int j=0; j < k1.Size(); ++j)
        PointCoordinates(j,0)=k1(j);

    k1.ProjectCoefficient(y);
    for (int j=0; j < k1.Size(); ++j)
        PointCoordinates(j,1)=k1(j);
      
    if (mesh->Dimension() ==3){
        k1.ProjectCoefficient(z);
        for (int j=0; j < k1.Size(); ++j)
            PointCoordinates(j,2)=k1(j);
        
    }

    delete fecL2;
    delete fespaceL2;

}

int FindClosestPointID(Mesh *mesh, 
        Vector &x)
{
    DenseMatrix PointCoordinates;
    GeneratePointCoordinates(mesh, PointCoordinates);
    double r=1000.;
    int out = -1;
    Vector tmp(x.Size());
    for (int i(0); i < PointCoordinates.Height(); ++i)
    {
        for (int coord(0); coord < x.Size(); ++coord)
            tmp(coord)= PointCoordinates(i,coord)- x(coord);
        double myr = tmp.Norml2();
        if (myr < r){
            r = myr;
            out = i;
        }
    }
    return out;
}

HypreParVector * chi_center_of_mass(ParMesh * pmesh)
{
    const int nDimensions = pmesh->Dimension();
    FiniteElementCollection * fec = new L2_FECollection(0, nDimensions);
    ParFiniteElementSpace * fes = new ParFiniteElementSpace(pmesh, fec);

    ConstantCoefficient one_coeff(1.);
    Array< FunctionCoefficient* > xcoeffs(nDimensions);
    xcoeffs[0] = new FunctionCoefficient(xfun<0>);
    xcoeffs[1] = new FunctionCoefficient(xfun<1>);
    if(nDimensions==3)
        xcoeffs[2] = new FunctionCoefficient(xfun<2>);

    ParGridFunction ones(fes);
    ones.ProjectCoefficient(one_coeff);
    HypreParVector * ones_v = ones.GetTrueDofs();

    ParLinearForm average(fes);
    average.AddDomainIntegrator(new DomainLFIntegrator(one_coeff));
    average.Assemble();
    HypreParVector * average_v = average.ParallelAssemble();

    Array< ParGridFunction* > xi(nDimensions);
    Array< HypreParVector* > xi_v(nDimensions);
    const double volume = InnerProduct(average_v, ones_v);
    Vector cm(nDimensions);
    for(int i = 0; i < nDimensions; ++i)
    {
        xi[i] = new ParGridFunction(fes);
        xi[i]->ProjectCoefficient(*xcoeffs[i]);
        xi_v[i] = xi[i]->GetTrueDofs();
        cm(i) = InnerProduct(average_v, xi_v[i])/volume;
    }
    const int ne = ones_v->Size();
    double minimum = 1e10;
    int minid = -1;
    for(int i = 0; i < ne; ++i)
    {
        (*ones_v)(i) = 0.0;
        for(int idim = 0; idim < nDimensions; ++idim)
        {
            double d = cm(idim) - (*xi_v[idim])(i);
            (*ones_v)(i) += d*d;
        }
        (*ones_v)(i) = sqrt((*ones_v)(i));
        if((*ones_v)(i) < minimum)
        {
            minimum = (*ones_v)(i);
            minid = i;
        }
    }
    double gmin = minimum;
    MPI_Allreduce(&minimum, &gmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());

    (*ones_v) = 0.0;
    if(minimum == gmin)
        (*ones_v)(minid) = 1.;

    for(int i = 0; i < nDimensions; ++i)
    {
        delete xi[i];
        delete xi_v[i];
        delete xcoeffs[i];
    }
    delete average_v;
    delete fes;
    delete fec;

    return ones_v;
}

double dot(const Vector & a, 
        const Vector & b, 
        MPI_Comm comm)
{
    double ldot = a*b;
    double gdot = 0.;
    MPI_Allreduce(&ldot, &gdot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return gdot;
}

double squared_dot(const Vector & a, 
        const Vector & b, 
        MPI_Comm comm)
{
    Vector a2(a.Size());
    for (int i=0; i<a.Size();i++)
        a2(i) = a(i)*a(i);
    double ldot = a2*b;
    double gdot = 0.;
    MPI_Allreduce(&ldot, &gdot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return gdot;
}

double sum(const Vector & a, 
        MPI_Comm comm)
{
    double lsum = a.Sum();
    double gsum = 0.;
    MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);

    return gsum;
}
void OutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Darcy Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Ndofs_l";
        ndofs_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Ndofs_g";
        ndofs_g.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Darcy_NNZ";
        nnz.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Darcy_Iters";
        iters.Print(std::cout);
    }
}

void OutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& nnz)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Darcy Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Ndofs_l";
        ndofs_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Ndofs_g";
        ndofs_g.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Darcy_NNZ";
        nnz.Print(std::cout);
    }
}

void OutputDofsInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Darcy Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Ndofs_l";
        ndofs_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Ndofs_g";
        ndofs_g.Print(std::cout);
    }
}

void OutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Sampler Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Stoch_size_l";
        stoch_size_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_g";
        stoch_size_g.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Sampler_NNZ";
        nnz.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Sampler_Iters";
        iters.Print(std::cout);
    }
}

void OutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Sampler Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Stoch_size_l";
        stoch_size_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_g";
        stoch_size_g.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Sampler_NNZ";
        nnz.Print(std::cout);
    }
}

void OutputStochInfo(const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Sampler Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Stoch_size_l";
        stoch_size_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_g";
        stoch_size_g.Print(std::cout);
    }
}

void OutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz,
                const mfem::Vector& iters)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Sampler Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Ndofs_l";
        ndofs_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Ndofs_g";
        ndofs_g.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_l";
        stoch_size_l.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_g";
        stoch_size_g.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "NNZ";
        nnz.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "Iters_avg";
        iters.Print(std::cout);        
    }
}

void OutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g,
                const mfem::Vector& nnz)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Sampler Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Ndofs_l";
        ndofs_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Ndofs_g";
        ndofs_g.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_l";
        stoch_size_l.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_g";
        stoch_size_g.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "NNZ";
        nnz.Print(std::cout);        
    }
}

void OutputBothInfo(const mfem::Vector& ndofs_l,
                const mfem::Vector& ndofs_g,
                const mfem::Vector& stoch_size_l,
                const mfem::Vector& stoch_size_g)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    int total_width = 79;
    int name_width = 40;

    if (!myid)
    {
        std::cout << std::fixed<< std::endl;
        std::cout << std::string(total_width, '=') << std::endl;
        std::cout << "Sampler Dofs Statistics:" << std::endl
            << std::string(total_width, '-') << std::endl
            << std::setw(name_width+2) << std::left << "Ndofs_l";
        ndofs_l.Print(std::cout);
        std::cout << std::setw(name_width+2) << std::left << "Ndofs_g";
        ndofs_g.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_l";
        stoch_size_l.Print(std::cout);        
        std::cout << std::setw(name_width+2) << std::left << "Stoch_size_g";
        stoch_size_g.Print(std::cout);        
    }
}

void OutputRandomFieldErrors(const mfem::Vector& exp_errors_L2,
                             const mfem::Vector& var_errors_L2)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);
    const int nLevels = exp_errors_L2.Size();

    if (myid == 0)
    {
        std::cout << "|| E[u] - Ex ||   || V[u] - Ex ||" <<  std::endl;
        for (int i = 0; i < nLevels; i++)
        {
            std::cout << std::scientific << std::setprecision(6)
                    << exp_errors_L2[i] << "  " << std::setw(17) <<
                    var_errors_L2[i] << '\n';
        }
    }
}

void ReduceAndOutputRandomFieldErrors(const mfem::Vector& exp_errors_L2_,
                                      const mfem::Vector& var_errors_L2_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);
    int nLevels = exp_errors_L2_.Size();
    Vector exp_errors_L2(nLevels);
    Vector var_errors_L2(nLevels);

    MPI_Reduce(exp_errors_L2_.GetData(), exp_errors_L2.GetData(),
               exp_errors_L2.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(var_errors_L2_.GetData(), var_errors_L2.GetData(),
               var_errors_L2.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        auto self_sqrt = [](double& a){a = std::sqrt(a);};
        std::for_each(                                                         
            exp_errors_L2.GetData(),                                                
            exp_errors_L2.GetData()+exp_errors_L2.Size(),       
            self_sqrt);                                                        
        std::for_each(                                                         
            var_errors_L2.GetData(), 
            var_errors_L2.GetData()+var_errors_L2.Size(),         
            self_sqrt);                                       
    }

    OutputRandomFieldErrors(exp_errors_L2,
                            var_errors_L2);
}

void ReduceAndOutputRandomFieldErrorsMax(const mfem::Vector& exp_errors_L2_,
                                      const mfem::Vector& var_errors_L2_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);
    int nLevels = exp_errors_L2_.Size();
    Vector exp_errors_L2(nLevels);
    Vector var_errors_L2(nLevels);

    MPI_Reduce(exp_errors_L2_.GetData(), exp_errors_L2.GetData(),
               exp_errors_L2.Size(), MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(var_errors_L2_.GetData(), var_errors_L2.GetData(),
               var_errors_L2.Size(), MPI_DOUBLE, MPI_MAX, 0, comm);

    OutputRandomFieldErrors(exp_errors_L2,
                            var_errors_L2);
}

void ReduceAndOutputDofsInfo(const mfem::Vector& ndofs_l_,
                         const mfem::Vector& ndofs_g_,
                         const mfem::Vector& nnz_,
                         const mfem::Vector& iters_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = iters_.Size();
    Vector ndofs_l(nLevels);
    MPI_Reduce(ndofs_l_.GetData(), ndofs_l.GetData(),
            ndofs_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (myid == 0)
    {
        // Get averages
        std::transform(ndofs_l.GetData(), ndofs_l.GetData()+ndofs_l.Size(),
                ndofs_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }
    OutputDofsInfo(ndofs_l, ndofs_g_, nnz_, iters_);
}

void ReduceAndOutputDofsInfo(const mfem::Vector& ndofs_l_,
                         const mfem::Vector& ndofs_g_,
                         const mfem::Vector& nnz_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = ndofs_l_.Size();
    Vector ndofs_l(nLevels);
    MPI_Reduce(ndofs_l_.GetData(), ndofs_l.GetData(),
        ndofs_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        // Get averages
        std::transform(ndofs_l.GetData(), ndofs_l.GetData()+ndofs_l.Size(),
                ndofs_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }

    OutputDofsInfo(ndofs_l, ndofs_g_, nnz_);
}

void ReduceAndOutputDofsInfo(const mfem::Vector& ndofs_l_,
                         const mfem::Vector& ndofs_g_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = ndofs_l_.Size();
    Vector ndofs_l(nLevels);
    MPI_Reduce(ndofs_l_.GetData(), ndofs_l.GetData(),
        ndofs_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        // Get averages
        std::transform(ndofs_l.GetData(), ndofs_l.GetData()+ndofs_l.Size(),
                ndofs_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }

    OutputDofsInfo(ndofs_l, ndofs_g_);
}

void ReduceAndOutputStochInfo(const mfem::Vector& stoch_size_l_,
                         const mfem::Vector& stoch_size_g_,
                         const mfem::Vector& nnz_,
                         const mfem::Vector& iters_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = iters_.Size();
    Vector stoch_size_l(nLevels);
    MPI_Reduce(stoch_size_l_.GetData(), stoch_size_l.GetData(),
            stoch_size_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (myid == 0)
    {
        // Get averages
        std::transform(stoch_size_l.GetData(), stoch_size_l.GetData()+stoch_size_l.Size(),
                stoch_size_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }
    OutputStochInfo(stoch_size_l, stoch_size_g_, nnz_, iters_);
}

void ReduceAndOutputStochInfo(const mfem::Vector& stoch_size_l_,
                         const mfem::Vector& stoch_size_g_,
                         const mfem::Vector& nnz_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = stoch_size_l_.Size();
    Vector stoch_size_l(nLevels);
    MPI_Reduce(stoch_size_l_.GetData(), stoch_size_l.GetData(),
            stoch_size_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (myid == 0)
    {
        // Get averages
        std::transform(stoch_size_l.GetData(), stoch_size_l.GetData()+stoch_size_l.Size(),
                stoch_size_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }
    OutputStochInfo(stoch_size_l, stoch_size_g_, nnz_);
}

void ReduceAndOutputStochInfo(const mfem::Vector& stoch_size_l_,
                         const mfem::Vector& stoch_size_g_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = stoch_size_l_.Size();
    Vector stoch_size_l(nLevels);
    MPI_Reduce(stoch_size_l_.GetData(), stoch_size_l.GetData(),
            stoch_size_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (myid == 0)
    {
        // Get averages
        std::transform(stoch_size_l.GetData(), stoch_size_l.GetData()+stoch_size_l.Size(),
                stoch_size_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }
    OutputStochInfo(stoch_size_l, stoch_size_g_);
}

void ReduceAndOutputBothInfo(const mfem::Vector& ndofs_l_,
                         const mfem::Vector& ndofs_g_,
                         const mfem::Vector& stoch_size_l_,
                         const mfem::Vector& stoch_size_g_,
                         const mfem::Vector& nnz_,
                         const mfem::Vector& iters_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = iters_.Size();
    Vector ndofs_l(nLevels);
    Vector stoch_size_l(nLevels);
    MPI_Reduce(stoch_size_l_.GetData(), stoch_size_l.GetData(),
        stoch_size_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(ndofs_l_.GetData(), ndofs_l.GetData(),
        ndofs_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (myid == 0)
    {
        // Get averages
        std::transform(stoch_size_l.GetData(), stoch_size_l.GetData()+stoch_size_l.Size(),
                stoch_size_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
        std::transform(ndofs_l.GetData(), ndofs_l.GetData()+ndofs_l.Size(),
                ndofs_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }

    OutputBothInfo(ndofs_l, ndofs_g_, stoch_size_l, stoch_size_g_, nnz_, iters_);
}

void ReduceAndOutputBothInfo(const mfem::Vector& ndofs_l_,
                         const mfem::Vector& ndofs_g_,
                         const mfem::Vector& stoch_size_l_,
                         const mfem::Vector& stoch_size_g_,
                         const mfem::Vector& nnz_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = ndofs_l_.Size();
    Vector ndofs_l(nLevels);
    Vector stoch_size_l(nLevels);
    MPI_Reduce(stoch_size_l_.GetData(), stoch_size_l.GetData(),
        stoch_size_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(ndofs_l_.GetData(), ndofs_l.GetData(),
        ndofs_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (myid == 0)
    {
        // Get averages
        std::transform(stoch_size_l.GetData(), stoch_size_l.GetData()+stoch_size_l.Size(),
                stoch_size_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
        std::transform(ndofs_l.GetData(), ndofs_l.GetData()+ndofs_l.Size(),
                ndofs_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }

    OutputBothInfo(ndofs_l, ndofs_g_, stoch_size_l, stoch_size_g_, nnz_);
}

void ReduceAndOutputBothInfo(const mfem::Vector& ndofs_l_,
                         const mfem::Vector& ndofs_g_,
                         const mfem::Vector& stoch_size_l_,
                         const mfem::Vector& stoch_size_g_)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid, num_procs;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    int nLevels = ndofs_l_.Size();
    Vector ndofs_l(nLevels);
    Vector stoch_size_l(nLevels);
    MPI_Reduce(stoch_size_l_.GetData(), stoch_size_l.GetData(),
        stoch_size_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(ndofs_l_.GetData(), ndofs_l.GetData(),
        ndofs_l.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (myid == 0)
    {
        // Get averages
        std::transform(stoch_size_l.GetData(), stoch_size_l.GetData()+stoch_size_l.Size(),
                stoch_size_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
        std::transform(ndofs_l.GetData(), ndofs_l.GetData()+ndofs_l.Size(),
                ndofs_l.GetData(), std::bind2nd(std::divides<double>(), static_cast<double>(num_procs)));
    }

    OutputBothInfo(ndofs_l, ndofs_g_, stoch_size_l, stoch_size_g_);
}

void SaveFieldGLVis_H1(
        ParMesh * mesh,
        const Vector & coeff,
        const std::string prefix) 
{
    int myid;
    MPI_Comm_rank(mesh->GetComm(), &myid);

    std::ostringstream fid_name;
    fid_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;

    GridFunction x;
    
    FiniteElementCollection * fecH1 = new L2_FECollection(0, mesh->Dimension());
    FiniteElementSpace * fespaceH1 = new FiniteElementSpace(mesh, fecH1);
    auto feCoeff = make_unique<Vector>(coeff.GetData(), coeff.Size() );
    x.MakeRef(fespaceH1, *feCoeff, 0);
    x.MakeOwner(fecH1);

    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);
}

void SaveCoefficientGLVis(
        ParMesh * mesh,
        VectorFunctionCoefficient & coeff,
        const std::string prefix)
{
    int myid;
    MPI_Comm_rank(mesh->GetComm(), &myid);

    FiniteElementCollection * fec = new L2_FECollection(0,mesh->Dimension());
    FiniteElementSpace * fespace = new FiniteElementSpace(mesh, fec);

    GridFunction x(fespace);
    x.MakeOwner(fec);
    x.ProjectCoefficient(coeff); 
    
    std::ostringstream fid_name;
    fid_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;
    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);
}

void SaveCoefficientGLVis_H1(
        ParMesh * mesh,
        VectorFunctionCoefficient & coeff,
        const std::string prefix)
{
    int myid;
    MPI_Comm_rank(mesh->GetComm(), &myid);

    FiniteElementCollection * fec = new H1_FECollection(1,mesh->Dimension());
    FiniteElementSpace * fespace = new FiniteElementSpace(mesh, fec);

    GridFunction x(fespace);
    x.MakeOwner(fec);
    x.ProjectCoefficient(coeff); 
    
    std::ostringstream fid_name;
    fid_name << prefix << "." << std::setfill('0') << std::setw(6) << myid;
    std::ofstream fid(fid_name.str().c_str());
    fid.precision(8);
    x.Save(fid);
}
} /* namespace parelagmc */


