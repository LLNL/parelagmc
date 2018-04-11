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
 
#ifndef PAR_MORTAR_ASSEMBLER_HPP_
#define PAR_MORTAR_ASSEMBLER_HPP_ 

#include <memory>
#include <vector>

#include <mfem.hpp>

#include "MortarIntegrator.hpp"

namespace parelagmc 
{
class ParMortarAssembler 
{
public:

    ParMortarAssembler(
        const MPI_Comm comm,
        const std::shared_ptr<mfem::ParFiniteElementSpace> &master, 
        const std::shared_ptr<mfem::ParFiniteElementSpace> &slave);
    
    /// \brief assembles the coupling matrix B. B : master -> slave 
    /// If u is a coefficient associated with master and v with slave
    /// Then v = M^(-1) * B * u; where M is the mass matrix in slave.
    /// Works with L2_FECollection, H1_FECollection and DG_FECollection 
    /// (experimental with RT_FECollection and ND_FECollection). 
    /// \param B the assembled coupling operator. B can be passed uninitialized.
    /// \return true if there was an intersection and the operator has been 
    /// assembled. False otherwise.
    bool Assemble(std::shared_ptr<mfem::HypreParMatrix> &B);

    /// \brief if the transfer is to be performed multiple times use Assemble instead
    bool Transfer(mfem::ParGridFunction &src_fun, mfem::ParGridFunction &dest_fun, bool is_vector_fe = false);

    inline void AddMortarIntegrator(const std::shared_ptr<MortarIntegrator> &integrator)
    {
        integrators_.push_back(integrator);
    }

private:
    MPI_Comm comm_;
    std::shared_ptr<mfem::ParFiniteElementSpace> master_;
    std::shared_ptr<mfem::ParFiniteElementSpace> slave_;	
    std::vector< std::shared_ptr<MortarIntegrator> > integrators_;
};

} /* namespace parelagmc */

#endif /* PAR_MORTAR_ASSEMBLER_HPP_ */
