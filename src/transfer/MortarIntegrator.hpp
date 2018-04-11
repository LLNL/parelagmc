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
 
#ifndef MORTAR_INTEGRATOR_HPP_
#define MORTAR_INTEGRATOR_HPP_ 

#include <mfem.hpp>

namespace parelagmc 
{
/// \class MortarIntegrator
/// \brief Interface for mortar element assembly

class MortarIntegrator 
{
public:
    /// \brief Implements the assembly routine
    /// \param trial is the master/source element 
    /// \param trial_ir is the quadrature formula for evaluating quantities within the trial element
    /// \param trial_Trans the geometric transformation of the trial element
    /// \param test  is the slave/destination element
    /// \param test_ir is the quadrature formula for evaluating quantities within the test element
    /// \param test_Trans the geometric transformation of the test element
    /// \param elemmat the result of the assembly
    virtual void AssembleElementMatrix(const mfem::FiniteElement   &trial,
        const mfem::IntegrationRule &trial_ir,
        mfem::ElementTransformation &trial_Trans,
        const mfem::FiniteElement   &test,
        const mfem::IntegrationRule &test_ir,
        mfem::ElementTransformation &test_Trans,
        mfem::DenseMatrix             &elemmat
        ) = 0;


    /// \return the additional orders of quadarture required by the integrator.
    /// It is 0 by default, override method to change that.
    virtual int GetQuadratureOrder() const
    {
        return 0;
    }

    virtual ~MortarIntegrator() {}
}; /* class MortarIntegrator */

/// \class L2MortarIntegrator
/// \brief Integrator for scalar finite elements 
/// \f$ (u, v)_{L^2(\mathcal{T}_m \cap \mathcal{T}_s)}, u \in U(\mathcal{T}_m ) and v \in V(\mathcal{T}_s ) \f$
class L2MortarIntegrator : public MortarIntegrator 
{
public:
    void AssembleElementMatrix(
        const mfem::FiniteElement     &trial,
        const mfem::IntegrationRule   &trial_ir,
        mfem::ElementTransformation   &trial_Trans,
        const mfem::FiniteElement     &test,
        const mfem::IntegrationRule   &test_ir,
        mfem::ElementTransformation   &test_Trans,
        mfem::DenseMatrix             &elemmat
        ) override;
};

/// \class VectorL2MortarIntegrator
/// \brief Integrator for vector finite elements 
/// \f$ (u, v)_{L^2(\mathcal{T}_m \cap \mathcal{T}_s)}, u \in U(\mathcal{T}_m ) and v \in V(\mathcal{T}_s ) \f$
class VectorL2MortarIntegrator : public MortarIntegrator 
{
public:

#ifndef MFEM_THREAD_SAFE
    mfem::Vector shape;
    mfem::Vector D;
    mfem::DenseMatrix K;
    mfem::DenseMatrix test_vshape;
    mfem::DenseMatrix trial_vshape;
#endif

public:
    VectorL2MortarIntegrator() { Init(NULL, NULL, NULL); }
    VectorL2MortarIntegrator(mfem::Coefficient *_q) { Init(_q, NULL, NULL); }
    VectorL2MortarIntegrator(mfem::Coefficient &q) { Init(&q, NULL, NULL); }
    VectorL2MortarIntegrator(mfem::VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    VectorL2MortarIntegrator(mfem::VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    VectorL2MortarIntegrator(mfem::MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    VectorL2MortarIntegrator(mfem::MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    void AssembleElementMatrix(
        const mfem::FiniteElement     &trial,
        const mfem::IntegrationRule   &trial_ir,
        mfem::ElementTransformation   &trial_Trans,
        const mfem::FiniteElement     &test,
        const mfem::IntegrationRule   &test_ir,
        mfem::ElementTransformation   &test_Trans,
        mfem::DenseMatrix             &elemmat
        ) override;

private:
    mfem::Coefficient *Q;
    mfem::VectorCoefficient *VQ;
    mfem::MatrixCoefficient *MQ;

    void Init(mfem::Coefficient *q, mfem::VectorCoefficient *vq, mfem::MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }
};

} /* namespace parelagmc */

#endif /* MORTAR_INTEGRATOR_HPP_ */
