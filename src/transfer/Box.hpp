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
 
#ifndef BOX_HPP_
#define BOX_HPP_ 

#include <mfem.hpp>

namespace parelagmc 
{
void MaxCol(const mfem::DenseMatrix &mat, mfem::Vector &vec, bool include_vec_elements);
void MinCol(const mfem::DenseMatrix &mat, mfem::Vector &vec, bool include_vec_elements);

class Box 
{
public:

    Box(const int n);
    Box() = default;
    
    virtual ~Box() = default;

    void Reset(const int n);
    void Reset();

    Box(const mfem::DenseMatrix &points);
    Box & operator += (const mfem::DenseMatrix &points);
    Box & operator += (const Box &box);
    
    bool Intersects(const Box &other) const;
    bool Intersects(const Box &other, const double tol) const;
    
    void Enlarge(const double value);

    void Print(std::ostream &os = std::cout) const;

    inline double GetMin(const int index) const
    {
        return min_(index);
    }

    inline double GetMax(const int index) const
    {
        return max_(index);
    }

    inline const mfem::Vector &GetMin() const
    {
        return min_;
    }

    inline const mfem::Vector &GetMax() const
    {
        return max_;
    }
    
    inline mfem::Vector &GetMin() 
    {
        return min_;
    }

    inline mfem::Vector &GetMax() 
    {
        return max_;
    }
    
    inline int GetDims() const
    {
        return min_.Size();
    }

    inline bool Empty() const
    {
        if(min_.Size() == 0) return true;
        return GetMin(0) > GetMax(0);
    }

private:
    mfem::Vector min_;
    mfem::Vector max_;
};

}

#endif /* BOX_HPP_ */
