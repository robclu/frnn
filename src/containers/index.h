// ==========================================================================================================
//! @file   Header file for fastRNN Index class to access indices of containers.
// ==========================================================================================================

/*
 * ==========================================================================================================
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 * ==========================================================================================================
 */ 

#ifndef _FRNN_TENSOR_INDICES_
#define _FRNN_TENSOR_INDICES_

#include <cstdint>
#include <typeinfo>
#include <iostream>

namespace frnn      {

// ==========================================================================================================
//! @class      Index 
//! @brief      Used to represent an index of a container
//!       
//!             By using the namespace: frnn::index, an index can then be defined for a container to make    \n
//!             accessing elements look nice, for example to access elements of a Tensor.                    \n
//!                                                                                                          \n
//!             For example:                                                                                 \n
//!                                                                                                          \n
//!             frnn::Index i(0)    // Define an Index for the 0th dimension of a container                  \n
//!             frnn::Index j(1)    // Define an Index for the 1st dimension of a container                  \n
//!                                                                                                          \n
//!             Tensor = Tensor(j, i)   // Swap i and j dimensions of a Tensor                               \n
//!             Tensor = Tensor(j, i) * Tensor(i, k)    // Multiply Tensors by dimension                    
// ==========================================================================================================
class Index {
private:   
    size_t _idx;                                                                //!< Value of the dimension
public:
     // =====================================================================================================
     //! @brief         Sets the value of the Index.
     //! @param[in] i   The value to set the Index to.
     // =====================================================================================================
    constexpr Index(size_t i) : _idx(i) {}
        
     // =====================================================================================================
     //! @brief         Returns the value of the Index (which dimension the Index instance represents).
     //! @return        The dimension which the Index instance is representing.
     // =====================================================================================================
    constexpr size_t operator()() const { return _idx; }

    // ======================================================================================================
    //! @brief      Comapares if another Index is equal to this one.
    //! @param[in]  other   The other Index to check for equality with.
    // ======================================================================================================
    bool operator==(const Index& other) const {return _idx == other(); }
};

// ==========================================================================================================
//! @class      IndexHasher
//! @brief      Computes the hash of an Index so that Indexes can be used in uordered_maps.
// ==========================================================================================================
class IndexHasher {
public:
    // ======================================================================================================
    //! @brief      Computes the hash of an Index.
    //! @param[in]  idx     The Index element to determine the hash of.
    //! @return     The hash value of the Index.
    // ======================================================================================================
    size_t operator()(const Index& idx) const 
    {
        return std::hash<size_t>()(idx());
    }
};

/* ============================================== Typedefs ================================================ */

namespace index {
    
Index i(0);                                                 //!< Represents the first dimension of an Index
Index j(1);                                                 //!< Represents the second dimension of an Index
Index k(2);                                                 //!< Represents the third dimension of an Index 
Index l(3);                                                 //!< Represents the fourth dimension of an Index 
Index m(4);                                                 //!< Represents the fifth dimension of an Index 
Index n(5);                                                 //!< Represents the sixth dimension of an Index
Index o(6);                                                 //!< Represents the seventh dimension of an Index 
Index p(7);                                                 //!< Represents the eighth dimension of an Index
Index q(8);                                                 //!< Represents the ninth dimension of an Index 
Index r(9);                                                 //!< Represents the tenth dimension of an Index 
Index s(10);                                                //!< Represents the eleventh dimension of an Index 
Index t(11);                                                //!< Represents the twelvth dimension of an Index 
Index u(12);                                                //!< Represents the thirteenth dimension of an Index 

}       // End namespace dim

}       // End namespace frnn

#endif
