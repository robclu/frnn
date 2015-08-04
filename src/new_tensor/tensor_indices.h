// ==========================================================================================================
//! @file   Header file for fastRNN tensor Index class and defined variables used to slice tensors by 
//!         dimension.
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
namespace tensor    {

// ==========================================================================================================
//! @class      Index 
//! @brief      Used to represent a dimension of a Tensor. 
//!       
//!             By using the namespace: frnn::tensor, an index can then be defined for a tensor, making the 
//!             code read more like maths.                                                                   \n
//!                                                                                                          \n
//!             For example:                                                                                 \n
//!                                                                                                          \n
//!             frnn:tensor::Index i(0)                 // Define an Index for the 0th dimension of a Tensor \n
//!             frnn:tensor::Index j(1)                 // Define an Index for the 1st dimension of a Tensor \n
//!                                                                                                          \n
//!             tensor.size(i);                         // Get the size of dimension 0 of a Tensor           \n 
//!             Tensor<> slicedTensor = tensor(j, i)    // Transpose a 2D Tensor
// ==========================================================================================================
class Index {
private:   
    size_t _idx;         //!< Value of the dimension
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
};

/* ================================================ Typedefs ============================================== */

namespace dim {
    
Index i(0);         //!< Represents the first dimension of a Tensor
Index j(1);         //!< Represents the second dimension of a Tensor
Index k(2);         //!< Represents the third dimension of a Tensor 
Index l(3);         //!< Represents the fourth dimension of a Tensor 
Index m(4);         //!< Represents the fifth dimension of a Tensor 
Index n(5);         //!< Represents the sixth dimension of a Tensor;
Index o(6);         //!< Represents the seventh dimension of a Tensor 
Index p(7);         //!< Represents the eighth dimension of a Tensor
Index q(8);         //!< Represents the ninth dimension of a Tensor 
Index r(9);         //!< Represents the tenth dimension of a Tensor 
Index s(10);        //!< Represents the eleventh dimension of a Tensor 
Index t(11);        //!< Represents the twelvth dimension of a Tensor 
Index u(12);        //!< Represents the thirteenth dimension of a Tensor 

}       // End namespace dim

}       // End namespace tensor 
}       // End namespace frnn

#endif
