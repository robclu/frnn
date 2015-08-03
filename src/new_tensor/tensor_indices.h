/*
 *  Header file for fastRNN tensor dimension index variables.
 *
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
 */ 

#ifndef _FRNN_TENSOR_INDICES_
#define _FRNN_TENSOR_INDICES_

#include <cstdint>
#include <typeinfo>
#include <iostream>

namespace frnn      {
namespace tensor    {

/*
 * ==========================================================================================================
 * Class        : Index 
 * Description  : Struct representing an index (dimension) of a tensor, for example 1 would be an index for 
 *                the second dimension (since the tensors use 0 indexing)
 * ==========================================================================================================
 */
class Index {
private:   
    size_t _idx;         // Value of the dimension
public:
    
    /*
     * ======================================================================================================
     * Function         : Index
     * Description      : Constructor for the index class, which sets the value of the index
     * Params           : i     : The value of the index
     * ======================================================================================================
     */
    constexpr Index(size_t i) : _idx(i) {}
        
    /*
     * ======================================================================================================
     * Function         : operator() 
     * Description      : Returns the value of the Index (which dimension the Index represents)
     * Outputs          : The index of the dimension 
     * ======================================================================================================
     */
    constexpr size_t operator()() const { return _idx; }
};

/* ================================================ Typedefs ============================================== */

namespace dim {
    
Index i(0);
Index j(1);
Index k(2);
Index l(3);
Index m(4);
Index n(5);
Index o(6);
Index p(7);
Index q(8);
Index r(9);
Index s(10);
Index t(11);
Index u(12);
Index v(13);
Index w(14);
Index x(15);
Index y(16);
Index z(17);

}       // End namespace dim

}       // End namespace tensor 
}       // End namespace frnn

#endif
