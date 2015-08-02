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
 * Function         : Index 
 * 
 * Description      : Struct representing an index (dimension) of a tensor, for example 1 would be an index
 *                    for the second dimension (0 indexing)
 * 
 * Params           : idx       : The value of the index of a tensor the struct represents
 * ==========================================================================================================
 */
struct Index
{
    public:   
        size_t idx;
        Index(size_t i) : idx(i) {}
        
        /*
         * ==================================================================================================
         * Function     : operator()
         * 
         * Description  : Returns the value of the dimension (which dimension the struct represents)
         * 
         * Outputs      : The index of the dimension 
         * ==================================================================================================
         */
        size_t operator()() const { return idx; }
};

/* ================================================ Typedefs ============================================== */

namespace dim {
    
    Index i(0);
    Index j(1);
    Index k(2);

}       // End namespace dim

}       // End namespace tensor 
}       // End namespace frnn

#endif
