/*
 *  Header file for cuRNN rand functions, which are simply structs
 *  with function pointers that call curand functions, but can determine
 *  if the float or double version of the curand functions should be called.
 *
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT AN_size.y WARRANTY; without even the implied warranty of
 *  MERCHANTABILIT_size.y or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _CURNN_RAND_
#ifndef _CURNN_RAND_

namespace curnn {
namespace rng {
    
#include <cuda.h> 
#include <curand.h>
    
/*
 * ==========================================================================================================
 * Struct       : generators
 * 
 * Description  : Struct that hold the function pointer to curand random number generators
 * 
 * Params       : dType     : The type of data to use for the number generator
 * ==========================================================================================================
 */
template <typename dType> struct generators;

// Specialization for floats
template <> struct<float> generators {
    
    // Normal distribution
    typedef curandStatus_t (*curandUniform)( curandGenerator_t, float*, size_t );
    static constexpr curandUniform uniform = &curandGenerateUniform;
};

// Specialization for doubles
template <> struct<double> generators {
    
    // Normal distribution
    typedef curandStatus_t (*curandUniform)( curandGenerator_t, double*, size_t );
    static constexpr curandUniform uniform = &curandGenerateUniformDouble;
};

}   // Namespace rng
}   // Namepsace curnn
#endif 
