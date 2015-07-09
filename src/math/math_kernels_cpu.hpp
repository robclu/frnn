/*
 *  Header file for cuRNN CPU math kernel functions.
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

#ifndef _CURNN_MATH_KERNELS_CPU_
#define _CURNN_MATH_KERNELS_CPU_

#include <vector>

#include "../curnn/types.h"

template <typename dType>
void xmyCpu( std::vector<dType>& v_1, std::vector<dType>& v_2, std::vector<dType>& result ) {
    typedef typename curnn::VectorizedInstructionsCpu<dType>    vect_ins;
    typedef typename curnn::VectorizedTypeCpu<dType>::vect_type vect4;
    
    // Get size and pointers 
    const size_t N     = v_1.size();
    size_t step        = vect_ins::typeSize();
    dType* v_1p        = N > 0 ? &v_1[ 0 ] : NULL;
    dType* v_2p        = N > 0 ? &v_2[ 0 ] : NULL;
    dType* result_p    = &result[ 0 ];
    
    vect4 v_14; vect4 v_24; vect4 result_4;
    
    // For every 4 elements
    for ( size_t i = 0; i < N; i += step ) {
        v_14     = vect_ins::mm_load_u( v_1p );           // Load to vectorized version
        v_24     = vect_ins::mm_load_u( v_2p );           // Load to vectorized version
        result_4 = vect_ins::mm_sub_p( v_14, v_24 );      // Subtract
        vect_ins::mm_store_p( result_p, result_4 );       // Move result to output
        
        // Increment
        v_1p += step; v_2p += step; result_p += step;
    }  
    for ( size_t i = 0; i < N % step; i ++ ) {
        result[ N - 1 - i ] = v_1[ N - 1 - i] - v_2[ N - i - i ];
    }
}

#endif
