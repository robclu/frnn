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
void xmyCpu( std::vector<dType>& v1, std::vector<dType>& v2, std::vector<dType>& res ) {
    typedef typename curnn::vectInstructions<dType>            vect_ins;
    typedef typename curnn::vectorizedTypeCpu<dType>::vectType vect4;
    
    // Get size and pointers 
    const size_t N    = v1.size();
    size_t step       = vect_ins::typeSize();
    dType* v1p        = N > 0 ? &v1[ 0 ] : NULL;
    dType* v2p        = N > 0 ? &v2[ 0 ] : NULL;
    dType* resp       = &res[ 0 ];
    
    vect4 v14; vect4 v24; vect4 res4;
    
    // For every 4 elements
    for ( size_t i = 0; i < N; i += step ) {
        v14  = vect_ins::mm_load_u( v1p );          // Load to vectorized version
        v24  = vect_ins::mm_load_u( v2p );          // Load to vectorized version
        res4 = vect_ins::mm_sub_p( v14, v24 );      // Subtract
        vect_ins::mm_store_p( resp, res4 );          // Move result to output
        
        // Increment
        v1p += step; v2p += step; resp += step;
    }  
    for ( size_t i = 0; i < N % step; i ++ ) {
        res[ N - 1 - i ] = v1[ N - 1 - i] - v2[ N - i - i ];
    }
}

#endif
