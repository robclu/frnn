/*
 *  Header file for fastRNN CPU math kernel functions.
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

#ifndef _FRNN_MATH_KERNELS_CPU_
#define _FRNN_MATH_KERNELS_CPU_

#include <vector>
#include <random>

#include "../frnn/types.h"

/*
 * ==========================================================================================================
 * Function     : rand 
 * 
 * Descrition   : Generates a random number between 2 limits for each element in an array
 * 
 * Inputs       : x         : The array that must be filled with random numbers
 *              : N         : The number of elements in the array that must be filled with a randomm number
 *              : lo        : The lower bound for each random number
 *              : hi        : The upper bound for each random number 
 *              
 * Oututs       : An array of random number on the range lo - hi
 * 
 * Params       : dType     : The type of data of the output element
 * ==========================================================================================================
 */
template <typename dType>
void randCpu( dType* x, size_t N, dType lo, dType hi ) {
    std::random_device                  rd;                             // Random device
    std::mt19937                        gen( rd() );                    // Generator
    std::uniform_real_distribution<>    dist( lo, hi );
    
    for ( size_t i = 0; i < N; i++ ) {
        x[ i ] = static_cast<dType>( dist( gen ) );
    }
}

/*
 * ==========================================================================================================
 * Function     : xmyCpu
 * 
 * Description  : Performs X minus Y for two vectors X and Y, on the CPU
 * 
 * Inputs       : x         : The first input vector
 *              : y         : The second input vector
 *              
 * Outputs      : result    : The resultant vector from X - Y
 * 
 * Params       : dType     : The type of data in the vectors
 * ==========================================================================================================
 */
template <typename dType>
void xmyCpu( std::vector<dType>& x, std::vector<dType>& y, std::vector<dType>& result ) {
    typedef typename frnn::VectorizedInstructionsCpu<dType>    vect_ins;
    typedef typename frnn::VectorizedTypeCpu<dType>::vect_type vect4;
    
    // Get size and pointers 
    const size_t N     = x.size();
    size_t step        = vect_ins::typeSize();
    dType* x_p         = N > 0 ? &x[ 0 ] : NULL;
    dType* y_p         = N > 0 ? &y[ 0 ] : NULL;
    dType* result_p    = &result[ 0 ];
    
    vect4 x4; vect4 y4; vect4 result4;
    
    // For every 4 elements
    for ( size_t i = 0; i < N; i += step ) {
        x4      = vect_ins::mm_load_u( x_p );             // Load to vectorized version
        y4      = vect_ins::mm_load_u( y_p );             // Load to vectorized version
        result4 = vect_ins::mm_sub_p( x4, y4 );           // Subtract
        vect_ins::mm_store_p( result_p, result4 );        // Move result to output
        
        // Increment
        x_p += step; y_p += step; result_p += step;
    }  
    for ( size_t i = 0; i < N % step; i ++ ) {
        result[ N - 1 - i ] = x[ N - 1 - i] - y[ N - i - i ];
    }
}

#endif
