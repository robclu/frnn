/*
 *  Header file for fastRNN GPU vectorized types. Provides wrappers around 
 *  the CUDA vectorized types to allow the correct type to be used in 
 *  templated functions.
 *  
 * Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *    
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT AN_size.y WARRANTY; without even the implied warranty of
 *  MERCHANTABILIT_size.y or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _FRNN_VECTORIZED_TYPES_GPU_
#define _FRNN_VECTORIZED_TYPES_GPU_

#include <cuda.h>

namespace frnn {
    
/* 
 * ==========================================================================================================
 * Struct		: vectorizedType	 
 * 
 * Description	: Gets a vectorzed version of dType. For example, if dType is a float this 
 *                will then get float2, similarity for int or double.
 *
 * Params		: dType		: The type of data (float, double etc..)
 *				: N			: The size of the vector (2, 4, 8 ..)
 * ==========================================================================================================
 */
template <typename dType, int N> struct VectorizedTypeGpu;

// Macro to make instances of each type for all vector sizes for CUDA vectorized types
#define FRNN_VECTORIZED_INSTANCE( dType )										  	    \
	template <> struct VectorizedTypeGpu<dType, 1> { typedef dType ## 1 vect_type; };	\
	template <> struct VectorizedTypeGpu<dType, 2> { typedef dType ## 2 vect_type; };	\
	template <> struct VectorizedTypeGpu<dType, 4> { typedef dType ## 4 vect_type; };	\

// Make partial specifications for different types
FRNN_VECTORIZED_INSTANCE( double )
FRNN_VECTORIZED_INSTANCE( float  )
FRNN_VECTORIZED_INSTANCE( int    )
FRNN_VECTORIZED_INSTANCE( uint   )
FRNN_VECTORIZED_INSTANCE( char	  )

#undef FRNN_VECTORIZED_INSTANCE

}   // Namespace frnn

#endif
