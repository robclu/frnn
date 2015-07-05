/*
 *  Header file for cuRNN types.
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
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _CURNN_TYPES_
#define _CURNN_TYPES_

#include <cuda.h>
#include <type_traits>

#include <emmintrin.h>          // SSE vectorized types

// Change if necessary
#define MAX_BLOCKS			65536
#define THREADS_PER_BLOCK	256

namespace curnn {

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
template <typename dType, int N> struct vectorizedType;

// Macro to make instances of each type for all vector sizes for CUDA vectorized types
#define CURNN_VECTORIZED_INSTANCE( dType )											\
	template <> struct vectorizedType<dType, 1> { typedef dType ## 1  vectType; };	\
	template <> struct vectorizedType<dType, 2> { typedef dType ## 2 vectType; };	\
	template <> struct vectorizedType<dType, 4> { typedef dType ## 4 vectType; };	\

CURNN_VECTORIZED_INSTANCE( double )
CURNN_VECTORIZED_INSTANCE( float  )
CURNN_VECTORIZED_INSTANCE( int    )
CURNN_VECTORIZED_INSTANCE( uint   )
CURNN_VECTORIZED_INSTANCE( char	  )

#undef CURNN_VECTORIZED_INSTANCE

/*
 * ==========================================================================================================
 * Struct       : vectorizedTypeCpu
 * 
 * Description  : Gets a vectorized (v4) version of a type for sse instructions.
 * 
 * Params       : dType     : The type of data (float, double, int )
 * ==========================================================================================================
 */
template <typename dType> struct vectorizedTypeCpu;

template <> struct vectorizedTypeCpu<int>    { typedef __m128i vectType; };
template <> struct vectorizedTypeCpu<char>   { typedef __m128i vectType; };
template <> struct vectorizedTypeCpu<float>  { typedef __m128  vectType; };
template <> struct vectorizedTypeCpu<double> { typedef __m128d vectType; };
template <> struct vectorizedTypeCpu<float*>  { typedef __m128*  vectType; };

// For vectorized instructions
template <typename dType> struct vectInstructions;

template <> struct vectInstructions<float> {
    size_t sz() { return 4; }
    
    typedef __m128 (*load)( const float* );
    static constexpr load mm_load_u = &_mm_loadu_ps;
    
    typedef __m128 (*sub)( __m128, __m128 );
    static constexpr sub mm_sub_p = &_mm_sub_ps;
    
    typedef void (*store)( float*, __m128 );
    static constexpr store mm_store_p = &_mm_store_ps;
};

template <> struct vectInstructions<double> {
    size_t sz() { return 2; }
    
    typedef __m128d (*load)( const double* );
    static constexpr load mm_load_u = &_mm_loadu_pd;
    
    typedef __m128d (*sub)( __m128d, __m128d );
    static constexpr sub mm_sub_p = &_mm_sub_pd;
    
    typedef void (*store)( double*, __m128d );
    static constexpr store mm_store_p = &_mm_store_pd;
};
/*
 * ==========================================================================================================
 * Enum			: curnnError
 *
 * Description	: Enumerator for possible erorrs in curnn.
 * ==========================================================================================================
 */
enum curnnError {
	CURNN_ALLOC_ERROR		= 1,
	CURNN_COPY_ERROR		= 2,
	CURNN_DIMENSION_ERROR	= 3
	};
}

#endif
