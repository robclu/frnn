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

// Change if necessary
#define MAX_BLOCKS			65536
#define THREADS_PER_BLOCK	256
namespace curnn {

/* 
 * ==========================================================================================================
 * Struct		: vectorizeddType	 
 * 
 * Description	: Gets a vectorzed (2) version of dType. For example, if dType is a float this 
 *                will then get float2, similarity for int or double.
 *
 * Params		: dType		: The type of data (float, double etc..)
 *				: N			: The size of the vector (2, 4, 8 ..)
 * ==========================================================================================================
 */
template <typename dType, int N> struct vectorizedType;

// Macro to make instances of each type for all vector sizes
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
