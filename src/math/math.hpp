/*    
 *  cuRNN API general math functionality
    Copyright (C) 2015 Rob Clucas robclu1818@gmail.com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#ifndef CURRN_MATH_INCLUDED
#define CURNN_MATH_INCLUDED

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../util/error.hpp"

// To convert variable names to strings
#define stringify( name ) varname( #name )
char* varname( char* name ) { return name; }

namespace currn {
	namespace math {
		/*
		 * ==================================================================================================     
		 * Function		: axpy
		 *
		 * Description	: performs a*X + Y
		 *
		 * Inputs		: a		: Constant for multiplication 
		 *              : x     : Vector to multiply with a
		 *              : y     : Vector to add to a*x
		 * 
		 * Outputs		:
		 * ==================================================================================================
		 */
		template<class dType>
		void saxpy( const dType a, vector<dType> x, vector<dType> y ) {
			dType* da, *db, *dc;			// Device variables

			// Allocate memory for device variables			
			if ( cudaMalloc( (void**)&da, x.size() * sizeof( dType ) ) != cudaSuccess ) {
					curnn::util::err::allocError( stringify( da ) );
			}
				

		}
	}
}
#endif

