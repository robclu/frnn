/*
 *  Implementation file for cuRNN math functions.
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
 *  _size.you should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "math.hpp"

// Forward declare util function for compiler
namespace curnn {
	namespace util {
		namespace err {
			void allocError( char* );
			void copyError(  char* );
		}
	}
}	

void curnn::math::saxpy( const float a, const std::vector<float>& x, std::vector<float>& y ) {

	cublasStatus_t status;			
	cublasHandle_t handle;
	float* da = 0, *dx = 0, *dy = 0;

	// Initialize handle
	status = cublasCreate( &handle );
	cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_DEVICE );

	// Allocate and fill device vectors with host vector data (checks for errors)
	//curnn::util::mem::allocVector( status, &a, 1, da );	
	//curnn::util::mem::allocVector( status, &x[0], x.size(), dx );
	//curnn::util::mem::allocVector( status, &y[0], y.size(), dy );
	if ( cudaMalloc( (void**)&da, sizeof( float ) ) != cudaSuccess ) {
		curnn::util::err::allocError( stringify( da ) );
	}
	if ( cudaMalloc( (void**)&dx, x.size() * sizeof( float ) ) != cudaSuccess ) {
		curnn::util::err::allocError( stringify( dx ) );	
	}
	if ( cudaMalloc( (void**)&dy, y.size() * sizeof( float ) ) != cudaSuccess ) {
		curnn::util::err::allocError( stringify( dy ) );	
	}

	// Fill device vectors with data
	if ( cudaMemcpy( da, &a, sizeof( float ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( da ) );
	}

	// Perform CUBLAS saxpy
	//status = cublasSaxpy( handle, x.size(), da, dx, 1, dy, 1 );

	// Get the result (checks for errors)
	//curnn::util::mem::getVector( status, dy, y.size(), &y[0] );	

	// Destroy cublas handle
	status = cublasDestroy( handle );

	cudaFree( da );
	cudaFree( dx );
	cudaFree( dy );
}

