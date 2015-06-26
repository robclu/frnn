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
			void allocError( const char* );
			void copyError(  const char* );
		}
	}
}	

void curnn::math::axpy( cublasStatus_t& status     , const float a, 
		                 const std::vector<float>& x, std::vector<float>& y ) {

	cublasHandle_t handle;
	float* da = 0, *dx = 0, *dy = 0;

	// Initialize handle
	status = cublasCreate( &handle );
	cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_DEVICE );

	// Allocate and fill device vectors with host vector data 
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
	if ( cudaMemcpy( dx, &x[0], x.size() * sizeof( float ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( da ) );
	}
	if ( cudaMemcpy( dy, &y[0], y.size() * sizeof( float ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( da ) );
	}

	// Perform CUBLAS saxpy
	status = cublasSaxpy( handle, x.size(), da, dx, 1, dy, 1 );

	// Get the result (checks for errors)
	if ( cudaMemcpy( &y[0], dy, y.size() * sizeof( float ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( y ) );
	}

	// Destroy cublas handle
	status = cublasDestroy( handle );

	// Free device memory
	cudaFree( da );
	cudaFree( dx );
	cudaFree( dy );
}

void curnn::math::axpy( cublasStatus_t& status      , const double a, 
		                 const std::vector<double>& x, std::vector<double>& y ) {

	cublasHandle_t handle;
	double* da = 0, *dx = 0, *dy = 0;

	// Initialize handle
	status = cublasCreate( &handle );
	cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_DEVICE );

	// Allocate and fill device vectors with host vector data 
	if ( cudaMalloc( (void**)&da, sizeof( double ) ) != cudaSuccess ) {
		curnn::util::err::allocError( stringify( da ) );
	}
	if ( cudaMalloc( (void**)&dx, x.size() * sizeof( double ) ) != cudaSuccess ) {
		curnn::util::err::allocError( stringify( dx ) );	
	}
	if ( cudaMalloc( (void**)&dy, y.size() * sizeof( double ) ) != cudaSuccess ) {
		curnn::util::err::allocError( stringify( dy ) );	
	}

	// Fill device vectors with data
	if ( cudaMemcpy( da, &a, sizeof( double ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( da ) );
	}
	if ( cudaMemcpy( dx, &x[0], x.size() * sizeof( double ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( da ) );
	}
	if ( cudaMemcpy( dy, &y[0], y.size() * sizeof( double ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( da ) );
	}

	// Perform CUBLAS saxpy
	status = cublasDaxpy( handle, x.size(), da, dx, 1, dy, 1 );

	// Get the result (checks for errors)
	if ( cudaMemcpy( &y[0], dy, y.size() * sizeof( double ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		curnn::util::err::copyError( stringify( y ) );
	}

	// Destroy cublas handle
	status = cublasDestroy( handle );

	// Free device memory
	cudaFree( da );
	cudaFree( dx );
	cudaFree( dy );
}

void curnn::math::softmax( cublasStatus_t& status, std::vector<float>& x ) {
}

