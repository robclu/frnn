/*
 *  Header file for cuRNN math functions.
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

#ifndef _CURNN_MATH_
#define	_CURNN_MATH_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>
#include <omp.h>

#include "../tensor/tensor.cuh"
#include "../util/errors.h"
#include "../curnn/curnn.h"
#include "math.cuh"

#define THREADS_PER_BLOCK 256

namespace curnn {
namespace math  {
	
/*
 * ==========================================================================================================
 * Function		: axpy
 *
 * Description	: Performs simgle precision a*X + Y
 *
 * Inputs		: error		: cuRNN error type for result of operations
 *				: a			: Constant for multiplication 
 *              : x			: Vector to multiply with a
 * 
 * Outputs/(I)	: y			: Vector used in a*X + Y, and where the result of a*X + Y is stored
 * ==========================================================================================================
 */
void axpy( curnnError& error   , const float a, 
		   const std::vector<float>& x, std::vector<float>& y );	

/*
 * ==========================================================================================================
 * Function		: axpy
 *
 * Description	: Performs double precision a*X + Y
 *
 * Inputs		: error		: cuRNN error type for result of operations
 *				: a			: Constant for multiplication 
 *              : x			: Vector to multiply with a
 * 
 * Outputs/(I)	: y			: Vector used in a*X + Y, and where the result of a*X + Y is stored
 * ==========================================================================================================
 */
void axpy( curnnError& error    , const double a, 
		   const std::vector<double>& x, std::vector<double>& y );	


/*
 * ==========================================================================================================
 * Function		: sum 
 *
 * Description	: Performs the sum of the elements in a vector
 *					
 * Inputs		: error		: cuRNN error type for results of operations
 *				: x			: The vector, araary etc.. (data) to comupte the sum of
 *        
 * Outputs		: val		: The result of the sum of the array 
 *
 * Params		: dType		: The data type of the array elements
 * ==========================================================================================================
 */	 
template <typename dType>
dType sum( curnnError& error, const std::vector<dType>& x );

/*
 * ==========================================================================================================
 * Function		: sumVectorized 
 *
 * Description	: Performs the sum of the elements in a vector and returns a vector of the same
 *                dimension with each element having the result
 *					
 * Inputs		: error		: cuRNN error type for results of operations
 *				: x			: The vector, araary etc.. (data) to comupte the sum of
 *        
 * Outputs		: val		: A vector where each element holds the result of the sum
 *
 * Params		: dType		: The data type of the array elements
 * ==========================================================================================================
 */	 
template <typename dType>
void sumVectorized( curnnError& error, const std::vector<dType>& x, std::vector<dType>& val );

/*
 * ==========================================================================================================
 * Function		: softmax
 *
 * Description	: Performs the softmax function of a vector of data x, which is 
 *					
 *				  softmax( x_i ) = exp( x_i ) / sum[ j=1 to J ]( exp( x_j )
 *
 * Inputs		: status	: Cublas status for determining correct completion of operation
 *        
 * Outputs/(I)  : x			: Vector to compute the softmax of, and to store the result in
 *
 * Params		: dType		: The type of data (float or int) double not supported due to lack of
 *                            support for doubles in some Nvidia kernel functions
 * ==========================================================================================================
 */	
template <typename dType>
void softmax( curnnError& error, const std::vector<dType>& x, std::vector<dType>& val );

template <typename dType>
void softmax( curnnError& error, const std::vector<dType>& x, tensor4<dType>& wb, size_t wStride,
			  std::vector<dType>& outputs );

/* ============================= Implementations for templated functions ================================== */

template <typename dType>
dType sum( curnnError& error, const std::vector<dType>& x ) {

	dType* in = 0, *out = 0, val = 0;

	// Alllocate memory on the device
	if ( cudaMalloc( (void**)&in, x.size() * sizeof( dType ) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( in ) );
	}
	if ( cudaMalloc( (void**)&out, sizeof( dType) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( out ) );
	}

	// Copy data from x to in
	if ( cudaMemcpy( in, &x[0], x.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( in ) );
	}
	// Set out to 0 on the device
	if ( cudaMemsetAsync( out, 0, sizeof( dType) ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( out ) );
	}

	// 256 threads per block is optimal, however, if this isn't enough use more
	int threads;
	x.size() > 256 * MAX_BLOCKS ? threads = 512 : threads = 256;
	int blocks  = std::min( static_cast<int>( ( ( x.size() / 2 ) + threads - 1 ) / threads ), MAX_BLOCKS );

	blockReduceAtomicVectorized<<<blocks, threads>>>( in, out, x.size() );

	if ( cudaMemcpy( &val, out, sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( out ) );
	}

	cudaFree( in ); cudaFree( out );
	return val;
}

template <typename dType>
void sumVectorized( curnnError& error, const std::vector<dType>& x, std::vector<dType>& val ) {

	dType* in = 0, *out = 0;
	
	// Check output vector can hold results
	if ( val.capacity() < x.size() ) val.reserve( x.size() );

	// Alllocate memory on the device
	if ( cudaMalloc( (void**)&in, x.size() * sizeof( dType ) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( in ) );
	}
	if ( cudaMalloc( (void**)&out, x.size() * sizeof( dType) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( out ) );
	}
	
	// Copy data from x to in
	if ( cudaMemcpy( in, &x[0], x.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( in ) );
	}
	if ( cudaMemset( out, 0, x.size() * sizeof( dType ) ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( out ) );
	}
	
	// Determine the size of the grids for the kernel, we need enough blocks
	// to make sure that each element of the output vector gets a result
	int threads;
	x.size() > 256 * MAX_BLOCKS ? threads = 512 : threads = 256;
	int blocks  = std::min( static_cast<int>( x.size() / threads ), MAX_BLOCKS );
	if (  blocks * threads < x.size() ) blocks++;

	blockReduceAtomicVectorizedAll<<<blocks, threads>>>( in, out, x.size() );
	// Copy result from first thread in each block to the others
	blockScatter<<<blocks, threads>>>( out, x.size() );			

	if ( cudaMemcpy( &val[0], out, x.size() * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( val ) );
	}

	cudaFree( in ); cudaFree( out );
}

template <typename dType>
void softmax( curnn::curnnError& error, const std::vector<dType>& x, std::vector<dType>& val ) {

	dType* in = 0, *out = 0;
	expFunctor expOp;			// Define operation on each element to be exponentiation

	// Check output vector can hold all reasults
	if ( val.capacity() < x.size() ) val.reserve( x.size() );
	
	// Alllocate memory on the device
	if ( cudaMalloc( (void**)&in, x.size() * sizeof( dType ) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( in ) );
	}
	if ( cudaMalloc( (void**)&out, x.size() * sizeof( dType) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( out ) );
	}
	
	// Copy data from x to in
	if ( cudaMemcpy( in, &x[0], x.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( in ) );
	}
	if ( cudaMemset( out, 0, x.size() * sizeof( dType ) ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( out ) );
	}
	
	// Determine the size of the grids for the kernel, we need enough blocks
	// to make sure that each element of the output vector gets a result
	int threads;
	x.size() > 256 * MAX_BLOCKS ? threads = 512 : threads = 256;
	int blocks  = std::min( static_cast<int>( x.size() / threads ), MAX_BLOCKS );
	if (  blocks * threads < x.size() ) blocks++;

	// Execute kernel to reduce all blocks, using the exp functor to
	// exponentiate each element before addition
	blockReduceAtomicVectorizedAll<<<blocks, threads>>>( in, out, x.size(), expOp );
	// Copy result from the first thread inea ch block to the others
	blockScatter<<<blocks, threads>>>( out, x.size() );
	// Do normalization to get get softmax
	softmaxKernel<<<blocks, threads>>>( in, out, x.size() );		

	if ( cudaMemcpy( &val[0], out, x.size() * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( val ) );
	}

	cudaFree( in ); cudaFree( out );
}

template <typename dType>
void softmax( curnn::curnnError& error, 
		      const std::vector<dType>& x,
			  curnn::tensor4<dType>& wb  , size_t wStride,
			  std::vector<dType>& y )		{

	// Cublas initialization
	cublasHandle_t handle;
	cublasStatus_t status;
	status = cublasCreate( &handle );
	cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_HOST );

	// Device pointers 
	std::vector<dType*> dPointers( 3 * wb.z, 0 );
	dType* results_h[ wb.z ]; dType** results_d;
	expFunctor expOp;			// Define operation on each element to be exponentiation

	// Check output vector can hold all reasults
	if ( y.capacity() < wb.x ) y.reserve( wb.x );
	// Check dimensions
	if ( x.size() != wStride ) {
		curnn::err::dimError( error, stringify( x ), stringify( wStride ) );
		return;
	}

	// Each page is done by a separate kernel
	#pragma omp parallel num_threads( wb.z )
	{
		int threadId = omp_get_thread_num();	
		// Alllocate memory on the device
		if ( cudaMalloc( (void**)&dPointers[ 3 * threadId ], x.size() * sizeof( dType ) ) != cudaSuccess ) {
			curnn::err::allocError( error, stringify( in ) );
		}
		if ( cudaMalloc( (void**)&dPointers[ 3 * threadId + 1 ], wb.x * wStride * sizeof( dType ) ) != cudaSuccess ) {
			curnn::err::allocError( error, stringify( weights ) );
		}
		if ( cudaMalloc( (void**)&dPointers[ 3 * threadId + 2 ], wb.x * sizeof( dType) ) != cudaSuccess ) {
			curnn::err::allocError( error, stringify( out ) );
		}

		// Copy data from host to device
		if ( cudaMemcpy( dPointers[ 3 * threadId ], &x[0], x.size() * sizeof( dType ), 
					     cudaMemcpyHostToDevice ) != cudaSuccess ) {
			curnn::err::copyError( error, stringify( in ) );
		}
		if ( cudaMemcpy( dPointers[ 3 * threadId + 1 ], &wb( 0, 0, threadId, 0 ), 
					     wb.x * wStride * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
			curnn::err::copyError( error, stringify( weights ) );
		}
		if ( cudaMemcpy( dPointers[ 3 * threadId + 2 ], &wb( 0, wStride, threadId, 0 ), 
					     wb.x * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
			curnn::err::copyError( error, stringify( biases ) );
		}
		// Multiply inputs and weights (column-wise) and add biases
		dType alpha = 1; dType beta = 1;
		status = cublasSgemv( handle, CUBLAS_OP_N, wb.x, wStride, &alpha, dPointers[ 3 * threadId + 1 ], 
							  wb.x, dPointers[ 3 * threadId ], 1, &beta, dPointers[ 3 * threadId + 2 ], 1 );
		
		// Assign results to results pointer array
		results_h[ threadId ] = dPointers[ 3 * threadId + 2 ];
	}

	// Allocate space and copy the pointers to the resuls to host memory
	if ( cudaMalloc( (void**)&results_d, wb.z * sizeof( dType* ) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( results_h ) );
	}
	if ( cudaMemcpy( results_d, results_h, wb.z * sizeof( dType* ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( results_d ) );
	}

	// Determine sizes of blocks and threads for next kernel
	size_t threadsX = wb.x >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : wb.x;
	size_t threadsY = wb.z >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : wb.z;
	size_t blocksX  = wb.x  > THREADS_PER_BLOCK ? wb.x / THREADS_PER_BLOCK + 1 : 1;
	size_t blocksY  = wb.z  > THREADS_PER_BLOCK ? wb.z / THREADS_PER_BLOCK + 1 : 1;
	size_t sharedMemAmount = wb.z * ( wb.x / 2 ) * sizeof( dType );

	dim3 blocks( blocksX, blocksY );
	dim3 threads( threadsX, threadsY );

	// Sum all the results of W*x + b from each page (or layer in network)
	xpny<<<blocks, threads, sharedMemAmount>>>( results_d, wb.x, wb.z );

	// Create pointer to the softmax result
	dType* out;
	if ( cudaMalloc( (void**)&out, wb.x * sizeof( dType ) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( out ) );
	}
	if ( cudaMemset( out, 0, wb.x * sizeof( dType ) ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( out ) );
	}

	// Define grid size for the softmax operations 
	wb.x > THREADS_PER_BLOCK * MAX_BLOCKS	? 
		threadsX = 2 * THREADS_PER_BLOCK	: 
		threadsX = THREADS_PER_BLOCK;
	
	blocksX = std::min( static_cast<int>( wb.x / threadsX ), MAX_BLOCKS );
	if ( blocksX * threadsX < wb.x ) blocksX++;

	// Perform softmax on the resultant vector 
	blockReduceAtomicVectorizedAll<<<blocksX, threadsX>>>( dPointers[ 2 ], out, wb.x, expOp );
	// Copy result from the first thread in each block to the others
	blockScatter<<<blocksX, threadsX>>>( out, wb.x );
	// Do normalization to get get softmax
	softmaxKernel<<<blocksX, threadsX>>>( dPointers[ 2 ], out, wb.x );		

	if ( cudaMemcpy( &y[0], out, wb.x * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
		curnn::err::copyError( error, stringify( y ) );
	}

	cublasDestroy( handle );

	for ( int i = 0; i < dPointers.size(); i++ ) cudaFree( dPointers[i] );
	cudaFree( results_d ); cudaFree( out );
}

}	// Namespace math
}	// Namespace curnn

#endif
