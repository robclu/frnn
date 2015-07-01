/*
 *  Header file for cuRNN math kernel functions.
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

#ifndef _CURNN_MATH_KERNELS_
#define _CURNN_MATH_KERNELS_

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <cmath>

#include "../curnn/types.h"
#include "../functors/functors.cuh"

/* ============================================= NOTES ======================================================
 *
 * 1.See Justin Luitjens excellence post on using the __shfl functinos at 
 *   http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 *
 * ==========================================================================================================
 */
 
using namespace curnn;

/*
 * NOTE : Need to make these funcitons handle any value array and test if it is better to pad with
 *        zeros or to just add one or two steps */

/*
 * ==========================================================================================================
 * Function		: warpReduce
 *
 * Description	: Performs reduction sum (log(N) sum) within a warp (Thanks to Nvidia for making this
 *                so easy with __shfl_down). Only the first thread gets the shared[ 0 ]
 *
 * Outputs/(I)	: val		: Value of first element in array, and where the shared[ 0 ]
 *
 * Params		: dType		: The data type (double, float, int)
 * ==========================================================================================================
 */	
template <typename dType>
__inline__ __device__ dType warpReduce( dType val ) {
	// Add top half elements to bottom half elements
	for ( int offset = ( warpSize / 2); offset > 0; offset /= 2 ) {
		val += __shfl_down( val, offset );	
	}
	return val;
}

/*
 * ==========================================================================================================
 * Function		: warpReduceAll
 *
 * Description	: Performs reduction sum (log(N) sum) within a warp (Thanks to Nvidia for making this
 *                so easy with __shfl_down). All threads in warp get the shared[ 0 ]
 *
 * Outputs/(I)	: val		: Value of first element in array, and where the shared[ 0 ]
 *
 * Params		: dType		: The data type (double, float, int)
 * ==========================================================================================================
 */	
template <typename dType>
__inline__ __device__ dType warpReduceAll( dType val ) {
	// Xor each element (butterfly operation)
	for ( int offset = ( warpSize / 2); offset > 0; offset /= 2 ) {
		val += __shfl_xor( val, offset );	
	}
	return val;
}

/*
 * ==========================================================================================================
 * Function		: blockReduce
 *
 * Description	: Performs reduction sum (log(N) sum) within a block using the above warpReduce
 *                function and the shared[ 0 ]
 *
 * Outputs/(I)	: val		: Value of first element in array, and where the shared[ 0 ]
 *
 * Params		: dType		: The data type (double, float, int)
 * ==========================================================================================================
 */	
template <typename dType>
__inline__ __device__ dType blockReduce( dType val ) {
	// Allocate shared memory
	static __shared__ dType shared[ 32 ];
	int lane = threadIdx.x % warpSize;				// Index in warp
	int wid  = threadIdx.x / warpSize;				// Warp index

	val = warpReduce( val );						// Do reduction on warp

	// For first index in each warp, write shared[ 0 ]
	if ( lane == 0 ) shared[ wid ] = val;	
	__syncthreads();								// Make sure all threads are finished

	// Read from shared memory the shared[ 0 ]
	// moving the shared[ 0 ]
	val = ( threadIdx.x < blockDim.x / warpSize ) ? shared[ lane ] : 0;

	// Do the reduction on the first warp
	if ( wid == 0 ) val = warpReduce( val );

	return val;
}

/*
 * ==========================================================================================================
 * Function		: blockReduceAll
 *
 * Description	: Performs reduction sum (log(N) sum) within a block using the above warpReduceAll
 *                function and the shared[ 0 ]
 *
 * Outputs/(I)	: val		: Value of first element in array, and where the shared[ 0 ]
 *
 * Params		: dType		: The data type (double, float, int)
 * ==========================================================================================================
 */	
template <typename dType>
__inline__ __device__ dType blockReduceAll( dType val ) {
	// Allocate shared memory 
	static __shared__ dType shared[ 32 ];
	int lane = threadIdx.x % warpSize;				// Index in warp
	int wid  = threadIdx.x / warpSize;				// Warp index

	val = warpReduceAll( val );						// Do reduction on warp (all threads have shared[ 0 ]

	// For first index in each warp, write shared[ 0 ]
	if ( lane == 0 ) shared[ wid ] = val;	
	__syncthreads();								// Make sure all threads are finished

	// Give each warp the shared[ 0 ]
	val = ( lane < blockDim.x / warpSize ) ? shared[ lane ] : 0;

	// Have each war perform the butterfly reduction
	// so that all threads in the block have the shared[ 0 ]
	val = warpReduceAll( val );

	return val;
}

/*
 * ==========================================================================================================
 * Function		: blockReduceAtomicVectorized
 *
 * Description	: Performs reduction sum (log(N) sum) on an entire array using vectorized types (2
 *                - this will become general, to include 4 and 8, later) and atomic adds across the 
 *                blocks which is better for floats 
 *
 * Inputs		: in		: A pointer to the data to compute the sum of 
 *              : N			: The number of elements to sum (in the array)
 *
 * Outputs		: out		: A poiinter to where the shared[ 0 ]
 *
 * Params		: dType		: The data type (double, float, int)
 * ==========================================================================================================
 */	
template <typename dType>
__global__ void blockReduceAtomicVectorized( dType* in, dType* out, size_t N ) {
	dType sum = dType( 0 );
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Add values to sum
	for ( int i = idx; i < ( N / 2 ); i += blockDim.x * gridDim.x ) {
		// Convert to vectorized type and all to sum 
		typedef typename curnn::vectorizedType<dType, 2>::vectType vect2;
		vect2 val = reinterpret_cast<vect2*>( in )[ i ];
		sum += val.x + val.y;
	}

	// Add 'extra' elements (if not multiple of 4)
	int i = idx + ( N / 2 * 2 );
	if ( i < N ) sum += in[ i ]; 

	// Perform reduction on the blocks
	sum = blockReduce( sum );
	
	// Add all shared[ 0 ]
	if ( threadIdx.x == 0 ) atomicAdd( out, sum );
}

/*
 * ==========================================================================================================
 * Function		: blockReduceAtomicVectorizedAll
 *
 * Description	: Performs reduction sum (log(N) sum) on an entire array using vectorized types (2
 *                - this will become general, to include 4 and 8, later) and atomic adds across the 
 *                blocks which is better for floats, each thread then has the shared[ 0 ]
 *
 * Inputs		: in		: A pointer to the data to compute the sum of 
 *              : N			: The number of elements to sum (in the array)
 *
 * Outputs		: out		: A poiinter to where the shared[ 0 ]
 *
 * Params		: dType		: The data type (double, float, int)
 * ==========================================================================================================
 */	
template <typename dType, typename F = voidFunctor>
__global__ void blockReduceAtomicVectorizedAll( dType* in, dType* out, size_t N, F f = voidFunctor() ) {

	typedef typename curnn::vectorizedType<dType, 4>::vectType vect4;
	dType  sum  = dType( 0 );

	// Get global index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Add values to sum
	for ( int i = idx; i < ( N / 4 ); i += blockDim.x * gridDim.x ) {
		// Convert to vectorized type and all to sum 
		vect4 val = reinterpret_cast<vect4*>( in )[ i ];
		sum += f( val.x ) + f( val.y ) + f( val.z ) + f( val.w );
	}

	// Determine elements that were not vectorized
	int i = idx + ( N / 4 * 4 );
	if ( i < N ) sum += f( in[ i ] );

	// Perform reduction on the blocks, each thread in the 
	// block will have the sum of that block 
	sum = blockReduceAll( sum );

	// Add the results of all other blocks to the first 
	// element of this block 		
	int out_index = threadIdx.x * blockDim.x;
	atomicAdd( &out[ out_index ], sum );
}

/*
 * ==========================================================================================================
 * Function		: blockScatter
 *
 * Description	: 'Scatters' the first element of a block to all other elements of the block. Block
 *                sizes greated than the number of thrGeads per block get the values to scatter from 
 *                blocks blockDim.x blocks before
 *
 * Inputs		: data		: A pointer to the data where the first element of the block must be
 *                            scattered
 *              : N			: The number of elements in the array
 *
 * Outputs		: data		: The array where each thread in the block has the same value
 *
 * Params		: dType		: The data type (double, float, int)
 * ==========================================================================================================
 */	
template <class dType> 
__global__ void blockScatter( dType* data, size_t N ) {
	static __shared__ dType shared[ 1 ];

	// Get global index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Copy the first element of the block to shared memory
	if ( threadIdx.x == 0  && blockIdx.x < blockDim.x ) {
		shared[ 0 ] = data[ idx ];
	} else if ( threadIdx.x == 0 && blockIdx.x >= blockDim.x ) {
		shared[ 0 ] = data[ idx - blockDim.x * blockIdx.x ];
	}
	__syncthreads();

	// Copy from shared memory to each thread, if in range
	if ( idx < N ) data[ idx ] = shared[ 0 ];
}

/*
 * ==========================================================================================================
 * Function		: softmaxKernel
 *
 * Description	: Computes the softmax function for a vector. It required the above 2 kernels to be
 *                executed first so that each thread has the sum of the vector.
 *
 * Inputs		: in		: The vector to compute the softmax function on
 *              : N			: The number of elements in the vector
 *              : f			: The operation to perform on each element, defult to exponentiation as
 *                            per the softmax function
 *
 * Outputs		: out		: The vector where each element is the result of the softmax
 *
 * Params		: dType		: The data type (double, float, int)
 *				: F			: The functor that defines the operation on the input data
 * ==========================================================================================================
 */
template <typename dType, typename F = expFunctor>
__global__ void softmaxKernel( dType* in, dType* out, size_t N, F f = expFunctor() ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Each output is assumed to already have the sum of the exponent of each element
	// So invert and multiply each element by its exponent
	if ( idx < N ) out[ idx ] = f( in[ idx ] ) / out[ idx ];
}

#endif
