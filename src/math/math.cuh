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

#include "../curnn/types.h"

/* ============================================= NOTES ======================================================
 *
 * 1.See Justin Luitjens excellence post on using the __shfl functinos at 
 *   http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 *
 * ==========================================================================================================
 */

using namespace curnn;

		/* NOTE : Need to make these funcitons handle any value array and test if it is better to pad with
		 *        zeros or to just add one or two steps */

		/*
		 * ==================================================================================================     
		 * Function		: warpReduce
		 *
		 * Description	: Performs reduction sum (log(N) sum) within a warp (Thanks to Nvidia for making this
		 *                so easy with __shfl_down). Only the first thread gets the result.
		 *
		 * Outputs/(I)	: val		: Value of first element in array, and where the result is stored
		 *
		 * Params		: dType		: The data type (double, float, int)
		 * ==================================================================================================
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
		 * ==================================================================================================     
		 * Function		: warpReduceAll
		 *
		 * Description	: Performs reduction sum (log(N) sum) within a warp (Thanks to Nvidia for making this
		 *                so easy with __shfl_down). All threads in warp get the results
		 *
		 * Outputs/(I)	: val		: Value of first element in array, and where the result is stored
		 *
		 * Params		: dType		: The data type (double, float, int)
		 * ==================================================================================================
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
		 * ==================================================================================================     
		 * Function		: blockReduce
		 *
		 * Description	: Performs reduction sum (log(N) sum) within a block using the above warpReduce
		 *                function
		 *
		 * Outputs/(I)	: val		: Value of first element in array, and where the result is stored
		 *
		 * Params		: dType		: The data type (double, float, int)
		 * ==================================================================================================
		 */	
		template <typename dType>
		__inline__ __device__ dType blockReduce( dType val ) {
			// Allocate shared memory (this should be a parameter of the kernel, not explicitly declared)
			static __shared__ dType shared[ 32 ];
			int lane = threadIdx.x % warpSize;				// Index in warp
			int wid  = threadIdx.x / warpSize;				// Warp index

			val = warpReduce( val );						// Do reduction on warp

			// For first index in each warp, write result to shared memory
			if ( lane == 0 ) shared[ wid ] = val;	
			__syncthreads();								// Make sure all threads are finished

			// Read from shared memory the results of all warps (essentially 
			// moving the results so that the first warp can use them)
			val = ( threadIdx.x < blockDim.x / warpSize ) ? shared[ lane ] : 0;

			// Do the reduction on the first warp
			if ( wid == 0 ) val = warpReduce( val );

			return val;
		}

		/*
		 * ==================================================================================================     
		 * Function		: blockReduceAtomicVectorized
		 *
		 * Description	: Performs reduction sum (log(N) sum) on an entire array using vectorized types (2
		 *                - this will become general, to include 4 and 8, later) and atomic adds across the 
		 *                blocks which is better for floats and doubles.
		 *
		 * Inputs		: in		: A pointer to the data to compute the sum of 
		 *              : N			: The number of elements to sum (in the array)
		 *
		 * Outputs		: out		: A poiinter to where the result will reside
		 *
		 * Params		: dType		: The data type (double, float, int)
		 * ==================================================================================================
		 */	
		template <class dType>
		__global__ void blockReduceAtomicVectorized( dType* in, dType* out, size_t N ) {
			dType sum = dType( 0 );
			int idx = blockIdx.x * blockDim.x + threadIdx.x;

			// Add values to sum
			for ( int i = idx; i <(  N / 2 ); i += blockDim.x * gridDim.x ) {
				// Convert to vectorized type and all to sum 
				typedef typename curnn::vectorizedType<dType, 2>::vectType vect2;
				vect2 val = reinterpret_cast<vect2*>( in )[ i ];
				sum += val.x + val.y;
			}

			// Add 'extra' elements (if more elements than threads)
			int i = idx + ( N / 2 * 2 );
			if ( i < N ) sum += in[ i ]; 

			// Perform reduction on the blocks
			sum = blockReduce( sum );
			
			// Add all results from each block
			if ( threadIdx.x == 0 ) atomicAdd( out, sum );
		}
#endif
