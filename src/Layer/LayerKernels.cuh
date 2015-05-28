/*
 *  Layer header file for CUBDLRNN which defines the Cuda kernels that will 
 *  be used for the Layer class.
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

#ifndef LAYER_CUDA_KERNELS_INCLUDED
#define LAYER_CUDA_KERNELS_INCLUDED 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace cubdlrnn {

	/* 
	 * ============================================================================
	 * Function		: MatrixSum
	 *
	 * Description  : Computes the sum of each row of a matrix in log_2(N) where 
	 *                N = num elements in each row.
	 *
	 * Params       : matrix	: The matrix with the data for the sum
	 *              : N         : The number of elements in the row.
	 * ============================================================================
	 */
	template <typename Precision> 
	__global__ void reductionSum( Precision* a, int N, T* c ) {}

	/*
	 * ============================================================================	 
	 * Function		: UpdateLayer
	 *
	 * Description  : Device kernel that updates the layer by computing all the
	 *                cell gate values, and computing the cell output.
	 *
	 * Params       :
	 *
	 * NOTE         : This function should be called using many compute units
	 *                like would be done for a global kernel. But my GPU doesn't 
	 *                support  dynamic parallelism so this will have to wait.
	 * =============================================================================
	 */
	template<class Precision, size_t max_input_size, size_t num_cells, size_t num_input_types>
	__global__ void UpdateLayer( Precision* inputs, Precision* weights, Precision* cells ) {     

		// Each gate has three wight matrices : 
		//     - For connections to data inputs
		//     - For connections to previous hidden state
		//     - For connections to cell states
		//
		// Each of these matrices are concatenated onto the end of the previous
		// one so that all data can be passed to the GPU in one pass. This can
		// be though of as a 3D matrix where each page of for one of the above
		// mentioned matrices this blockDim.z == 3 always (for now)

		int idx = blockId.x * blockDim.x + threadId.x;        // X index ( input index )
		int idy = blockId.y * blockDim.y + threadId.y;        // Y index ( cell index  )
        int idz = blockId.z * blockDim.z + threadId.z;        // Z index ( input type  )

		// Copy weight data to shared memory
		__shared__ float sharedWeights[ max_inputs_size *     // Size of weights x dimension
			                            num_cells       *     // Size of weights y dimension 
										num_input_types       // Size of weights z dimension
									   ];

		// Copy the weights into shared memory
		sharedWeights[ ( idz * grdDim.x * gridDim.y ) +      // Offset due to pages (z)
			            ( idy * gridDim.x ) +                 // Offset due to y
						( idx )                               // Offset due to x
					   ] 
	     = weights[ ( idz * grdDim.x * gridDim.y ) +      
			         ( idy * gridDim.x ) +          
					 ( idx )                            
				  ];

		// Make sure all threads are done
		__syncthreads();

		// Now do the multiplication

		// When all the threads are done, there is a cube of dimension 
		// num_cells x num_inputs x num_input_types (3 for now) where each
		// element of the cube is the result of the weight input multiplication
		//
		// Since this is matrix multiplication, we need to do the addition along
		// each row of each page in the matrix 
		
	}
}

#endif 
