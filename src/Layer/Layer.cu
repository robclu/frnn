/*
 *  Layer implementation file for CUBDLRNN that defines the member 
 *  implementation for the Layer Class 
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

#include "Layer.hpp"

using namespace cubdlrnn;

template<size_t num_inputs_x, size_t num_inputs_h, size_t num_cells, 
	     size_t num_outputs , class Type>
void Layer<num_inputs_x, num_inputs_h, num_cells, num_outputs , Type>::Update( 
		const Type* inputs_x, const Type* inputs_h ) {

	// NOTE : There should be a check here that determined is the GPU or CPU
	//        kernel should be used - which should actually be done when the 
	//        class is created
  	
	// Add all the inputs into a single matrix 
	Type* inputs_d;
	cudaMalloc( (void**)&inputs_d, maxInputs * INPUT_TYPES );
	// Do the memcpy's here
	
	// Create the output array
	Type* updated_results_h[ numCells * INPUT_TYPES ];
	Type* updated_results_d;
    cudaMalloc( (void**)&updated_results_d, numCells * INPUT_TYPES * sizeof( Type ) );

	// Create weight matrix for device
	Type* weights_d;
	cudaMalloc( (void**)&weights_d, numCells * maxInputs * INPUT_TYPES * sizeof( Type ) );
    // Do all the memcpy's	

	// Define the grid size 
	dim3 dimBlock( 1 );
	dim3 dimGrid( maxInputs, numCells, INPUT_TYPES );

	// Invoke the kernel 
	UpdateLayer<<< dimGrid, dimBlock >>>( inputs_d, weights_d, updated_results_d );

	// Get the results back from the GPU
	cudaMemcpy( updated_results_h, updated_results_d, 
			    sizeof( updated_results_h ), cudaMemcpyDeviceToHost );

	// Free memory
	cudaFree( updated_results_d );
}
