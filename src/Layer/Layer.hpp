/*
 *  Layer header file for CUBDLRNN that defines a layer of the 
 *  neural network.
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

#ifndef CUNLRNN_LAYER_INCLUDED
#define CUBDLRNN_LAYER_INCLUDED

#include "../Cell/Cell.h"
#include "LayerKernels.cu"

using namespace cubdlrnn;
using namespace cubdlrnn::cell;

namespace cubdlrnn {

	/*
	 * ============================================================================
	 * Class        : Layer
	 *
	 * Description  : Layer class that defines a layer of the network using
	 *                template parameters for the number of inputs, outputs and 
	 *                number of cells
	 *
	 * Params (T)	: num_inputs_x	: The number of inputs to the layer from data.
	 *              : num_inputs_h  : The number of inputs from the hidden layer
	 *                                at the previous time step.
	 *              : num_cells     : The number of cells (vertically) in the layer.
	 *              : num_outputs   : The number of outputs from the layer.
	 *              : Type		    : What class type to use (double, float ...)
	 * ============================================================================
	 */
	template<size_t num_inputs_x, size_t num_inputs_h, size_t num_cells, 
	         size_t num_outputs , class Type>
	class Layer {

		public:
			/* 
			 * ====================================================================
			 * Function     : Layer 
			 *
			 * Description  : Constructs the layer class, initializing the size
			 *                parameters.	
			 * ====================================================================
			 */
			explicit Layer() : 
				numInputsX( num_inputs_x ), numInputsH( num_inputs_h ), 
				numCells( num_cells )     , numOutputs( num_outputs ) {
				    num_inputs_x >= num_inputs_h ? maxInputs = num_inputs_x :
			                                       maxInputs = num_inputs_h;	
				}

            /* 
			 * ====================================================================
			 * Function     : GetOutputs 
			 *
			 * Description  : Gets a constant reference to the Layer outputs 
			 * ====================================================================
			 */
			// NOTE : Maybe inline
			const Type& GetOutputs() const { return outputs; }

			/* 
			 * ====================================================================
			 * Function		: Update
			 *
			 * Description	: Updates the layer by forward propagating the
			 *                inputs through the layer.
			 *
			 * Params       : inputsX   : The inputs to the layer from (new)
			 *                            data which is used to update the
			 *                            layer.
			 *              : inputsH   : The inputs from the previous or future
			 *                            hidden layer that are used to update the 
			 *                            layer
			 * ====================================================================
			 */
			void Update( const Type* inputs_x, const Type* inputs_h ); 

		private:
			/* ------------------------ Size Variables -------------------------- */

			size_t      numInputsX;                        // Number of inputs from data
			size_t      numInputsH;                        // Number of inputs from the hidden layer at the previous time step
			size_t      maxInputs;                         // The maximum number of inputs
			size_t      numCells;                          // Number of cells 
			size_t      numOutputs;                        // Number of outputs 
			
			/* -------------------------- Layer Data ---------------------------- */
			
			// Probably wont need to store the inputs and the hidden 
			Type        inputsX[ num_inputs_x ];           // Layer data inputs
			Type        inputsH[ num_inputs_h ];           // Layer hidden inputs
			Type        outputs[ num_outputs ];            // Layer outputs
			Cell<Type>  cells[ num_cells ];                // Layer cells

			/* --------------------- Layer Weight matrices ---------------------- */

			// NOTE : The wight matrices for the cells are diagonal, which means
			//        that the only input that matters to the cell at time t, is the
			//        output of the same cell at time t-1.
	
			// Weight matrix for the inputs (data, prev hidden values, prev cell
			// states) to the cell input gates
		    Type        Wi[ num_inputs_x * num_cells +
				            num_inputs_h * num_cells * 3 ];

			Type        Wxi[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to cell input gates
			Type        Whi[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the cell input gates
			Type        Wci[ num_inputs_h * num_cells ];    // Weight matrix for the prev cell outputs to the cell input gates

			Type        Wxf[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to the cell forget gates 
            Type        Whf[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the cell forget gates
			Type        Wcf[ num_inputs_h * num_cells ];    // Weight matrix for the prev cell outputs to the cell forget gates

			Type        Wxo[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to the cell output gates
			Type        Who[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the cell output gates
			Type        Wco[ num_cells * num_cells ];       // Weight matrix for the current cell state gates to the cell output gates

			Type        Wxc[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to the current cell stat gates
			Type        Whc[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the current cell state gates

			Type        Whh[ num_cells * num_cells ];       // Weight matrix for the connections between the outputs at this time step

		private:	
			/* 
			 * ====================================================================
			 * Function		: GetPreviousCellOutputs
			 *
			 * Description  : Adds the cell outputs at the previous iteration to
			 *                the array of all inputs (x, h and c)
			 *
			 * Params       : input_array	: The current input array.
			 *              : start_index   : The index that the elements must
			 *                                be added from 
			 * ====================================================================
			 */
			void GetPreviousCellOutputs( Type* input_array, size_t start_index ) const;
	};
}           // End namespace cubdlrnn
#endif      // CUBDLRNN_LAYER_INCLUDE
