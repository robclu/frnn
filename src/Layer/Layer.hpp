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
	 *              : num_cells		: The number of cells (vertically) in the layer.
	 *              : num_outputs   : The number of outputs from the layer.
	 *              : Precision		: What precision to use (double, float ...)
	 * ============================================================================
	 */
	template<size_t num_inputs_x, size_t num_inputs_h, size_t num_cells, 
	         size_t num_outputs , class Precision>
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
				numCells( num_cells )     , numOutputs( num_outputs ) {}

		    /* 
			 * ====================================================================
			 * Function     : GetOutputs 
			 *
			 * Description  : Gets a constant reference to the Layer outputs 
			 * ====================================================================
			 */
			// NOTE : Maybe inline
			const Precision& GetOutputs() const { return outputs; }

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
			void Update( const Precision* inputsX, const Precision* inputsH ); 

		private:
			/* ------------------------ Size Variables -------------------------- */

			size_t           numInputsX;                        // Number of inputs from data
			size_t           numInputsH;                        // Number of inputs from the hidden layer at the previous time step
			size_t           numCells;                          // Number of cells 
			size_t           numOutputs;                        // Number of outputs 
			
			/* -------------------------- Layer Data ---------------------------- */
			
			// Probably wont need to store the inputs and the hidden 
			Precision        inputsX[ num_inputs_x ];           // Layer data inputs
			Precision        inputsH[ num_inputs_h ];           // Layer hidden inputs
			Precision        outputs[ num_outputs ];            // Layer outputs
			Cell<Precision>  cells[ num_cells ];                // Layer cells

			/* --------------------- Layer Weight matrices ---------------------- */

			// NOTE : The wight matrices for the cells are diagonal, which means
			//        that the only input that matters to the cell at time t, is the
			//        output of the same cell at time t-1.
			Precision		 Wxi[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to cell input gates
			Precision        Whi[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the cell input gates
			Precision        Wci[ num_inputs_h * num_cells ];    // Weight matrix for the prev cell outputs to the cell input gates

			Precision        Wxf[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to the cell forget gates 
            Precision        Whf[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the cell forget gates
			Precision        Whc[ num_inputs_h * num_cells ];    // Weight matrix for the prev cell outputs to the cell forget gates

			Precision        Wxo[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to the cell output gates
			Precision        Who[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the cell output gates
			Precision        Wco[ num_cells * num_cells ];       // Weight matrix for the current cell state gates to the cell output gates

			Precision        Wxc[ num_inputs_x * num_cells ];    // Weight matrix for the data inputs to the current cell stat gates
			Precision        Whc[ num_inputs_h * num_cells ];    // Weight matrix for the prev hidden outputs to the current cell state gates

			Precision        Whh[ num_cells * num_cells ];       // Weight matrix for the connections between the outputs at this time step

			

		private:	
			/* 
			 * ====================================================================
			 * ====================================================================
			 */
	};
}           // End namespace cubdlrnn
#endif      // CUBDLRNN_LAYER_INCLUDE
