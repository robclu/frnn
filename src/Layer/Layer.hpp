/*
 *  Layer header file for CUBLRNN that defines a layer of the 
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
#define CUBLRNN_LAYER_INCLUDED

#include "../Cell.h"

namespace cublrnn {
	/*
	 * ============================================================================
	 * Class        : Layer
	 *
	 * Description  : Layer class that defines a layer of the network using
	 *                template parameters for the number of inputs, outputs and 
	 *                number of cells
	 * ============================================================================
	 */
	tenplate<size_t num_inputs, size_t num_cells, size_t num_outputs, class Type>
	class Layer {

		private:
			Type        inputs[ num_inputs ];       // Layer inputs
			Type        outputs[ num_outputs ];     // Layer outputs
			Cell<Type>  cells[ num_cells ];         // Layer cells 
	};
}           // End namespace cublrnn
#endif      // CUBLRNN_LAYER_INCLUDED
