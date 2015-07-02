/*
 *  Header file for cuRNN layer class.
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

#ifndef _CURNN_LAYER_
#define _CURNN_LAYER_

#include "../tensor/tensor.cuh"

namespace curnn {

/*
 * ==========================================================================================================
 * Class		: layer 
 *
 * Description	: Layer class for the cuRNN that defines a generic class for a layer
 *
 * Params		: dType		: The type of data for the layer
 *				: _nodes	: The number of nodes in the layer
 *				: _depth	: The number of timesteps back or forward that have inputs 
 * ==========================================================================================================
 */
template <typename	dType	, 
		  uint		_nodes	,
		  uint		_inputs	,
		  uint		_depth	> 
class layer {
	public:
		uint					numNodes;
		uint					numInputs;
		uint					depth;
		curnn::tensor4<dType>	weights;	// 1D,2D -> Inputs - Node weights
											// 3D	 -> Biases
											// 4D    -> Activation results (outputs)
	public:
		/*
		 * ==================================================================================================
		 * Function		: layer 
		 *
		 * Description	: Defines the size of the layer parameters
		 * ==================================================================================================
		 */
		explicit layer() :
			numNodes( _nodes ), numInputs( _inputs ), depth( _depth ),
            weights( numInputs, numNodes, numNodes, numNodes ) {}

		/*
		 * ==================================================================================================
		 * Function		: outputs 
		 *
		 * Description	: Retuns a pointer to the outputs of the layer
		 * ==================================================================================================
		 */
		template <typename dType>
		inline const dType* outputs() {
		}

		/*
		 * ==================================================================================================
		 * Function		: forward
		 *
		 * Description	: Compute the forward pass on the layer
		 *===================================================================================================
		 /





};

}	// Namespace curnn

#endif
