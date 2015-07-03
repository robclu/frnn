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
		uint				numNodes;
		uint				numInputs;
		uint				depth;
		tensor4<dType>		weights;	// See constructor comment for what the tensor holds
		std::vector<dType>  outputs;
	public:
		/*
		 * ==================================================================================================
		 * Function		: layer 
		 *
		 * Description	: Defines the size of the layer parameters. The wights are stores as pages where each
		 *                page is the weights between the inputs or a previous iteration of a hidden layer. 
		 *                Each page has the following format :
		 *
		 *                | Woo Wo1 ... WoN | N = nodes
		 *                | W1o W11 ... W1N |
		 *                |  .   .  .    .  |
		 *                |  .   .    .  .  | 
		 *                | WM0 WM1     WMN | M = max( inputs, nodes )
		 *                | boP b1P ... bNP | b = bias, P = page = inputs, hidden_-1, hidden_-2 etc
		 *                | aoP a1P ... aNP | a = activation, from Wx + b from its page
		 *
		 * ==================================================================================================
		 */
		explicit layer() :
			numNodes( _nodes ), numInputs( _inputs ), depth( _depth ),
            weights( std::max( _inputs, _nodes ) + 2		// +2 from biases and activaitons
					, _nodes										
					, _depth								// Number of previous hidden inputs
				    , 0  ),									// Nor using 4th dimension
			outputs( _nodes, 0 ) 
			{}

		/*
		 * ==================================================================================================
		 * Function		: outputs 
		 *
		 * Description	: Retuns a pointer to the outputs of the layer
		 *
		 * Outputs		: A constant pointer to the outputs of the layer
		 * ==================================================================================================
		 */
		inline const dType* getOutputs() const {
			return &outputs[ 0 ]; 
		}

		/*
		 * ==================================================================================================
		 * Function		: forward
		 *
		 * Description	: Compute the forward pass on the layer
		 *===================================================================================================
		 */
		void forward();
};

/* ==================================== Template Implementation =========================================== */

template <typename dType, uint n, uint i, uint d>
void layer<dType,n, i, d>::forward( ) {

}

}	// Namespace curnn

#endif
