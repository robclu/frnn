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

template <typename	dType, 
		  uint		nodes,
		  uint		inputs,
		  uint		outputs,
		  uint		depth		> 
class layer {

	public:
		curnn::tensor4<dType> dims;
	public:
		explicit layer() :
			dims( nodes, inputs, outputs, depth ) {}

};

}	// Namespace curnn

#endif
