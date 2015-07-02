/*
 *  Header file for cuRNN tensor class.
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

#ifndef _CURNN_TENSOR_
#define _CURNN_TENSOR_

#include <vector>

namespace curnn  {

/*
 * ==========================================================================================================
 * Class		: tensor
 *
 * Description	: Provides a 4D tensor to store 4 dimensionaly data, or to join data to 4 dimensions to that
 *				  less passes need to be made to the GPU
 *
 * Params		: dType		: The data type for the matrix
 *				: _w		: Num elements in 1st dimension
 *				: _x		: Num elements in 2nd dimension
 *				: _y		: Num elements in 3rd dimension
 *				: _z		: Num elements in 4th dimension
 * ==========================================================================================================
 */
template <typename dType, uint _w = 0, uint _x = 0, uint _y = 0, uint _z = 0>
class tensor {
	public:
		uint				w;
		uint				x;
		uint				y; 
		uint				z;
		std::vector<dType>	data;
	public:
		/*
		 * ==================================================================================================
		 * Function		: tensor (constructor)
		 *
		 * Description	 : Sets the number of elements in each dimension of the tensor and allocates and sets
		 *                 the tensor data to be zero
		 * ==================================================================================================
		 */
		explicit tensor() :
			w( _w ), x( _x ), y( _y ), z( _z ), data( _w * _x * y * _z, 0 ) {}

		/*
		 * ==================================================================================================
		 * Function		: size
		 *
		 * Description	: Retuns the size of the tensor (total number of elements)
		 * ==================================================================================================
		 */
		__inline__ __device__ __host__ size_t size() {
			return data.size();
		}

		/*
		 * ==================================================================================================
		 * Function		: reshape 
		 *
		 * Description	: Reshapes the tensor along each dimension, -1 keeps the dimensionality 
		 *
		 * Inputs		: w_new		: New number of elements for 1st dimensino
		 *				: x_new		: New number of elements for 2nd dimension
		 *				: y_new		: New number of elements for 3rd dimension
		 *				: z_new		: New number of elements for 4th dimension
		 * ==================================================================================================
		 */
		__inline__ __device__ __host__ void reshape( int w_new, int x_new, int y_new, int z_new ) {
			w_new != -1 ? w = static_cast<uint>(w_new) : w = w;		
			x_new != -1 ? x = static_cast<uint>(x_new) : x = x;		
			y_new != -1 ? y = static_cast<uint>(y_new) : y = y;		
			z_new != -1 ? z = static_cast<uint>(z_new) : z = z;		
			data.resize( w * x * y * z, 0 );
		}
};

}
#endif
