/*
 *  cuRNN functors. Define general operations which are normally passed 
 *  as template parameters to allow multiple operations for a function/
 *  kernel.
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

#ifndef _CURNN_FUNCTORS_
#define _CURNN_FUNCTORS_

#include <math.h>
#include <cmath>

namespace curnn {

/*
 * ==========================================================================================================
 * Struct		: expFunctor
 *
 * Description	: Functor which provides the exp operation
 * ==========================================================================================================
 */
struct expFunctor {
	/*
	 * ======================================================================================================
	 * Function		: operator()
	 *
	 * Description	: Overloads the () operator to provide exponentiation
	 *
	 * Inputs		: value		: The value to exponentiate
	 *
	 * Outputs		:			: The exponentiation of the input
	 *
	 * Params		: dType		: Type of data of the input
	 */
	template <typename dType>
	__host__ __device__ dType operator() ( const dType& value ) {
		return exp( value );
	}
};

/*
 * ==========================================================================================================
 * Struct		: voidFunctor
 *
 * Description	: Functor which does noting. It can be used where generality is required, for example to
 *                provide a template fuctor, but allow the defulat to do nothing.
 * ==========================================================================================================
 */
struct voidFunctor {
	/*
	 * ======================================================================================================
	 * Function		: operator()
	 *
	 * Description	: Overloads the () operator to return the input
	 *
	 * Inputs		: value		: The value to operatre on
	 *
	 * Outputs		:			: The input value
	 *
	 * Params		: dType		: Type of data of the input
	 */
	template <typename dType>
	__host__ __device__ dType operator() ( const dType& value ) {
		return value;
	}
};

}	// Namespace curnn

#endif
