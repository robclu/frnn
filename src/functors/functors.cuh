/*
 *  fastRNN functors. Define general operations which are normally passed 
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
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _FRNN_FUNCTORS_
#define _FRNN_FUNCTORS_

#include <math.h>
#include <cmath>

namespace frnn {
namespace functors {
   
// Note : Whike functors are technically structs, they behave as functions and hence the naming conventions of
//        functions are used for functors (camel case, start with lowercase letter) 

/*
 * ==========================================================================================================
 * Struct       : sigmoid
 * 
 * Description  : Functor which provides the sigmoid operation
 * ==========================================================================================================
 */
struct sigmoid {
    /*
     * ======================================================================================================
     * Function     : operator()
     * 
     * Decription   : Overlaods the () operator to provide the sigmoid operation 
     * 
     * Inputs       : x     : The value on which the sigmoid function should operate
     * 
     * Outputs      : The results of applying the sigmoid operation to the input
     * 
     * dType        : The type of data to use
     * ======================================================================================================
     */
    template <typename dType>
    __host__ __device__ dType operator() ( const dType& x ) const {
        return ( dType( 1 ) / ( dType( 1 ) + std::exp( dType( -1 ) * x ) ) );
    }
};

/*
 * ==========================================================================================================
 * Struct       : sigmoidDerivative
 * 
 * Description  : Functor which provides the derivative of the sigmoid sigmoid operation
 * ==========================================================================================================
 */
struct sigmoidDerivative {
    /*
     * ======================================================================================================
     * Function     : operator()
     * 
     * Decription   : Overlaods the () operator to provide the derivative of the sigmoid operation 
     * 
     * Inputs       : x     : The value at whhich the derivative should be evaluated
     * 
     * Outputs      : The results of the derivative evaluted at the input value
     * 
     * dType        : The type of data to use
     * ======================================================================================================
     */
    template <typename dType>
    __host__ __device__ dType operator() ( const dType& x ) const {
        return ( sigmoid( x ) * ( dType( 1 ) - sigmoid( x ) ) );
    }
};

/*
 * ==========================================================================================================
 * Struct		: exp
 *
 * Description	: Functor which provides the exp operation
 * ==========================================================================================================
 */
struct exp {
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
	 * ======================================================================================================
	 */
	template <typename dType>
	__host__ __device__ dType operator() ( const dType& value ) {
		return std::exp( value );
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
	 * ======================================================================================================
	 */
	template <typename dType>
	__host__ __device__ dType operator() ( const dType& value ) {
		return value;
	}
};

}   // Namespace functors
}	// Namespace frnn

#endif
