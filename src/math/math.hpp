/*
 *  Header file for cuRNN math functions.
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
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.i
 *
 *
 *	======
 */

#ifndef __CURNN_MATH__
#define	__CURNN_MATH__

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>

#include "../util/errors.hpp"

namespace curnn {
	namespace math  {
		
		/*
		 * ==================================================================================================     
		 * Function		: axpy
		 *
		 * Description	: Performs simgle precision a*X + Y
		 *
		 * Inputs		: status	: Cublas status for determining correct completion of operation
		 *				: a			: Constant for multiplication 
		 *              : x			: Vector to multiply with a
		 * 
		 * Outputs/(I)	: y			: Vector used in a*X + Y, and where the result of a*X + Y is stored
		 * ==================================================================================================
		 */
		void axpy( cublasStatus_t& status, const float a, const std::vector<float>& x, std::vector<float>& y );	

		/*
		 * ==================================================================================================     
		 * Function		: axpy
		 *
		 * Description	: Performs double precision a*X + Y
		 *
		 * Inputs		: status	: Cublas status for determining correct completion of operation
		 *				: a			: Constant for multiplication 
		 *              : x			: Vector to multiply with a
		 * 
		 * Outputs/(I)	: y			: Vector used in a*X + Y, and where the result of a*X + Y is stored
		 * ==================================================================================================
		 */
		void axpy( cublasStatus_t& status, const double a, const std::vector<double>& x, std::vector<double>& y );	

	
		/*
		 * ==================================================================================================     
		 * Function		: softmax
		 *
		 * Description	: Performs the softmax function of a vector of floats x, which is 
		 *					
		 *				  softmax( x_i ) = exp( x_i ) / sum[ j=1 to J ]( exp( x_j )
		 *
		 * Inputs		: status	: Cublas status for determining correct completion of operation
		 *        
		 * Outputs/(I)  : x			: Vector to compute the softmax of, and to store the result in
		 * ==================================================================================================
		 */	 
		void softmax( cublasStatus_t& status, std::vector<float>& x );
	}
}

#endif
