/*
 *  Header file for cuRNN math kernel functions.
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

#ifndef _CURNN_MATH_KERNELS_
#define _CURNN_MATH_KERNELS_

#include <cuda.h>

#include "../util/types.h"

namespace curnn {
	namespace math {

		/*
		 * ==================================================================================================     
		 * Function		: warpReduce
		 *
		 * Description	: Performs reduction sum (log(N) sum) within a warp.
		 *
		 * Inputs		: status	: Cublas status for determining correct completion of operation
		 *				: a			: Constant for multiplication 
		 *              : x			: Vector to multiply with a
		 * 
		 * Outputs/(I)	: y			: Vector used in a*X + Y, and where the result of a*X + Y is stored
		 * ==================================================================================================
		 */
		void axpy	
		template <typename dType>
		__device__void  
		template <typename dType, size_t blockSize>
		__global__ void reductionSum( dType* xIn, dType* xOut, size_t N ) {
			
			// Allocate shared memory
			extern __shared__dtype xShared[];

			int idx = threadIdx.x;
			int i   = blocj

		}
	}
}

#endif
