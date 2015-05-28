/*
 *  Layer header file for CUBDLRNN which defines the Cuda kernels that will 
 *  be used for the Layer class.
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

#ifndef LAYER_CUDA_KERNELS_INCLUDED
#define LAYER_CUDA_KERNELS_INCLUDED 

namespace cubdlrnn {

	/*
	 * ============================================================================	 
	 * Function		: UpdateLayer
	 *
	 * Description  : Device kernel that updates the layer by computing all the
	 *                cell gate values, and computing the cell output.
	 *
	 * Params       :
	 *
	 * NOTE         : This function should be called using many compute units
	 *                like would be done for a global kernel. But my GPU doesn't 
	 *                support  dynamic parallelism so this will have to wait.
	 * =============================================================================
	 */
	template<class Precision>
	void UpdateLater( Precision* xi , Precision* xh , Precision* cells,          // Inputs and cells
			          Precision* wxi, Precision* whi, Precision* wci    ) {      // Weights
		
		
	}
}

#endif 
