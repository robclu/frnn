/*
 *  Header file for cuRNN matrix class.
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

#ifndef _CURNN_MATRIX_
#define _CURNN_MATRIX_

#include <vector>
#include <initializer_list>

using std::vector;
using std::initializer_list;

namespace curnn {

/*
 * ==========================================================================================================
 * Class		: matrix
 *
 * Description	: Provides functionality for matrix storage. Sets size and allocates data at compile time but
 *                can be resized.
 *
 * Params		: dType		: The data type for the matrix
 *				: r			: The numer of rows for the matrix
 *				: c			: The number of columns for the matrix
 * ==========================================================================================================
 */
template <typename dType, size_t r = 0, size_t c = 0>
class matrix {
	public:
		size_t			rows;		
		size_t			cols;
		vector<dType>	data;
	public:
		/*
		 * ==================================================================================================
		 * Function		: matrix (constructor)
		 *
		 * Description	: Sets the size of the matrix and intializes all the data to 0
		 * ==================================================================================================
		 */
		explicit matrix() :
			rows( r ), cols( c ), data( r * c, 0 ) {}
	
		/*
		 * ==================================================================================================
		 * Function		: matrix (constructor)
		 *
		 * Description	: Sets the size of the matrix and intializes all the data to that specified by the
		 *                list.
		 *
		 * Inputs		: list		: A list representing the data for the matrix
		 * ==================================================================================================
		 */
		matrix( initializer_list<dType> list ) :
			rows( r ), cols( c ), data( list ) {}

		/*
		 * ==================================================================================================
		 * Function		: resize
		 *
		 * Description	: Resizes the matrix.
		 *
		 * Inputs		: r_new		: The new number of rows for the matrix
		 *				: c_new		: The new number of columns for the matrix
		 * ==================================================================================================
		 */	
		void resize( const size_t r_new, const size_t c_new );
};

template<typename dType, size_t r, size_t c>
void curnn::matrix<dType, r, c>::resize( const size_t r_new, const size_t c_new ) {
	rows = r_new;
	cols = c_new;
	data.resize( r_new * c_new );
}

}	// Namespace curnn
#endif

