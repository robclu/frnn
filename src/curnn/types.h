/*
 *  Header file for cuRNN types.
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

#ifndef _CURNN_TYPES_
#define _CURNN_TYPES_

#include <cuda.h>

namespace curnn {
	/* 
	 * ======================================================================================================
	 * Struct		: vectorizeddType	 
	 * 
	 * Description	: Gets a vectorzed (2) version of dType. For example, if dType is a float this 
	 *                will then get float2, similarity for int or double.
	 * ======================================================================================================
	 */
	template <typename dType> struct vectorizedType;

	// Different specifications for the dTypes (will add more later)
	template <> struct vectorizedType<double> { typedef double2 vectType; };    // Double implementation
	template <> struct vectorizedType<float>  { typedef float2  vectType; };	// Float implementation 
	template <> struct vectorizedType<int>    { typedef int2    vectType; };    // Integer implementation
	/*
	 * ======================================================================================================
	 * Enum			: curnnError
	 *
	 * Description	: Enumerator for possible erorrs in curnn.
	 * ======================================================================================================
	 */
	enum curnnError {
		CURNN_ALLOC_ERROR = 1,
		CURNN_COPY_ERROR
	};
}

#endif
