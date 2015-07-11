/*
 *  Cuda kernels for curand functions of other types (int for example) that 
 *  the curand library doesn't provide so that the math class works with 
 *  all types.
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
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _CURNN_RAND_KERNELS_
#define _CURNN_RAND_KERNELS_

#include <cuda.h>
#include <curand.h>

curandStatus_t curnnGenerateUniform( curandGenerator_t genenrator , int* outputPtr, size_t num ) {
    // Do nothing for the moment
}

#endif 
