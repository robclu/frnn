/*
 *  Test file for cuRNN util functions.
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

#include <gtest/gtest.h>
#include <iostream>
#include <typeinfo>

#include "types.h"

TEST( curnnTypes, CanDetermineDouble2FromDouble ) {
	
	// Declare double2 variable using normal CUDA 
	double2 cudaDouble2;

	// Declare a curnn vectorized double using type traits
	curnn::vectorizedType<double>::vectType curnnDouble2;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaDouble2 ).name(), typeid( curnnDouble2 ).name() );
}

TEST( curnnTypes, CanDetermineFloat2FromFloat ) {
	
	// Declare float2 variable using normal CUDA 
	float2 cudaFloat2;

	// Declare a curnn vectorized float using type traits
	curnn::vectorizedType<float>::vectType curnnFloat2;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaFloat2 ).name(), typeid( curnnFloat2 ).name() );
}

TEST( curnnTypes, CanDetermineInt2FromInt ) {
	
	// Declare int2 variable using normal CUDA 
	int2 cudaInt2;

	// Declare a curnn vectorized int using type traits
	curnn::vectorizedType<int>::vectType curnnInt2;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaInt2 ).name(), typeid( curnnInt2 ).name() );
}
