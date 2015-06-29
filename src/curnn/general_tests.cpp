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

/* NOTE: When using the vectorized instances in device code with templated functions, 
 *       you will need to first use a typedef, so for example if the template prameter 
 *       is dType then
 *
 *       template <typename dType> 
 *       void example_function( ... ) {
 *			
 *			typedef typename curnn::vectorizedType<dType, 2> vec2;
 *
 *			vec2 vectorized_dtype;
 *		}			
 *
 */

TEST( curnnTypes, CanDetermineVectorizedDoublesFromDouble ) {
	
	// Declare CUDA vectorized doubles
	// NOTE: No double4 exists for cuda
	double1 cudaDouble1;
	double2 cudaDouble2;

	// Declare a curnn vectorized doubles using type traits
	curnn::vectorizedType<double, 1>::vectType curnnDouble1;
	curnn::vectorizedType<double, 2>::vectType curnnDouble2;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaDouble1 ).name(), typeid( curnnDouble1 ).name() );
	EXPECT_EQ( typeid( cudaDouble2 ).name(), typeid( curnnDouble2 ).name() );
}

TEST( curnnTypes, CanDetermineVectorizedFloatFromFloat ) {
	// Declare CUDA vectorized doubles
	float1 cudaFloat1;
	float2 cudaFloat2;
	float4 cudaFloat4;

	// Declare curnn vectorized floats using type traits
	curnn::vectorizedType<float, 1>::vectType curnnFloat1;
	curnn::vectorizedType<float, 2>::vectType curnnFloat2;
	curnn::vectorizedType<float, 4>::vectType curnnFloat4;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaFloat1 ).name(), typeid( curnnFloat1 ).name() );
	EXPECT_EQ( typeid( cudaFloat2 ).name(), typeid( curnnFloat2 ).name() );
	EXPECT_EQ( typeid( cudaFloat4 ).name(), typeid( curnnFloat4 ).name() );
}

TEST( curnnTypes, CanDetermineVectorizedIntFromInt ) {
	// Declare CUDA vectorized ints
	int1 cudaInt1;
	int2 cudaInt2;
	int4 cudaInt4;

	// Declare curnn vectorized ints using type traits
	curnn::vectorizedType<int, 1>::vectType curnnInt1;
	curnn::vectorizedType<int, 2>::vectType curnnInt2;
	curnn::vectorizedType<int, 4>::vectType curnnInt4;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaInt1 ).name(), typeid( curnnInt1 ).name() );
	EXPECT_EQ( typeid( cudaInt2 ).name(), typeid( curnnInt2 ).name() );
	EXPECT_EQ( typeid( cudaInt4 ).name(), typeid( curnnInt4 ).name() );
}

TEST( curnnTypes, CanDetermineVectorizedUIntFromUInt ) {
	// Declare CUDA vectorized unsigned ints
	uint1 cudaUint1;
	uint2 cudaUint2;
	uint4 cudaUint4;

	// Declare curnn vectorized unsigned ints using type traits
	curnn::vectorizedType<uint, 1>::vectType curnnUint1;
	curnn::vectorizedType<uint, 2>::vectType curnnUint2;
	curnn::vectorizedType<uint, 4>::vectType curnnUint4;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaUint1 ).name(), typeid( curnnUint1 ).name() );
	EXPECT_EQ( typeid( cudaUint2 ).name(), typeid( curnnUint2 ).name() );
	EXPECT_EQ( typeid( cudaUint4 ).name(), typeid( curnnUint4 ).name() );
}

TEST( curnnTypes, CanDetermineVectorizedCharFromChar ) {
	// Declare CUDA vectorized chars
	char1 cudaChar1;
	char2 cudaChar2;
	char4 cudaChar4;

	// Declare curnn vectorized unsigned ints using type traits
	curnn::vectorizedType<char, 1>::vectType curnnChar1;
	curnn::vectorizedType<char, 2>::vectType curnnChar2;
	curnn::vectorizedType<char, 4>::vectType curnnChar4;
	
	// Check equivalence
	EXPECT_EQ( typeid( cudaChar1 ).name(), typeid( curnnChar1 ).name() );
	EXPECT_EQ( typeid( cudaChar2 ).name(), typeid( curnnChar2 ).name() );
	EXPECT_EQ( typeid( cudaChar4 ).name(), typeid( curnnChar4 ).name() );
}
